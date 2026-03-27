"""
End-to-end inference pipeline.

Steps:
1. Load the MRI study directory (DICOM series).
2. Run tokenizer (VQ-VAE) to get series embeddings.
3. Run full PRIMA model (FullMRIModel) for classification, referral, and priority.

Config file (YAML or JSON) must contain:
- study_dir: path to the study directory
- output_dir: path to the output directory
- tokenizer_model_config: path to tokenizer config (or inline dict)
- prima_model_config: path to PRIMA config (or inline dict)

PRIMA config should contain either:
- full_model_ckpt: path to a saved FullMRIModel checkpoint (.pt), or
- component paths: clip_ckpt, diagnosis_heads_json, referral_heads_json, priority_head_ckpt

Paths in the PRIMA config file are resolved relative to that config file's directory.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on path so "tools" and other packages import correctly
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import SimpleITK as sitk
import torch
import json
import argparse
import logging
import yaml
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import gc
import psutil
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
from tools.DicomUtils import DicomUtils
from tools.models import ModelLoader
from tools.mrcommondataset import MrVoxelDataset
from tools.utilities import chartovec, convert_serienames_to_tensor, filtercoords
from Prima_training_and_evaluation.patchify import MedicalImagePatchifier


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    data_dir: str
    output_dir: str
    tokenizer_model_config: str
    prima_model_config: str
    batch_size: int = 1
    num_workers: int = 2
    max_tokens_per_chunk: int = 400
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create a PipelineConfig from a dictionary."""
        required_keys = ['data_dir', 'output_dir', 'tokenizer_model_config', 'prima_model_config']
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        return cls(**config_dict)


class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = PipelineConfig.from_dict(config)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f'Initializing pipeline with config: {self.config}')
        
        # Initialize models as None
        self.tokenizer_model = self.load_tokenizer_model()
        self.prima_model = self.load_full_prima_model()
        self.patchifier = MedicalImagePatchifier(in_dim = 256)

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.output_dir / 'pipeline.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        if self.tokenizer_model is not None:
            del self.tokenizer_model
            self.tokenizer_model = None
        if self.prima_model is not None:
            del self.prima_model
            self.prima_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_mri_study(self, study_dir) -> Tuple[List[sitk.Image], List[str]]:
        """
        Load the MRI study from the study directory.
        
        Returns:
            Tuple containing list of MRI images and series names
        """
        self.logger.info('Loading MRI study')
        try:
            self.mri_study, self.series_list, self.z_idx_list, self.study_description = DicomUtils.load_mri_study(study_dir)
            self.logger.info(f'Successfully loaded {len(self.mri_study)} series')
            return self.mri_study, self.series_list, self.z_idx_list, self.study_description
        except Exception as e:
            self.logger.error(f'Failed to load MRI study: {str(e)}')
            raise
    
    def load_tokenizer_model(self) -> torch.nn.Module:
        """Load the tokenizer model."""
        self.logger.info('Loading tokenizer model')
        try:
            # Handle path or dict; support JSON or YAML
            if isinstance(self.config.tokenizer_model_config, str):
                p = Path(self.config.tokenizer_model_config)
                with open(p, 'r') as f:
                    tokenizer_config = yaml.safe_load(f) if p.suffix in ('.yaml', '.yml') else json.load(f)
            else:
                tokenizer_config = self.config.tokenizer_model_config
            self.tokenizer_model = ModelLoader.load_vqvae_model(tokenizer_config)
            self.tokenizer_model = self.tokenizer_model.to(self.config.device)
            self.tokenizer_model.eval()
        except Exception as e:
            self.logger.error(f'Failed to load tokenizer model: {str(e)}')
            raise
        return self.tokenizer_model
    
    def load_full_prima_model(self) -> torch.nn.Module:
        """Load the full Prima model (FullMRIModel). Config path or dict; supports JSON/YAML."""
        self.logger.info('Loading Prima model')
        try:
            if isinstance(self.config.prima_model_config, str):
                config_path = Path(self.config.prima_model_config)
                with open(config_path, 'r') as f:
                    prima_config = yaml.safe_load(f) if config_path.suffix in ('.yaml', '.yml') else json.load(f)
                # Resolve relative paths in config (e.g. full_model_ckpt) relative to config file dir
                config_dir = config_path.resolve().parent
                if "full_model_ckpt" in prima_config:
                    p = Path(prima_config["full_model_ckpt"])
                    if not p.is_absolute():
                        prima_config = {**prima_config, "full_model_ckpt": str(config_dir / p)}
            else:
                prima_config = self.config.prima_model_config
            self.prima_model = ModelLoader.load_full_prima_model(prima_config)
            self.prima_model = self.prima_model.to(self.config.device)
            self.prima_model.eval()
        except Exception as e:
            self.logger.error(f'Failed to load Prima model: {str(e)}')
            raise
        return self.prima_model
    
    def create_dataset(self,
                       mri_study: List[sitk.Image],
                       z_indices: List[int]
                       ) -> DataLoader:
        """
        Create a dataset from the MRI study.
        
        Args:
            mri_study: List of MRI images
            z_indices: List of z-indexes for each MRI series
            
        Returns:
            DataLoader for the dataset
        """
        try:
            dataset = MrVoxelDataset(mri_study, z_indices)
            # Use num_workers=0 on GPU to avoid extra process memory and CUDA context issues
            num_workers = 0 if 'cuda' in str(self.config.device) else self.config.num_workers
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            return dataloader
        except Exception as e:
            self.logger.error(f'Failed to create dataset: {str(e)}')
            raise

    def prepare_prima_input(
        self,
        study_description,
        series_embeddings: List[torch.Tensor],
        series_names: List[str],
        all_ser_emb_meta: Optional[List[Dict[str, Any]]] = None,
        otsu_percentage = 5,
    ) -> Dict[str, Any]:
        """
        Prepare the input for the Prima model.
        
        Args:
            study_description: Study description in str.
            series_embeddings: Series embeddings after tokenization.
            series_names: Must be with series_embeddings.
        
        Returns:
            Dictionary containing model inputs
        """
        try:
            assert series_embeddings is not None and series_names is not None and len(series_embeddings) == len(series_names)

            coords = None

            if all_ser_emb_meta is not None:
                coords = []
                new_series_embeddings = []
                for i, ser_emb_meta in enumerate(all_ser_emb_meta):
                    for percent in range(otsu_percentage, -1,-1):
                        embs,embspos,_ = filtercoords(ser_emb_meta, percent, series_embeddings[i])
                        print("Using otsu percentage: " + str(percent) + " for series: " + series_names[i]+" before filtering: "+str(len(series_embeddings[i]))+" after filtering: "+str(len(embs)))
                        if len(embspos) > 25:
                            break
                    new_series_embeddings.append(embs)
                    coords.append(embspos)
                series_embeddings = new_series_embeddings

            # Create lengths tensors
            study_lens = torch.tensor([len(series_embeddings)], dtype=torch.long)
            serie_lenss = torch.tensor([len(v) for v in series_embeddings], dtype=torch.long).unsqueeze(0)


            # Prepare visual input for HierViT: patchify and pad
            patched = self.patchifier(series_embeddings, coords = coords) # if has otsu-filtered coordinates, replace this with otsu coordinates
            max_len = serie_lenss.max()
            visuals = []
            for img in patched:
                sizes = list(img.shape)
                h = sizes[0]
                img_pad_len = max_len - len(img)
                sizes[0] = img_pad_len
                img_pad = torch.zeros(sizes)
                visuals.append(torch.cat([img, img_pad], dim=0).unsqueeze(0))
            

            
            # Create series name tensors
            # The model expects serienames as a list where each element is a tensor of shape [num_series, max_chars]
            # For a single study, we need to stack the series name tensors
            seriename_tensors = [chartovec(name) for name in series_names]
            # Find max length and pad
            max_seriename_len = max(len(t) for t in seriename_tensors)
            num_series = len(seriename_tensors)
            serienames_tensor = torch.zeros(num_series, max_seriename_len, dtype=torch.long)
            for i, t in enumerate(seriename_tensors):
                serienames_tensor[i, :len(t)] = t
            serienames = serienames_tensor.unsqueeze(0)
            
            # Create study description tensor
            study_desc = chartovec(study_description).unsqueeze(0)
            
        
            return {
                'visual': visuals,
                'lens': study_lens,
                'lenss': serie_lenss,
                'hash': ["study_0"],
                'serienames': serienames,
                'studydescription': study_desc
            }
        except Exception as e:
            self.logger.error(f'Failed to prepare Prima input: {str(e)}')
            raise

    def run_tokenizer_model(
        self,
        mri_study: List[sitk.Image],
        z_indices: List[int],
        series_names: Optional[List[str]] = None,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Run the tokenizer model to get series embeddings.
        On per-series failure, logs and skips that series so the pipeline can continue.

        Args:
            mri_study: List of MRI images
            z_indices: List of z-indexes for each MRI series
            series_names: Optional list of series names; if provided, returned names match embeddings (failed series omitted).

        Returns:
            (series_embeddings, series_names_for_embeddings). If series_names was not provided, second is None.
        """
        self.logger.info('Running tokenizer model')
        self.tokenizer_model.to(self.config.device)
        self.tokenizer_model.eval()
        dataloader = self.create_dataset(mri_study, z_indices)
        series_embeddings = []
        filtered_names = [] if series_names is not None else None
        all_ser_emb_meta = []
        try:
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dataloader, desc="Processing series")):
                    self.logger.info('Processing series: ' + str(idx))
                    series_name = (series_names[idx] if series_names is not None else f"series_{idx}")
                    try:
                        batch, ser_emb_meta = batch
                        if batch.view(-1).size() == 0:
                            raise Exception("No tokens found for "+series_name)
                        token_list = []
                        tokens = batch[0]
                        num_tokens = tokens.shape[0]
                        if num_tokens > 5000:
                            raise Exception("Too many tokens: " + str(num_tokens) + " for "+series_name)

                        # VQ-VAE encoder expects (B, C, D, H, W) with C=1; tokens are (N, D, H, W)
                        num_chunks = (num_tokens + self.config.max_tokens_per_chunk - 1) // self.config.max_tokens_per_chunk
                        for j in range(num_chunks):
                            start_idx = j * self.config.max_tokens_per_chunk
                            end_idx = min((j + 1) * self.config.max_tokens_per_chunk, num_tokens)
                            # Add channel dim: (N, D, H, W) -> (N, 1, D, H, W)
                            chunk = tokens[start_idx:end_idx].unsqueeze(1)
                            token_list.append(chunk)

                        embeddings = [
                            self.tokenizer_model.encode(chunk.to(self.config.device)).detach().cpu()
                            for chunk in token_list
                        ]
                        series_embedding = torch.cat(embeddings, dim=0)
                        series_embeddings.append(series_embedding)
                        if filtered_names is not None:
                            filtered_names.append(series_name)
                        all_ser_emb_meta.append(ser_emb_meta)
                    except Exception as e:
                        self.logger.warning(
                            "Skipping series index=%s name=%s due to error (review later): %s",
                            idx, series_name, str(e),
                            exc_info=True,
                        )
                        continue

            self.logger.info(
                'Tokenizer model run complete: %d/%d series succeeded',
                len(series_embeddings), len(mri_study),
            )
            return series_embeddings, (filtered_names if series_names is not None else None), all_ser_emb_meta
        finally:
            self.tokenizer_model.to('cpu') # Free up CUDA memory by moving to CPU instead of deleting the object
            torch.cuda.empty_cache()
            gc.collect()

    def run_prima_model(
        self,
        study_id: str,
        study_description: str,
        series_embeddings: Optional[List[torch.Tensor]] = None,
        series_names: Optional[List[str]] = None,
        all_ser_emb_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the Prima model to get final predictions.
        
        Args:
            study_id: Accession of the study for storing the results.
            study_description: Study description in str
            series_embeddings: If provided, reuse these (avoids re-running tokenizer; saves memory).
            series_names: If provided with series_embeddings, reuse these.
        
        Returns:
            Dictionary containing model predictions
        """
        self.logger.info('Running Prima model')
        
        def move_to_device(obj, device):
            """Recursively move tensors to device."""
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(item, device) for item in obj]
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            else:
                return obj

        try:
            # Free tokenizer and reclaim GPU memory before loading Prima (single-GPU friendly, e.g. L40S)
            self.tokenizer_model.to('cpu')
            if 'cuda' in str(self.config.device):
                torch.cuda.empty_cache()
            gc.collect()

            # Prepare input for Prima model (reuse embeddings if provided)
            prima_input = self.prepare_prima_input(
                study_description=study_description,
                series_embeddings=series_embeddings,
                series_names=series_names,
                all_ser_emb_meta=all_ser_emb_meta,
            )
            prima_input = move_to_device(prima_input, self.config.device)

            # Load model if not already loaded
            if self.prima_model is None:
                self.prima_model = self.load_full_prima_model()
            if hasattr(self.prima_model, 'make_no_flashattn'):
                self.prima_model.make_no_flashattn()

            # Run inference (autocast for memory and speed on L40S)
            device_type = 'cuda' if 'cuda' in str(self.config.device) else 'cpu'
            with torch.no_grad():
                if device_type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        predictions = self.prima_model(prima_input, inference_only_once = True)
                else:
                    predictions = self.prima_model(prima_input)

            # Free input from GPU before serialization
            del prima_input
            if 'cuda' in str(self.config.device):
                torch.cuda.empty_cache()

            # Convert tensors to lists for JSON serialization
            def tensor_to_serializable(obj):
                """Recursively convert tensors to lists/numpy arrays."""
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()
                elif isinstance(obj, dict):
                    return {k: tensor_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [tensor_to_serializable(item) for item in obj]
                else:
                    return obj
            
            predictions_serializable = tensor_to_serializable(predictions)

            # Save predictions with study_id prefix (study_id = last component of study_dir)
            output_path = (self.output_dir / f"{study_id}_predictions.json").resolve()
            with open(output_path, 'w') as f:
                json.dump(predictions_serializable, f, indent=2)
            self.logger.info(f'Predictions saved to {output_path}')
            return predictions
        except Exception as e:
            self.logger.error(f'Failed to run Prima model: {str(e)}')
            raise
        # finally:
        #     self._cleanup()


if __name__=="__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='End-to-end inference pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    args = parser.parse_args()

    try:
        # Load config (JSON or YAML)
        config_path = Path(args.config)
        with open(config_path, 'r') as f:
            if config_path.suffix in ('.yaml', '.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Initialize pipeline
        pipeline = Pipeline(config)
        
        # Run pipeline steps
        pipeline.logger.info("Starting pipeline execution")
        
        # Step 1: Load MRI study
        # Base accession directory has a subfolder which contains the study
        for acc_dir in Path(pipeline.config.data_dir).iterdir():
            if not acc_dir.is_dir():
                continue
            acc_path = os.path.join(pipeline.config.data_dir, acc_dir.name)
            study_dir = [Path(os.path.join(acc_path, item.name))
                         for item in Path(acc_path).iterdir() if item.is_dir()
                         ][0] # Selecting 0th index assuming there would be only one study dir
            pipeline.logger.info(f"Processing {study_dir}")
            mri_study, series_names, z_indices, study_description = pipeline.load_mri_study(study_dir)
            pipeline.logger.info(f"Loaded {len(mri_study)} series from study")

            # Step 2: Run tokenizer model (pass series_names so failed series are skipped and names stay in sync)
            series_embeddings, series_names_for_embeddings, all_ser_emb_meta = pipeline.run_tokenizer_model(
                mri_study, z_indices, series_names=series_names
            )
            if series_names_for_embeddings is not None:
                series_names = series_names_for_embeddings
            if not series_embeddings:
                raise RuntimeError("No series could be tokenized; pipeline cannot continue. Check logs for skipped series.")
            pipeline.logger.info(f"Generated embeddings for {len(series_embeddings)} series")

            # Step 3: Run Prima model (pass embeddings to avoid re-running tokenizer; saves GPU memory)
            predictions = pipeline.run_prima_model(
                study_id=acc_dir.name,
                study_description=study_description,
                series_embeddings=series_embeddings,
                series_names=series_names,
                all_ser_emb_meta=all_ser_emb_meta,
            )
            pipeline.logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise