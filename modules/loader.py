import os
import torch
import logging
from huggingface_hub import snapshot_download

import folder_paths

from ..src.voxcpm.core import VoxCPM
from .model_info import AVAILABLE_VOXCPM_MODELS

logger = logging.getLogger(__name__)

LOADED_MODELS_CACHE = {}


class VoxCPMModelHandler(torch.nn.Module):
    """
    A lightweight handler for a VoxCPM model. It acts as a container
    that ComfyUI's ModelPatcher can manage, while the actual heavy model
    is loaded on demand.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = None  # This will hold the actual loaded VoxCPM instance
        self.size = int(2 * (1024**3))

class VoxCPMLoader:
    @staticmethod
    def load_model(model_name: str):
        """
        Loads a VoxCPM model, downloading it if necessary. Caches the loaded model instance.
        """
        if model_name in LOADED_MODELS_CACHE:
            logger.info(f"Using cached VoxCPM model instance: {model_name}")
            return LOADED_MODELS_CACHE[model_name]

        model_info = AVAILABLE_VOXCPM_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(AVAILABLE_VOXCPM_MODELS.keys())}")

        voxcpm_path = None

        if model_info["type"] == "local":
            voxcpm_path = model_info["path"]
            logger.info(f"Loading local model from: {voxcpm_path}")

        elif model_info["type"] == "official":
            base_tts_path = os.path.join(folder_paths.get_folder_paths("tts")[0])
            voxcpm_models_dir = os.path.join(base_tts_path, "VoxCPM")
            os.makedirs(voxcpm_models_dir, exist_ok=True)
            
            # Handle VoxCPM model download
            voxcpm_path = os.path.join(voxcpm_models_dir, model_name)
            if not os.path.exists(os.path.join(voxcpm_path, "pytorch_model.bin")):
                logger.info(f"Downloading official VoxCPM model '{model_name}'...")
                snapshot_download(
                    repo_id=model_info["repo_id"],
                    local_dir=voxcpm_path,
                    local_dir_use_symlinks=False,
                )

        if not voxcpm_path:
             raise RuntimeError(f"Could not determine path for model '{model_name}'")

        logger.info("Instantiating VoxCPM model...")
        model_instance = VoxCPM(voxcpm_model_path=voxcpm_path)

        LOADED_MODELS_CACHE[model_name] = model_instance
        return model_instance