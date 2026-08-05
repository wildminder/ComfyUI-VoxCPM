import os
import torch
import logging

import folder_paths

from ..src.voxcpm.core import VoxCPM
from ..src.voxcpm.model.voxcpm import LoRAConfig
from .model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS
from .downloader import DownloadManager, DownloadCancelledError, get_download_manager

logger = logging.getLogger(__name__)

LOADED_MODELS_CACHE = {}

# ── Default LoRA Configuration Constants ─────────────────────────────
# These values are optimized for single-speaker voice cloning:
# - r=32: Good balance between capacity and overfitting risk for voice cloning.
#   Higher rank (64+) is recommended for style/language adaptation.
# - alpha=16: Scaling factor (scaling = alpha / r = 0.5).
#   Usually set to r or 2*r. Adjust to control LoRA influence strength.
# - enable_lm=True: Essential for voice quality (language model backbone).
# - enable_dit=True: Essential for voice quality (diffusion transformer).
# - enable_proj=False: Projection layers rarely benefit from LoRA.
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_ENABLE_LM = True
DEFAULT_LORA_ENABLE_DIT = True
DEFAULT_LORA_ENABLE_PROJ = False


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
        
        # Get model size from config
        model_config = MODEL_CONFIGS.get(model_name, {})
        size_gb = model_config.get("size_gb", 2.5)
        # Allocate slightly more than model size for offloading calculations
        self.size = int((size_gb + 0.5) * (1024**3))


class VoxCPMLoader:
    @staticmethod
    def load_model(model_name: str, client_id: str = None):
        """
        Loads a VoxCPM model, downloading it if necessary. Caches the loaded model instance.

        Args:
            model_name: Name of the model to load.
            client_id: Optional ComfyUI client ID for targeted progress events.

        Raises:
            DownloadCancelledError: If the user cancels the download.
            RuntimeError: If the model path cannot be determined.
            ValueError: If the model name is not found.
        """
        if model_name in LOADED_MODELS_CACHE:
            logger.info(f"Using cached VoxCPM model instance: {model_name}")
            return LOADED_MODELS_CACHE[model_name]

        model_info = AVAILABLE_VOXCPM_MODELS.get(model_name)
        if not model_info:
            # Fall back to MODEL_CONFIGS if not in AVAILABLE_VOXCPM_MODELS
            model_info = MODEL_CONFIGS.get(model_name)
            if not model_info:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_CONFIGS.keys())}")

        voxcpm_path = None

        if model_info.get("type") == "local":
            voxcpm_path = model_info["path"]
            logger.info(f"Loading local model from: {voxcpm_path}")

        else:
            # Official model from HuggingFace
            base_tts_path = os.path.join(folder_paths.get_folder_paths("tts")[0])
            voxcpm_models_dir = os.path.join(base_tts_path, "VoxCPM")
            os.makedirs(voxcpm_models_dir, exist_ok=True)

            voxcpm_path = os.path.join(voxcpm_models_dir, model_name)

            has_bin = os.path.exists(os.path.join(voxcpm_path, "pytorch_model.bin"))
            has_safe = os.path.exists(os.path.join(voxcpm_path, "model.safetensors"))

            if not (has_bin or has_safe):
                logger.info(f"Downloading official VoxCPM model '{model_name}' from {model_info['repo_id']}...")
                download_manager = get_download_manager()
                download_manager.download_model(
                    model_name=model_name,
                    repo_id=model_info["repo_id"],
                    local_dir=voxcpm_path,
                    client_id=client_id,
                )

        if not voxcpm_path:
            raise RuntimeError(f"Could not determine path for model '{model_name}'")

        logger.info("Instantiating VoxCPM model...")

        default_lora_config = LoRAConfig(
            enable_lm=DEFAULT_LORA_ENABLE_LM,
            enable_dit=DEFAULT_LORA_ENABLE_DIT,
            enable_proj=DEFAULT_LORA_ENABLE_PROJ,
            r=DEFAULT_LORA_RANK,
            alpha=DEFAULT_LORA_ALPHA
        )

        model_instance = VoxCPM(
            voxcpm_model_path=voxcpm_path,
            enable_denoiser=False,
            optimize=False,
            lora_config=default_lora_config
        )

        LOADED_MODELS_CACHE[model_name] = model_instance
        return model_instance