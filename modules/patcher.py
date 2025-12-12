import torch
import gc
import logging
import comfy.model_patcher
import comfy.model_management as model_management

from .loader import VoxCPMLoader, LOADED_MODELS_CACHE

logger = logging.getLogger(__name__)

class VoxCPMPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher for managing VoxCPM models in ComfyUI.
    This class handles moving the model to the correct device (GPU) for inference
    and offloading it to free VRAM.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.cache_key = getattr(model, 'model_name', 'VoxCPM_Unknown')

    @property
    def is_loaded(self) -> bool:
        """Check if the model's core components are loaded."""
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'model') and self.model.model is not None

    def patch_model(self, device_to=None, *args, **kwargs):
        """
        Called by ComfyUI's model manager to load the model onto the GPU.
        """
        target_device = self.load_device if device_to is None else device_to
        
        if self.model.model is None:
            logger.info(f"Loading VoxCPM model '{self.model.model_name}' into RAM...")
            self.model.model = VoxCPMLoader.load_model(self.model.model_name)

        self.model.model.tts_model.to(target_device)

        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        """
        Called by ComfyUI's model manager to offload the model.
        """
        if unpatch_weights:
            if self.is_loaded:
                try:
                    self.model.model.tts_model.to(self.offload_device)
                except Exception:
                    pass
            
            self.model.model = None


            gc.collect()
            model_management.soft_empty_cache()

        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)