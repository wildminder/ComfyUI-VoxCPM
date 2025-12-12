import torch
import gc
import logging
import tempfile
import os
import soundfile as sf
from typing import cast, List, Optional

import comfy.model_management as model_management
from comfy_api.latest import ComfyExtension, io, ui

from .modules.model_info import AVAILABLE_VOXCPM_MODELS
from .modules.loader import VoxCPMModelHandler
from .modules.patcher import VoxCPMPatcher

logger = logging.getLogger(__name__)

VOXCPM_PATCHER_CACHE = {}

def get_available_devices():
    """Detects and returns a list of available PyTorch devices in order of preference."""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    
    try:
        import platform
        if platform.system() == "Windows" and hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
            devices.append("directml")
    except:
        pass

    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_name(0).lower().find('amd') != -1:
                 devices.append("hip")
        except:
            pass

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
        
    devices.append("cpu")
    return devices

def set_seed(seed: int):
    if seed < 0:
        seed = torch.randint(0, 0xFFFFFFFFFFFFFFFF, (1,)).item()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class VoxCPMNode(io.ComfyNode):
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())
        if not model_names:
            model_names.append("No models found. Please download VoxCPM1.5.")

        available_devices = get_available_devices()
        default_device = available_devices[0]

        return io.Schema(
            node_id="VoxCPM_TTS",
            display_name="VoxCPM TTS",
            category=cls.CATEGORY,
            description="Generate speech or clone voices using the VoxCPM model (v1.5).",
            inputs=[
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM model to use."),
                io.String.Input("text", multiline=True, default="VoxCPM is an innovative TTS model designed to generate highly expressive speech.", tooltip="Text to synthesize. Each line is processed as a separate chunk."),
                io.Audio.Input("prompt_audio", optional=True, tooltip="Reference audio for voice cloning."),
                io.String.Input("prompt_text", multiline=True, optional=True, tooltip="The transcript of the reference audio. Required for voice cloning."),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="Guidance scale. Higher values adhere more to the prompt but may sound less natural."),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps. Higher values may improve quality but are slower."),
                io.Boolean.Input("normalize_text", default=True, label_on="Normalize", label_off="Raw", tooltip="Enable text normalization (recommended for general text)."),
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Seed for reproducibility. -1 for random."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto-Manage", tooltip="Force the model to be offloaded from VRAM after generation."),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="Device to run inference on. Defaults to the best available."),
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="Max retry attempts for bad cases (e.g., babbling). Set to 0 to disable retrying."),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="Audio/text length ratio to trigger a retry. Increase for very slow speakers."),
            ],
            outputs=[
                io.Audio.Output(display_name="Generated Audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        device: str,
        text: str,
        cfg_value: float,
        inference_timesteps: int,
        normalize_text: bool,
        seed: int,
        force_offload: bool,
        retry_max_attempts: int,
        retry_threshold: float,
        prompt_audio: Optional[io.Audio.Type] = None,
        prompt_text: Optional[str] = None,
    ) -> io.NodeOutput:
        
        is_cloning = prompt_audio is not None
        
        if is_cloning and prompt_text is None:
            raise ValueError("Prompt text is required when providing prompt audio for voice cloning.")

        if device == "cuda":
            load_device = model_management.get_torch_device()
            offload_device = model_management.intermediate_device()
        else:
            load_device = torch.device("cpu")
            offload_device = torch.device("cpu")

        cache_key = f"{model_name}_{device}"
        if cache_key not in VOXCPM_PATCHER_CACHE:
            handler = VoxCPMModelHandler(model_name)
            patcher = VoxCPMPatcher(
                handler,
                load_device=load_device,
                offload_device=offload_device,
                size=handler.size
            )
            VOXCPM_PATCHER_CACHE[cache_key] = patcher
        
        patcher = VOXCPM_PATCHER_CACHE[cache_key]
        
        model_management.load_model_gpu(patcher)
        voxcpm_model = patcher.model.model

        if not voxcpm_model:
            raise RuntimeError(f"Failed to load VoxCPM model '{model_name}'. Check logs for details.")

        set_seed(seed)
        enable_retry = retry_max_attempts > 0

        prompt_waveform = None
        prompt_sample_rate = None
        
        if is_cloning:
            if isinstance(prompt_audio, dict) and 'waveform' in prompt_audio and 'sample_rate' in prompt_audio:
                prompt_waveform = prompt_audio['waveform']
                
                if prompt_waveform.dim() == 3:
                    prompt_waveform = prompt_waveform[0]
                elif prompt_waveform.dim() == 2:
                    pass
                
                prompt_sample_rate = prompt_audio['sample_rate']
                
                if prompt_waveform.numel() == 0:
                    raise ValueError("Provided prompt audio is empty.")
            else:
                raise ValueError("Provided prompt audio format is invalid.")

        try:
            wav_array = voxcpm_model.generate(
                text=text,
                prompt_text=prompt_text,
                prompt_waveform=prompt_waveform,
                prompt_sample_rate=prompt_sample_rate,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize_text,
                retry_badcase=enable_retry,
                retry_badcase_max_times=retry_max_attempts,
                retry_badcase_ratio_threshold=retry_threshold,
                denoise=False # Explicitly disable denoiser for Tensor inputs
            )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            
            output_sr = voxcpm_model.tts_model.sample_rate
            
            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            logger.info("Audio generation complete.")

            if force_offload:
                logger.info(f"Force offloading VoxCPM model '{model_name}' from VRAM...")
                patcher.unpatch_model(unpatch_weights=True)
                
            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise e

class VoxCPMExtension(ComfyExtension):
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        return [VoxCPMNode]

async def comfy_entrypoint() -> VoxCPMExtension:
    return VoxCPMExtension()