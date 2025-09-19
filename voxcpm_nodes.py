import torch
import gc
import logging
import tempfile
import os
import soundfile as sf
from typing import cast

import comfy.model_management as model_management
from comfy_api.latest import ComfyExtension, io, ui

from .modules.model_info import AVAILABLE_VOXCPM_MODELS
from .modules.loader import VoxCPMModelHandler
from .modules.patcher import VoxCPMPatcher

logger = logging.getLogger(__name__)

# Global cache for patcher instances to avoid re-creation
VOXCPM_PATCHER_CACHE = {}

def get_available_devices():
    """Detects and returns a list of available PyTorch devices in order of preference."""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    
    # Check for DirectML on Windows
    try:
        import platform
        if platform.system() == "Windows" and hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
            devices.append("directml")
    except:
        pass

    # Check for HIP on AMD
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_name(0).lower().find('amd') != -1:
                 devices.append("hip")
        except:
            pass

    # Check for MPS on Apple Silicon
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

def save_temp_audio(audio_dict: dict | None) -> str | None:
    if not audio_dict or "waveform" not in audio_dict:
        return None
    waveform = cast(torch.Tensor, audio_dict["waveform"])
    sample_rate = cast(int, audio_dict["sample_rate"])
    if waveform.numel() == 0:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_path = tmp_file.name
    waveform_np = waveform.squeeze().cpu().numpy()
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=0)
    sf.write(temp_path, waveform_np, sample_rate)
    return temp_path


class VoxCPMNode(io.ComfyNode):
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())
        if not model_names:
            model_names.append("No models found. Please download VoxCPM-0.5B.")

        available_devices = get_available_devices()
        default_device = available_devices[0]

        return io.Schema(
            node_id="VoxCPM_TTS",
            display_name="VoxCPM TTS",
            category=cls.CATEGORY,
            description="Generate speech or clone voices using the VoxCPM model.",
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
        prompt_audio: dict | None = None,
        prompt_text: str | None = None,
    ) -> io.NodeOutput:
        
        is_cloning = prompt_audio is not None
        if is_cloning and not prompt_text:
            raise ValueError("Prompt text is required when providing prompt audio for voice cloning.")

        if device == "cuda":
            load_device = model_management.get_torch_device()
            offload_device = model_management.intermediate_device()
        else:
            load_device = torch.device("cpu")
            offload_device = torch.device("cpu")

        # Use a composite key for the patcher cache to include the device
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

        temp_audio_path = None
        try:
            if is_cloning:
                temp_audio_path = save_temp_audio(prompt_audio)
                if not temp_audio_path:
                    raise ValueError("Provided prompt audio is empty or invalid.")

            logger.info("Generating audio...")
            wav_array = voxcpm_model.generate(
                text=text,
                prompt_wav_path=temp_audio_path,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize_text,
                retry_badcase=True,
            )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_audio = {"waveform": output_tensor, "sample_rate": 16000}

            logger.info("Audio generation complete.")

            if force_offload:
                logger.info(f"Force offloading VoxCPM model '{model_name}' from VRAM...")
                patcher.unpatch_model(unpatch_weights=True)
                
            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)