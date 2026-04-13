"""
Shared generation utilities for VoxCPM nodes.

This module contains common functions for model loading, audio processing,
and generation to avoid code duplication across nodes.
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Any

import comfy.model_management as model_management
from comfy.utils import ProgressBar
import folder_paths

from .loader import VoxCPMModelHandler
from .patcher import VoxCPMPatcher
from .utils import VOXCPM_PATCHER_CACHE, set_seed

logger = logging.getLogger(__name__)


def load_voxcpm_model(
    model_name: str,
    device: str = "cuda",
    force_reload: bool = False
) -> Tuple[VoxCPMPatcher, Any]:
    """
    Load or retrieve cached VoxCPM model.
    
    Args:
        model_name: Name of the model to load
        device: Device to load on ("cuda" or "cpu")
        force_reload: Force reload even if cached
        
    Returns:
        Tuple of (patcher, model)
        
    Raises:
        RuntimeError: If model fails to load
    """
    # Setup device
    if device == "cuda":
        load_device = model_management.get_torch_device()
        offload_device = model_management.intermediate_device()
    else:
        load_device = torch.device("cpu")
        offload_device = torch.device("cpu")
    
    cache_key = f"{model_name}_{device}"
    
    if cache_key not in VOXCPM_PATCHER_CACHE or force_reload:
        handler = VoxCPMModelHandler(model_name)
        patcher = VoxCPMPatcher(
            handler,
            load_device=load_device,
            offload_device=offload_device,
            size=handler.size
        )
        VOXCPM_PATCHER_CACHE[cache_key] = patcher
        logger.debug(f"Created new patcher for {model_name}")
    
    patcher = VOXCPM_PATCHER_CACHE[cache_key]
    model_management.load_model_gpu(patcher)
    model = patcher.model.model
    
    if not model:
        raise RuntimeError(f"Failed to load VoxCPM model '{model_name}'. Check logs for details.")
    
    return patcher, model


def extract_audio_tensor(
    audio_input: Optional[Dict],
    name: str = "audio"
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Extract waveform tensor and sample rate from ComfyUI audio input.
    
    Args:
        audio_input: ComfyUI audio dictionary with 'waveform' and 'sample_rate'
        name: Name for error messages
        
    Returns:
        Tuple of (waveform tensor, sample rate) or (None, None) if input is None
        
    Raises:
        ValueError: If audio format is invalid or empty
    """
    if audio_input is None:
        return None, None
    
    if not isinstance(audio_input, dict):
        raise ValueError(f"{name}: Expected dict, got {type(audio_input).__name__}")
    
    if 'waveform' not in audio_input or 'sample_rate' not in audio_input:
        raise ValueError(f"{name}: Missing 'waveform' or 'sample_rate' keys")
    
    waveform = audio_input['waveform']
    sample_rate = audio_input['sample_rate']
    
    # Remove batch dimension if present [1, C, T] -> [C, T]
    if waveform.dim() == 3:
        waveform = waveform[0]
    
    if waveform.numel() == 0:
        raise ValueError(f"{name}: Audio is empty")
    
    return waveform, sample_rate


def apply_lora_if_needed(
    model: Any,
    lora_name: str
) -> None:
    """
    Apply LoRA weights if specified, otherwise disable LoRA.
    
    Args:
        model: VoxCPM model instance
        lora_name: Name of LoRA to apply (or "None" to disable)
        
    Raises:
        FileNotFoundError: If LoRA file not found
        RuntimeError: If LoRA fails to load
    """
    if lora_name == "None":
        model.set_lora_enabled(False)
        return
    
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        raise FileNotFoundError(f"LoRA file not found: {lora_name}")
    
    try:
        model.load_lora(lora_path)
        model.set_lora_enabled(True)
        logger.info(f"Applied LoRA: {lora_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA '{lora_name}': {e}")


def generate_audio(
    model: Any,
    text: str,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    min_len: int = 2,
    max_len: int = 2048,
    normalize: bool = True,
    trim_silence_vad: bool = False,
    temperature: float = 1.0,
    sway_sampling_coef: float = 1.0,
    use_cfg_zero_star: bool = True,
    retry_badcase: bool = True,
    retry_max_attempts: int = 3,
    retry_threshold: float = 6.0,
    prompt_text: Optional[str] = None,
    prompt_waveform: Optional[torch.Tensor] = None,
    prompt_sample_rate: Optional[int] = None,
    reference_waveform: Optional[torch.Tensor] = None,
    reference_sample_rate: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Generate audio using VoxCPM model.
    
    Args:
        model: VoxCPM model instance
        text: Text to synthesize
        cfg_value: Guidance scale for generation
        inference_timesteps: Number of diffusion steps
        min_len: Minimum audio token length
        max_len: Maximum audio token length
        normalize: Whether to normalize text
        trim_silence_vad: Whether to trim silence using VAD
        temperature: Sampling temperature
        sway_sampling_coef: Sway sampling coefficient
        use_cfg_zero_star: Whether to use CFG-Zero* optimization
        retry_badcase: Whether to retry on bad cases
        retry_max_attempts: Maximum retry attempts
        retry_threshold: Threshold for triggering retry
        prompt_text: Transcript for prompt audio
        prompt_waveform: Prompt audio waveform tensor
        prompt_sample_rate: Prompt audio sample rate
        reference_waveform: Reference audio waveform tensor
        reference_sample_rate: Reference audio sample rate
        
    Returns:
        Tuple of (output audio tensor [1, 1, T], sample rate)
        
    Raises:
        RuntimeError: If generation fails
    """
    try:
        # Estimate total generation steps based on text length: ~1.2 steps per character + base overhead
        text_length = len(text)
        estimated_total_steps = int(text_length * 1.2 + 20)
        estimated_total_steps = min(estimated_total_steps, max_len)
        
        # Initialize ComfyUI native progress bar
        pbar = ProgressBar(estimated_total_steps)
        
        def progress_update(current_step, total_steps):
            pbar.update_absolute(current_step)
            # Check for user cancellation
            model_management.throw_exception_if_processing_interrupted()
        
        wav_array = model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_waveform=prompt_waveform,
            prompt_sample_rate=prompt_sample_rate,
            reference_waveform=reference_waveform,
            reference_sample_rate=reference_sample_rate,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            min_len=min_len,
            max_len=max_len,
            normalize=normalize,
            trim_silence_vad=trim_silence_vad,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_max_attempts,
            retry_badcase_ratio_threshold=retry_threshold,
            temperature=temperature,
            sway_sampling_coef=sway_sampling_coef,
            use_cfg_zero_star=use_cfg_zero_star,
            denoise=False,
            progress_callback=progress_update
        )
        
        # Convert numpy array to tensor with proper shape [1, 1, T]
        output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
        sample_rate = model.tts_model.sample_rate
        
        return output_tensor, sample_rate
        
    except model_management.InterruptProcessingException:
        # Re-raise interrupt exceptions directly - ComfyUI handles these properly
        raise
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise RuntimeError(f"Generation failed: {e}")


def build_final_text(text: str, control_instruction: Optional[str] = None) -> str:
    """
    Build final text with optional control instruction for VoxCPM2 voice design.
    
    Args:
        text: Base text to synthesize
        control_instruction: Voice design instruction (e.g., "warm female voice")
        
    Returns:
        Final text with control instruction prepended if provided
    """
    if control_instruction and control_instruction.strip():
        return f"({control_instruction.strip()}){text}"
    return text


def validate_voxcpm2_features(
    model_name: str,
    model_configs: dict,
    reference_audio: Optional[Dict],
    control_instruction: Optional[str]
) -> Tuple[bool, Optional[str], bool]:
    """
    Validate that VoxCPM2-only features are used with VoxCPM2 models.

    For non-fatal incompatibilities (e.g., reference_audio with VoxCPM1.5),
    logs a warning and returns flags to gracefully ignore the incompatible input.

    Args:
        model_name: Name of the model being used
        model_configs: Dictionary of model configurations
        reference_audio: Reference audio input (VoxCPM2 only)
        control_instruction: Control instruction (VoxCPM2 only)

    Returns:
        Tuple of (is_voxcpm2, warning_message, ignore_reference)
        is_voxcpm2: True if model is VoxCPM2
        warning_message: Warning message for UI display (if any)
        ignore_reference: True if reference_audio should be ignored (VoxCPM1.5 case)
    """
    from .utils import is_voxcpm2_model

    is_v2 = is_voxcpm2_model(model_name, model_configs)

    warning = None
    ignore_reference = False

    # Handle reference_audio with non-VoxCPM2 model - graceful degradation
    if reference_audio is not None and not is_v2:
        warning = "Reference audio is only supported with VoxCPM2 models. Ignoring reference_audio for VoxCPM1.5. Use prompt_audio + prompt_text for voice cloning."
        ignore_reference = True

    # Handle control_instruction with non-VoxCPM2 model - graceful degradation
    if control_instruction and not is_v2:
        warning = "Voice design is only supported with VoxCPM2 models. Ignoring voice_design for VoxCPM1.5."

    return is_v2, warning, ignore_reference


def validate_prompt_pairing(
    prompt_audio: Optional[Dict],
    prompt_text: Optional[str]
) -> None:
    """
    Validate that prompt audio and text are properly paired.
    
    Args:
        prompt_audio: Prompt audio input
        prompt_text: Transcript for prompt audio
        
    Raises:
        ValueError: If prompt_audio provided without prompt_text
    """
    if prompt_audio is not None and prompt_text is None:
        raise ValueError("prompt_text is required when providing prompt_audio.")
