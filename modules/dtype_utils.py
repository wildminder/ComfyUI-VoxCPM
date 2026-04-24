"""
Dtype utilities for VoxCPM nodes.

This module provides unified dtype (precision) handling using ComfyUI's
model_management functions for consistent precision selection across devices.
"""

import torch
from typing import Optional, Tuple, List

# Lazy import to avoid circular dependencies
_model_management = None


def _get_model_management():
    """Lazy import of ComfyUI's model_management module."""
    global _model_management
    if _model_management is None:
        import comfy.model_management as mm
        _model_management = mm
    return _model_management


# Dtype option constants
DTYPE_AUTO = "auto"
DTYPE_BF16 = "bf16"
DTYPE_FP16 = "fp16"
DTYPE_FP32 = "fp32"


def get_dtype_options() -> List[str]:
    """Get list of dtype options for node dropdown.
    
    Returns:
        List of dtype option strings for UI dropdown
    """
    return [DTYPE_AUTO, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32]


def get_dtype_display_name(dtype_option: str) -> str:
    """Get human-readable display name for a dtype option.
    
    Args:
        dtype_option: Dtype option string (e.g., "auto", "bf16")
        
    Returns:
        Human-readable dtype name
    """
    display_names = {
        DTYPE_AUTO: "Auto (device optimal)",
        DTYPE_BF16: "BF16 (Bfloat16)",
        DTYPE_FP16: "FP16 (Float16)",
        DTYPE_FP32: "FP32 (Float32)",
    }
    return display_names.get(dtype_option, dtype_option)


def resolve_dtype(
    dtype_option: str,
    device: torch.device = None
) -> torch.dtype:
    """Resolve dtype option to actual torch.dtype.
    
    Uses ComfyUI's should_use_bf16() and should_use_fp16() for 'auto' mode.
    
    Args:
        dtype_option: Dtype option string ("auto", "bf16", "fp16", "fp32")
        device: torch.device to check for compatibility. If None, uses default device.
        
    Returns:
        torch.dtype instance (torch.bfloat16, torch.float16, or torch.float32)
    """
    mm = _get_model_management()
    
    if device is None:
        device = mm.get_torch_device()
    
    if dtype_option == DTYPE_AUTO:
        # Use ComfyUI's automatic dtype selection
        if mm.should_use_bf16(device):
            return torch.bfloat16
        elif mm.should_use_fp16(device):
            return torch.float16
        else:
            return torch.float32
    
    elif dtype_option == DTYPE_BF16:
        return torch.bfloat16
    
    elif dtype_option == DTYPE_FP16:
        return torch.float16
    
    elif dtype_option == DTYPE_FP32:
        return torch.float32
    
    else:
        # Unknown option, fall back to auto
        return resolve_dtype(DTYPE_AUTO, device)


def is_dtype_supported(dtype: torch.dtype, device: torch.device = None) -> bool:
    """Check if a dtype is supported on the given device.
    
    Args:
        dtype: torch.dtype to check
        device: torch.device to check. If None, uses default device.
        
    Returns:
        True if the dtype is supported on the device
    """
    mm = _get_model_management()
    
    if device is None:
        device = mm.get_torch_device()
    
    # FP32 is always supported
    if dtype == torch.float32:
        return True
    
    # Check BF16 support
    if dtype == torch.bfloat16:
        return mm.should_use_bf16(device)
    
    # Check FP16 support
    if dtype == torch.float16:
        return mm.should_use_fp16(device)
    
    # Unknown dtype
    return False


def get_dtype_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to string representation.
    
    Args:
        dtype: torch.dtype instance
        
    Returns:
        String representation (e.g., "bf16", "fp16", "fp32")
    """
    if dtype == torch.bfloat16:
        return DTYPE_BF16
    elif dtype == torch.float16:
        return DTYPE_FP16
    elif dtype == torch.float32:
        return DTYPE_FP32
    else:
        return str(dtype)


def get_optimal_dtype(device: torch.device = None) -> Tuple[torch.dtype, str]:
    """Get optimal dtype for the given device.
    
    Uses ComfyUI's automatic selection logic.
    
    Args:
        device: torch.device to check. If None, uses default device.
        
    Returns:
        Tuple of (torch.dtype, dtype_option_string)
    """
    mm = _get_model_management()
    
    if device is None:
        device = mm.get_torch_device()
    
    if mm.should_use_bf16(device):
        return torch.bfloat16, DTYPE_BF16
    elif mm.should_use_fp16(device):
        return torch.float16, DTYPE_FP16
    else:
        return torch.float32, DTYPE_FP32


def cast_tensor(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    non_blocking: bool = True
) -> torch.Tensor:
    """Cast a tensor to the specified dtype.
    
    Args:
        tensor: Input tensor
        dtype: Target dtype
        non_blocking: Whether to use non-blocking transfer
        
    Returns:
        Tensor cast to target dtype
    """
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype=dtype, non_blocking=non_blocking)


def cast_model_to_dtype(
    model: torch.nn.Module,
    dtype: torch.dtype
) -> torch.nn.Module:
    """Cast a model and all its submodules to the specified dtype.

    This function explicitly casts all parameters and buffers to ensure
    consistent dtype across the entire model, including submodules that
    may have been loaded with different dtypes (e.g., AudioVAE).

    Note: AudioVAE is kept in FP32 as it's designed to work with float32
    input/output and the decode() method explicitly casts to float32.

    Args:
        model: PyTorch model to cast
        dtype: Target dtype

    Returns:
        Model cast to target dtype (in-place operation)
    """
    # Special handling for VoxCPM models: keep AudioVAE in FP32
    # The AudioVAE decode() method expects float32 input and outputs float32
    if hasattr(model, 'audio_vae'):
        # Cast AudioVAE to FP32 explicitly
        model.audio_vae = model.audio_vae.to(torch.float32)

    # Cast the rest of the model to the target dtype
    model = model.to(dtype=dtype)

    # Re-ensure AudioVAE stays in FP32 after the model-wide cast
    if hasattr(model, 'audio_vae'):
        model.audio_vae = model.audio_vae.to(torch.float32)

    # Additionally, explicitly cast all parameters to handle edge cases
    # where some submodules may have been loaded with different dtypes
    for name, param in model.named_parameters():
        # Skip AudioVAE parameters - keep them in FP32
        if 'audio_vae' in name:
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        elif param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)

    # Also cast all buffers (e.g., running mean/var in batch norm)
    for name, buffer in model.named_buffers():
        # Skip AudioVAE buffers - keep them in FP32
        if 'audio_vae' in name:
            if buffer.dtype != torch.float32:
                buffer.data = buffer.data.to(torch.float32)
        elif buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype=dtype)

    return model


def get_autocast_context(device: torch.device, dtype: torch.dtype):
    """Get autocast context manager for mixed precision.
    
    Args:
        device: torch.device for autocast
        dtype: Target dtype for autocast
        
    Returns:
        torch.autocast context manager
    """
    mm = _get_model_management()
    device_type = mm.get_autocast_device(device)
    
    # For CPU, only BF16 is supported for autocast
    if device_type == "cpu" and dtype != torch.bfloat16:
        # Return a no-op context manager
        class NoOpContext:
            def __enter__(self):
                return None
            def __exit__(self, *args):
                pass
        return NoOpContext()
    
    return torch.autocast(device_type=device_type, dtype=dtype)


def validate_dtype_for_device(
    dtype_option: str,
    device_type: str
) -> Tuple[bool, Optional[str]]:
    """Validate if a dtype is appropriate for a device type.
    
    Args:
        dtype_option: Dtype option string
        device_type: Device type string (e.g., "cuda", "cpu")
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    from .device_utils import DEVICE_CPU, is_gpu_device
    
    # FP32 is always valid
    if dtype_option == DTYPE_FP32:
        return True, None
    
    # BF16/FP16 on CPU requires special handling
    if device_type == DEVICE_CPU:
        if dtype_option == DTYPE_BF16:
            # BF16 on CPU is supported in PyTorch 2.0+
            return True, None
        elif dtype_option == DTYPE_FP16:
            return False, "FP16 is not recommended for CPU. Use BF16 or FP32 instead."
    
    # GPU devices generally support all dtypes
    if is_gpu_device(device_type):
        return True, None
    
    return True, None


def get_dtype_memory_multiplier(dtype: torch.dtype) -> float:
    """Get memory multiplier for a dtype compared to FP32.
    
    Args:
        dtype: torch.dtype instance
        
    Returns:
        Memory multiplier (e.g., 0.5 for FP16/BF16, 1.0 for FP32)
    """
    if dtype in (torch.float16, torch.bfloat16):
        return 0.5
    elif dtype == torch.float32:
        return 1.0
    else:
        return 1.0  # Default to no reduction


def estimate_model_memory_mb(
    param_count: int,
    dtype: torch.dtype
) -> float:
    """Estimate model memory usage in MB.
    
    Args:
        param_count: Number of parameters
        dtype: torch.dtype for the model
        
    Returns:
        Estimated memory in MB
    """
    bytes_per_param = 4  # FP32 default
    if dtype in (torch.float16, torch.bfloat16):
        bytes_per_param = 2
    
    return (param_count * bytes_per_param) / (1024 * 1024)
