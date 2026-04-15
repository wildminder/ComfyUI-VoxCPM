"""Shared utilities for VoxCPM nodes.

This module contains utility functions shared between voxcpm_nodes.py and voxcpm2_nodes.py
to avoid circular imports.
"""

import torch

# Global cache for model patchers (shared across all nodes)
VOXCPM_PATCHER_CACHE = {}

# Import device utilities for backward compatibility
# New code should import directly from modules.device_utils
from .device_utils import (
    get_available_devices,
    get_torch_device,
    get_offload_device,
    should_use_fp16,
    should_use_bf16,
    get_device_display_name,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_MPS,
)


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Seed value. Use -1 for random seed generation.
    """
    if seed < 0:
        # Use Python's random to generate a valid seed within torch's range
        import random
        seed = random.randint(0, 2**31 - 1)  # Use 32-bit range for compatibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_voxcpm2_model(model_name: str, model_configs: dict) -> bool:
    """Check if the model is a VoxCPM2 variant.

    Args:
        model_name: Name of the model to check
        model_configs: Dictionary of model configurations

    Returns:
        True if the model is VoxCPM2, False otherwise
    """
    config = model_configs.get(model_name, {})
    return config.get("architecture", "voxcpm") == "voxcpm2"
