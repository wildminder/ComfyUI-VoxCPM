"""
Shared utilities for VoxCPM nodes.

This module contains utility functions shared between voxcpm_nodes.py and voxcpm2_nodes.py
to avoid circular imports.
"""

import torch

# Global cache for model patchers (shared across all nodes)
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
