"""VoxCPM model configurations, registry, and discovery utilities.

This module owns:
- MODEL_CONFIGS: Static metadata for official VoxCPM models
- AVAILABLE_VOXCPM_MODELS: Dynamic registry populated at startup
- TEXT_NORMALIZATION_AVAILABLE: Feature flag for text normalization deps
- Model scanning/detection functions for filesystem discovery
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# ── Official Model Configurations ─────────────────────────────────────

MODEL_CONFIGS = {
    # VoxCPM2 models (48kHz, 30 languages, voice design & cloning)
    "VoxCPM2": {
        "repo_id": "openbmb/VoxCPM2",
        "architecture": "voxcpm2",
        "sample_rate": 48000,
        "size_gb": 8.0,  # ~2B parameters
    },
    # VoxCPM1.5 models (44.1kHz, 2 languages)
    "VoxCPM1.5": {
        "repo_id": "openbmb/VoxCPM1.5",
        "architecture": "voxcpm",
        "sample_rate": 44100,
        "size_gb": 1.6,  # ~0.6B parameters
    },
    "VoxCPM-0.5B": {
        "repo_id": "openbmb/VoxCPM-0.5B",
        "architecture": "voxcpm",
        "sample_rate": 44100,
        "size_gb": 1.0,
    },
}

# Populate AVAILABLE_VOXCPM_MODELS from MODEL_CONFIGS
# This allows dynamic model discovery
AVAILABLE_VOXCPM_MODELS = {name: {"type": "official", **config} for name, config in MODEL_CONFIGS.items()}

# ── Text Normalization Feature Flag ──────────────────────────────────

# Late import for dependency check — avoid breaking import when dependencies are missing
try:
    from ..src.voxcpm.utils.text_normalize import TEXT_NORMALIZATION_AVAILABLE
except Exception:
    TEXT_NORMALIZATION_AVAILABLE = False

# ── Valid VoxCPM Architectures ───────────────────────────────────────

VALID_VOXCPM_ARCHITECTURES = ("voxcpm", "voxcpm2")

# ── Model Scanning Utilities ─────────────────────────────────────────


def read_model_config(model_dir: str) -> dict | None:
    """Read a VoxCPM model's config.json and return parsed dict.

    Returns None if config.json doesn't exist or can't be parsed.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Could not read config.json in {model_dir}: {e}")
        return None


def is_voxcpm_model_dir(path: str) -> bool:
    """Check if a directory is a valid VoxCPM model.

    A valid VoxCPM model directory must contain:
    - config.json with "architecture" field set to "voxcpm" or "voxcpm2"
    - At least one weight file (model.safetensors or pytorch_model.bin)

    This is stricter than just checking for config.json + weights,
    which would match any HuggingFace model. The architecture check
    ensures we only detect VoxCPM models.
    """
    if not os.path.isdir(path):
        return False
    config = read_model_config(path)
    if config is None:
        return False
    architecture = config.get("architecture", "")
    if architecture not in VALID_VOXCPM_ARCHITECTURES:
        return False
    weights_exist = (
        os.path.exists(os.path.join(path, "model.safetensors"))
        or os.path.exists(os.path.join(path, "pytorch_model.bin"))
    )
    return weights_exist


def scan_voxcpm_models(scan_dir: str) -> list[dict]:
    """Scan a directory for VoxCPM model subdirectories.

    Returns a list of dicts:
    [
        {"name": "VoxCPM2", "architecture": "voxcpm2", "path": "/path/to/VoxCPM2"},
        {"name": "my_custom_model", "architecture": "voxcpm", "path": "/path/to/my_custom_model"},
        ...
    ]
    """
    models = []
    if not os.path.isdir(scan_dir):
        return models

    for item in os.listdir(scan_dir):
        item_path = os.path.join(scan_dir, item)
        if is_voxcpm_model_dir(item_path):
            config = read_model_config(item_path) or {}
            architecture = config.get("architecture", "unknown")
            models.append({
                "name": item,
                "architecture": architecture,
                "path": item_path,
            })
            logger.debug(f"Found model: {item} (architecture={architecture}) at {item_path}")

    return models


def scan_custom_model_path(path: str) -> list[dict]:
    """Scan a custom model path for VoxCPM models.

    Returns a list of dicts with name, architecture, and path.
    Logs a warning if the path does not exist.
    """
    if not os.path.isdir(path):
        logger.warning(f"Custom model path does not exist: {path}")
        return []

    return scan_voxcpm_models(path)
