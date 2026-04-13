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