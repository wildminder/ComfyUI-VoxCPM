"""ComfyUI-VoxCPM: Text-to-speech custom node integrating the VoxCPM model family.

This is the package entrypoint loaded by ComfyUI. It handles:
- Environment detection (ComfyUI available or not)
- ComfyUI folder_paths registration for tts models
- Model discovery at startup (official + local)
- API route registration
- Frontend extension configuration broadcasting

All heavy logic is delegated to focused modules:
- modules/model_info.py: Model configs, registry, and scanning
- modules/api_routes.py: HTTP endpoint handlers
- modules/config_broadcast.py: Frontend config event broadcasting
"""

import os
import sys
import logging

# ── Pytest Guard ──────────────────────────────────────────────────────
# Pytest forcefully imports __init__ out-of-context during test collection.
# This causes relative imports to crash. Detect and exit early.
if "pytest" in sys.modules:
    WEB_DIRECTORY = "./js"
    __all__ = ['WEB_DIRECTORY']

else:
    logger = logging.getLogger(__name__)

    # ── Environment Detection ─────────────────────────────────────────
    try:
        import folder_paths
        _COMFYUI_AVAILABLE = True
    except ImportError:
        folder_paths = None
        _COMFYUI_AVAILABLE = False

    # ── Package Imports ───────────────────────────────────────────────
    from .modules.model_info import (
        AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS, TEXT_NORMALIZATION_AVAILABLE,
        scan_voxcpm_models,
    )
    from .modules.config_broadcast import schedule_config_send

    # ── Logger Setup ──────────────────────────────────────────────────
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[ComfyUI-VoxCPM] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # ── sys.path Registration ─────────────────────────────────────────
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    # ── ComfyUI Integration ───────────────────────────────────────────
    if _COMFYUI_AVAILABLE:
        # Register tts folder path with ComfyUI
        tts_path = os.path.join(folder_paths.models_dir, "tts")
        os.makedirs(tts_path, exist_ok=True)
        if "tts" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["tts"] = (
                [tts_path], folder_paths.supported_pt_extensions
            )
        elif tts_path not in folder_paths.folder_names_and_paths["tts"][0]:
            folder_paths.folder_names_and_paths["tts"][0].append(tts_path)

        # Populate AVAILABLE_VOXCPM_MODELS with official models
        for model_name, config in MODEL_CONFIGS.items():
            AVAILABLE_VOXCPM_MODELS[model_name] = {"type": "official", **config}

        # Discover local models in tts/VoxCPM/ subdirectories
        VOXCPM_SUBDIR_NAME = "VoxCPM"
        voxcpm_search_paths = []
        for tts_folder in folder_paths.get_folder_paths("tts"):
            potential_path = os.path.join(tts_folder, VOXCPM_SUBDIR_NAME)
            if os.path.isdir(potential_path) and potential_path not in voxcpm_search_paths:
                voxcpm_search_paths.append(potential_path)

        for search_path in voxcpm_search_paths:
            if not os.path.isdir(search_path):
                continue
            for model_info in scan_voxcpm_models(search_path):
                item = model_info["name"]
                if item not in AVAILABLE_VOXCPM_MODELS:
                    AVAILABLE_VOXCPM_MODELS[item] = {
                        "type": "local",
                        "path": model_info["path"],
                        "architecture": model_info["architecture"],
                    }

        # Register API routes
        try:
            from server import PromptServer
            from .modules.api_routes import register_routes
            register_routes(PromptServer.instance.routes, folder_paths, current_dir)
        except Exception as e:
            logger.warning(f"Failed to register API routes: {e}")

        # Schedule config broadcast to frontend
        try:
            schedule_config_send()
        except Exception as e:
            logger.debug(f"Failed to schedule config sender: {e}")

    # ── Exports ───────────────────────────────────────────────────────
    from .voxcpm_nodes import comfy_entrypoint

    WEB_DIRECTORY = "./js"
    __all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']
