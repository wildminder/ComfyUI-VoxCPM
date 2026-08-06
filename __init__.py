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
    
            # Migration: remove old voxcpm_settings.json (replaced by
            # comfy.settings.json via ComfyUI's built-in settings API).
            try:
                from .modules.settings import SETTINGS_FILE
                old_settings_path = os.path.join(
                    folder_paths.get_user_directory(), "default", "VoxCPM", SETTINGS_FILE
                )
                if os.path.exists(old_settings_path):
                    os.remove(old_settings_path)
                    logger.info("Removed old voxcpm_settings.json (migrated to comfy.settings.json)")
            except Exception:
                pass  # Non-critical migration

            # Re-register custom model paths from persisted settings.
            # When a user selects a custom model directory via the frontend
            # dialog, the path is saved to comfy.settings.json via
            # app.api.storeSetting("voxcpm.custom_model_path", path).
            # On server restart, we re-register it with folder_paths so
            # models in the custom directory are discoverable again.
            try:
                from .modules.settings import get_settings
                settings = get_settings()
                if settings.use_custom_path and settings.custom_model_path:
                    custom_path = settings.custom_model_path
                    if os.path.isdir(custom_path):
                        existing_paths = folder_paths.get_folder_paths("tts")
                        if custom_path not in existing_paths:
                            folder_paths.add_model_folder_path("tts", custom_path)
                            logger.info(f"Re-registered custom model path from settings: {custom_path}")
    
                        # Scan for models in the custom path
                        VOXCPM_SUBDIR = "VoxCPM"
                        voxcpm_subdir = os.path.join(custom_path, VOXCPM_SUBDIR)
                        scan_dir = voxcpm_subdir if os.path.isdir(voxcpm_subdir) else custom_path
                        for model_info in scan_voxcpm_models(scan_dir):
                            item = model_info["name"]
                            if item not in AVAILABLE_VOXCPM_MODELS:
                                AVAILABLE_VOXCPM_MODELS[item] = {
                                    "type": "local",
                                    "path": model_info["path"],
                                    "architecture": model_info["architecture"],
                                }
                    else:
                        logger.warning(f"Custom model path from settings no longer exists: {custom_path}")
            except Exception as e:
                logger.warning(f"Failed to re-register custom model paths: {e}")
    
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

        # Register event-driven config broadcast (primary mechanism)
        # This sends config when a client connects, replacing the fragile
        # time.sleep(3) approach in schedule_config_send().
        try:
            from .modules.config_broadcast import register_config_on_connect
            register_config_on_connect()
        except Exception as e:
            logger.debug(f"Failed to register client connect handler: {e}")

    # ── Exports ───────────────────────────────────────────────────────
    from .voxcpm_nodes import comfy_entrypoint

    WEB_DIRECTORY = "./js"
    __all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']
