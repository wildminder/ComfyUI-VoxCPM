"""VoxCPM API route handlers for ComfyUI server.

All HTTP endpoints for the VoxCPM custom node are registered here.
This module is imported by __init__.py which calls register_routes()
to attach the handlers to the ComfyUI server.

Endpoints:
- POST /voxcpm/models          — Scan a directory for VoxCPM models
- GET  /voxcpm/model_info       — Get info for all known models
- POST /voxcpm/cancel_download  — Cancel an active model download
- GET  /voxcpm/tts_search_paths — List registered tts paths with models
- GET  /voxcpm/validate_directory — Validate a directory for model registration
- POST /voxcpm/register_model_path — Register a new tts search path
- GET  /voxcpm/download_status  — Get status of active downloads
- GET  /voxcpm/heavy_extension/{filename} — Serve lazy-loaded JS
"""

import os
import json
import logging

from aiohttp import web
from .model_info import (
    AVAILABLE_VOXCPM_MODELS,
    MODEL_CONFIGS,
    read_model_config,
    is_voxcpm_model_dir,
    scan_voxcpm_models,
    scan_custom_model_path,
)

logger = logging.getLogger(__name__)

VOXCPM_SUBDIR_NAME = "VoxCPM"


def register_routes(routes, folder_paths_module, current_dir: str):
    """Register all VoxCPM API routes with the ComfyUI server.

    Args:
        routes: PromptServer.instance.routes — the aiohttp route decorator target
        folder_paths_module: The folder_paths module for path resolution
        current_dir: The package root directory (for locating js_lazy/)
    """

    @routes.post("/voxcpm/models")
    async def voxcpm_models_handler(request):
        """Scan a directory for VoxCPM models.
    
        Checks for models in:
        1. Direct subdirectories of the given path
        2. A VoxCPM/ subdirectory within the given path (convention: tts/VoxCPM/)
        3. The path itself (if it's a direct model folder)
        """
        try:
            data = await request.json()
            path = data.get("path", "")
    
            if not path:
                return web.json_response({"error": "Path is required"}, status=400)
            if not os.path.isabs(path):
                return web.json_response({"error": "Path must be absolute"}, status=400)
    
            # Check if the path itself is a direct model folder
            is_direct_model = is_voxcpm_model_dir(path)
            direct_model_info = None
            if is_direct_model:
                config = read_model_config(path) or {}
                direct_model_info = {
                    "name": os.path.basename(path),
                    "architecture": config.get("architecture", "unknown"),
                    "path": path,
                }
    
            # Scan for models: check VoxCPM/ subdir first (convention), then direct children
            models = []
            if not is_direct_model and os.path.isdir(path):
                voxcpm_subdir = os.path.join(path, VOXCPM_SUBDIR_NAME)
                scan_dir = voxcpm_subdir if os.path.isdir(voxcpm_subdir) else path
                models = scan_voxcpm_models(scan_dir)
    
            return web.json_response({
                "path": path,
                "models": models,
                "count": len(models),
                "is_direct_model": is_direct_model,
                "direct_model_info": direct_model_info,
            })
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error handling model scan request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @routes.get("/voxcpm/model_info")
    async def voxcpm_model_info_handler(request):
        """Get info for all known VoxCPM models (official + local)."""
        try:
            models_list = []
            for model_name, model_data in AVAILABLE_VOXCPM_MODELS.items():
                model_type = model_data.get("type", "local")
                architecture = model_data.get("architecture", "unknown")
                sample_rate = model_data.get("sample_rate", 0)
                size_gb = model_data.get("size_gb", 0)
                repo_id = model_data.get("repo_id", "")

                is_downloaded = False
                if model_type == "official":
                    for tts_folder in folder_paths_module.get_folder_paths("tts"):
                        model_dir = os.path.join(tts_folder, VOXCPM_SUBDIR_NAME, model_name)
                        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
                            is_downloaded = True
                            break
                elif model_type == "local":
                    model_path = model_data.get("path", "")
                    if model_path and os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                        is_downloaded = True

                info = {
                    "name": model_name,
                    "type": model_type,
                    "architecture": architecture,
                    "sample_rate": sample_rate,
                    "size_gb": size_gb,
                    "is_downloaded": is_downloaded,
                }
                if repo_id:
                    info["repo_id"] = repo_id
                if model_type == "local" and "path" in model_data:
                    info["path"] = model_data["path"]
                models_list.append(info)

            # Include settings in the response so the frontend can
            # restore custom model path configuration even if the
            # WebSocket config event was missed (e.g., heavy module
            # loaded after the initial voxcpm.config broadcast).
            # Reload settings from comfy.settings.json to ensure fresh
            # data (the frontend may have written new settings via
            # storeSetting since the backend last read the file).
            try:
                from .settings import get_settings
                from .model_info import TEXT_NORMALIZATION_AVAILABLE
                settings = get_settings()
                settings.reload()
                return web.json_response({
                    "models": models_list,
                    "normalization_available": TEXT_NORMALIZATION_AVAILABLE,
                    "settings": settings.to_dict(),
                })
            except Exception:
                # Settings module not available — return models only
                return web.json_response({"models": models_list})

        except Exception as e:
            logger.error(f"Error handling model info request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/voxcpm/cancel_download")
    async def voxcpm_cancel_download_handler(request):
        """Cancel an active model download.

        Expects JSON body: {"model_name": "VoxCPM2"}
        Returns: {"success": true/false, "message": "..."}
        """
        try:
            from .downloader import get_download_manager
            data = await request.json()
            model_name = data.get("model_name", "")
            if not model_name:
                return web.json_response({"error": "model_name is required"}, status=400)

            download_manager = get_download_manager()
            cancelled = download_manager.cancel_download(model_name)
            if cancelled:
                return web.json_response({"success": True, "message": f"Download cancelled for '{model_name}'"})
            else:
                return web.json_response({"success": False, "message": f"No active download for '{model_name}'"})

        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error handling cancel download request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @routes.get("/voxcpm/tts_search_paths")
    async def voxcpm_tts_search_paths_handler(request):
        """Return all registered tts model search paths with model listings.

        Returns:
        {
            "paths": [
                {
                    "path": "/path/to/tts",
                    "is_default": true,
                    "voxcpm_subdir": "/path/to/tts/VoxCPM",
                    "models": [
                        {"name": "VoxCPM2", "architecture": "voxcpm2", "path": "..."},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        try:
            tts_paths = folder_paths_module.get_folder_paths("tts")
            default_tts = os.path.join(folder_paths_module.models_dir, "tts")
            paths_info = []

            for tts_path in tts_paths:
                voxcpm_subdir = os.path.join(tts_path, VOXCPM_SUBDIR_NAME)
                models = []
                is_direct_model = False
                direct_model_info = None
    
                if os.path.isdir(voxcpm_subdir):
                    # Convention: tts/VoxCPM/<model_name>/
                    models = scan_voxcpm_models(voxcpm_subdir)
                elif is_voxcpm_model_dir(tts_path):
                    # The tts path itself is a direct model folder
                    is_direct_model = True
                    config = read_model_config(tts_path) or {}
                    direct_model_info = {
                        "name": os.path.basename(tts_path),
                        "architecture": config.get("architecture", "unknown"),
                        "path": tts_path,
                    }
                    models = [direct_model_info]
                else:
                    # Scan direct children for model folders
                    models = scan_voxcpm_models(tts_path)
    
                paths_info.append({
                    "path": tts_path,
                    "is_default": (tts_path == default_tts),
                    "voxcpm_subdir": voxcpm_subdir if os.path.isdir(voxcpm_subdir) else None,
                    "is_direct_model": is_direct_model,
                    "direct_model_info": direct_model_info,
                    "models": models,
                })

            return web.json_response({"paths": paths_info})

        except Exception as e:
            logger.error(f"Error handling tts_search_paths request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @routes.get("/voxcpm/validate_directory")
    async def voxcpm_validate_directory_handler(request):
        """Validate whether a directory is within a registered tts search path.

        Also detects if the path itself is a direct VoxCPM model folder
        (contains config.json + weights), in which case it returns the
        model's architecture from config.json.

        Query params:
            path: Directory path to validate

        Returns:
        {
            "valid": bool,
            "is_registered": bool,
            "registered_parent": str | null,
            "exists": bool,
            "is_direct_model": bool,
            "direct_model_info": {...} | null,
            "models": [...],
            "error": str | null
        }
        """
        dir_path = request.query.get("path", "")
        if not dir_path:
            return web.json_response({"error": "path is required"}, status=400)

        try:
            # Normalize the path for comparison
            dir_path = os.path.normpath(dir_path)
            tts_paths = folder_paths_module.get_folder_paths("tts")

            # Check if the path is within any registered tts directory
            is_registered = False
            registered_parent = None
            for tts_path in tts_paths:
                norm_tts = os.path.normpath(tts_path)
                if dir_path == norm_tts or dir_path.startswith(norm_tts + os.sep):
                    is_registered = True
                    registered_parent = tts_path
                    break

            # Check if directory exists
            exists = os.path.isdir(dir_path)

            # Check if this path is itself a direct VoxCPM model folder
            is_direct_model = exists and is_voxcpm_model_dir(dir_path)
            direct_model_info = None
            if is_direct_model:
                config = read_model_config(dir_path) or {}
                direct_model_info = {
                    "name": os.path.basename(dir_path),
                    "architecture": config.get("architecture", "unknown"),
                    "path": dir_path,
                }

            # Scan for VoxCPM models in subdirectories
            models = []
            if exists and not is_direct_model:
                # Check if this is a VoxCPM subdir itself
                voxcpm_subdir = os.path.join(dir_path, VOXCPM_SUBDIR_NAME)
                scan_dir = voxcpm_subdir if os.path.isdir(voxcpm_subdir) else dir_path
                models = scan_voxcpm_models(scan_dir)

            # A direct model folder is valid even if not within a registered tts
            # directory — it IS the model, not a parent directory containing models.
            is_valid = (is_registered and exists) or is_direct_model
            return web.json_response({
                "valid": is_valid,
                "is_registered": is_registered,
                "registered_parent": registered_parent,
                "exists": exists,
                "is_direct_model": is_direct_model,
                "direct_model_info": direct_model_info,
                "models": models,
                "error": None if is_valid else (
                    "Not within a registered tts model directory" if not is_registered
                    else "Directory does not exist" if not exists
                    else None
                ),
            })

        except Exception as e:
            logger.error(f"Error handling validate_directory request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/voxcpm/register_model_path")
    async def voxcpm_register_model_path_handler(request):
        """Register a new directory as a tts model search path.

        This calls folder_paths.add_model_folder_path("tts", path) so
        ComfyUI's model discovery will find models in the new directory.

        After registering, scans the path for VoxCPM models and returns
        the count in the response. If no models are found, a warning is
        included (but the path is still registered — the user may add
        models later).

        Expects JSON body: {"path": "/absolute/path/to/directory"}
        Returns: {"success": bool, "path": str, "error": str | null,
                  "already_registered": bool, "models": [...], "model_count": int,
                  "warning": str | null}
        """
        try:
            data = await request.json()
            path = data.get("path", "")

            if not path:
                return web.json_response({"error": "path is required"}, status=400)

            if not os.path.isabs(path):
                return web.json_response({"error": "Path must be absolute"}, status=400)

            if not os.path.isdir(path):
                return web.json_response({"error": f"Directory does not exist: {path}"}, status=400)

            # Check if already registered
            tts_paths = folder_paths_module.get_folder_paths("tts")
            norm_path = os.path.normpath(path)
            for existing in tts_paths:
                if os.path.normpath(existing) == norm_path:
                    # Already registered — scan for models to return in response
                    models = []
                    is_direct_model = is_voxcpm_model_dir(path)
                    if is_direct_model:
                        config = read_model_config(path) or {}
                        models = [{
                            "name": os.path.basename(path),
                            "architecture": config.get("architecture", "unknown"),
                            "path": path,
                        }]
                    else:
                        voxcpm_subdir = os.path.join(path, VOXCPM_SUBDIR_NAME)
                        scan_dir = voxcpm_subdir if os.path.isdir(voxcpm_subdir) else path
                        models = scan_voxcpm_models(scan_dir)

                    # Add discovered models to AVAILABLE_VOXCPM_MODELS (same
                    # as the new-registration branch below — needed when the
                    # path was registered at runtime but models weren't added
                    # to the registry yet).
                    for model in models:
                        if model["name"] not in AVAILABLE_VOXCPM_MODELS:
                            AVAILABLE_VOXCPM_MODELS[model["name"]] = {
                                "type": "local",
                                "path": model["path"],
                                "architecture": model.get("architecture", "unknown"),
                            }
                            logger.info(f"Added model to registry: {model['name']}")

                    return web.json_response({
                        "success": True,
                        "path": path,
                        "error": None,
                        "already_registered": True,
                        "models": models,
                        "model_count": len(models),
                        "warning": "No VoxCPM models found in this path" if not models else None,
                    })

            # Register the new path
            folder_paths_module.add_model_folder_path("tts", path)
            logger.info(f"Registered new tts model path: {path}")

            # NOTE: Settings persistence is handled by the frontend via
            # app.api.storeSetting("voxcpm.custom_model_path", path) which
            # writes to comfy.settings.json. The backend reads these settings
            # on startup (see __init__.py) and via get_settings().reload().
            # No backend write needed here.

            # Scan for VoxCPM models in the registered path
            models = []
            is_direct_model = is_voxcpm_model_dir(path)
            if is_direct_model:
                config = read_model_config(path) or {}
                models = [{
                    "name": os.path.basename(path),
                    "architecture": config.get("architecture", "unknown"),
                    "path": path,
                }]
            else:
                # Check for VoxCPM/ subdir convention
                voxcpm_subdir = os.path.join(path, VOXCPM_SUBDIR_NAME)
                scan_dir = voxcpm_subdir if os.path.isdir(voxcpm_subdir) else path
                models = scan_voxcpm_models(scan_dir)

            # Add discovered models to AVAILABLE_VOXCPM_MODELS so they're
            # available for validate_inputs and model loading at runtime.
            # Without this, models discovered after startup (via the model
            # directory dialog) won't pass prompt validation.
            for model in models:
                if model["name"] not in AVAILABLE_VOXCPM_MODELS:
                    AVAILABLE_VOXCPM_MODELS[model["name"]] = {
                        "type": "local",
                        "path": model["path"],
                        "architecture": model.get("architecture", "unknown"),
                    }
                    logger.info(f"Added model to registry: {model['name']}")

            return web.json_response({
                "success": True,
                "path": path,
                "error": None,
                "already_registered": False,
                "models": models,
                "model_count": len(models),
                "warning": "No VoxCPM models found in this path" if not models else None,
            })

        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error handling register_model_path request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @routes.delete("/voxcpm/settings")
    async def voxcpm_clear_settings_handler(request):
        """Clear all VoxCPM settings from comfy.settings.json.

        Removes all keys starting with "voxcpm." from the shared
        comfy.settings.json file. This is used for cleanup when the
        user wants to reset to defaults or when uninstalling.

        Returns: {"success": bool, "removed": [list of removed keys]}
        """
        try:
            import folder_paths

            user_dir = folder_paths.get_user_directory()
            if not user_dir:
                return web.json_response({"error": "User directory not found"}, status=500)

            settings_file = os.path.join(user_dir, "default", "comfy.settings.json")
            if not os.path.exists(settings_file):
                return web.json_response({"success": True, "removed": []})

            with open(settings_file, "r", encoding="utf-8") as f:
                all_settings = json.load(f)

            # Remove all voxcpm.* keys
            removed = [k for k in list(all_settings.keys()) if k.startswith("voxcpm.")]
            for key in removed:
                del all_settings[key]

            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(all_settings, f, indent=4)

            # Reload settings in the global instance
            from .settings import get_settings
            get_settings().reload()

            logger.info(f"Cleared VoxCPM settings: {removed}")
            return web.json_response({"success": True, "removed": removed})

        except Exception as e:
            logger.error(f"Error clearing VoxCPM settings: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @routes.get("/voxcpm/download_status")
    async def voxcpm_download_status_handler(request):
        """Get status of all active downloads.

        Returns: {"downloads": {"VoxCPM2": {...state...}}, ...}
        """
        try:
            from .downloader import get_download_manager
            download_manager = get_download_manager()
            statuses = download_manager.get_all_statuses()
            return web.json_response({"downloads": statuses})

        except Exception as e:
            logger.error(f"Error handling download status request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    # ── Lazy-Load Heavy Extension Endpoint ────────────────────────────
    # The heavy JS (voxcpmHeavy.js) is stored OUTSIDE the WEB_DIRECTORY
    # (js/) so ComfyUI's GET /extensions endpoint doesn't discover it.
    # The lightweight stub (js/extension.js) dynamically imports the
    # heavy code via this endpoint only when a VoxCPM node is actually
    # used in the workflow.
    _heavy_js_dir = os.path.join(current_dir, "js_lazy")

    @routes.get("/voxcpm/heavy_extension/{filename}")
    async def voxcpm_heavy_extension_handler(request):
        """Serve lazy-loaded JS files from outside WEB_DIRECTORY.

        Only voxcpmHeavy.js is allowed — prevents arbitrary file reads.
        Returns application/javascript with correct charset.
        """
        filename = request.match_info.get("filename", "")
        if filename not in ("voxcpmHeavy.js",):
            return web.json_response(
                {"error": f"File not allowed: {filename}"}, status=403
            )

        file_path = os.path.join(_heavy_js_dir, filename)
        if not os.path.isfile(file_path):
            return web.json_response(
                {"error": f"File not found: {filename}"}, status=404
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return web.Response(
                text=content,
                content_type="application/javascript",
                charset="utf-8",
            )
        except Exception as e:
            logger.error(f"Error serving heavy extension file {filename}: {e}")
            return web.json_response({"error": str(e)}, status=500)

    logger.debug("VoxCPM API routes registered")
