import os
import sys
import logging

# Pytest forcefully imports __init__  out-of-context during test collection.
# causes relative imports to crash. If we detect pytest, we safely exit. Just quick fix.
if "pytest" in sys.modules:
    WEB_DIRECTORY = "./js"
    __all__ = ['WEB_DIRECTORY']

else:
    logger = logging.getLogger(__name__)

    # Detect if running inside ComfyUI environment
    try:
        import folder_paths
        _COMFYUI_AVAILABLE = True
    except ImportError:
        folder_paths = None
        _COMFYUI_AVAILABLE = False

    # These imports are safe even without ComfyUI (no folder_paths dependency)
    from .modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS

    # Late import for dependency check - avoid breaking import when dependencies are missing
    try:
        from .src.voxcpm.utils.text_normalize import TEXT_NORMALIZATION_AVAILABLE
    except Exception:
        TEXT_NORMALIZATION_AVAILABLE = False

    # Import settings module
    from .modules.settings import get_settings

    # Track if config has been sent once
    _config_sent_once = False

    def send_config_to_client(client_id=None):
        """Send configuration to a specific client or broadcast to all."""
        global _config_sent_once
        try:
            from server import PromptServer
            if PromptServer.instance is not None:
                # Get user settings
                settings = get_settings()
                config_data = {
                    "normalization_available": TEXT_NORMALIZATION_AVAILABLE,
                    "settings": settings.to_dict(),
                }
                logger.debug(f"Sending config: {config_data} to client: {client_id or 'all'}")
                if client_id:
                    PromptServer.instance.send_sync("voxcpm.config", config_data, client_id)
                else:
                    PromptServer.instance.send_sync("voxcpm.config", config_data)
                logger.debug("Config sent successfully")

            if not _config_sent_once and not TEXT_NORMALIZATION_AVAILABLE:
                _config_sent_once = True
                logger.info("Text normalization packages (inflect, wetext) not found. Normalization will be disabled. Install them using: pip install inflect wetext")
        except Exception as e:
            logger.warning(f"Failed to send config: {e}")

    def send_config_event():
        """Send configuration to frontend (legacy function for compatibility)."""
        send_config_to_client()

    def _schedule_config_send():
        """Schedule config to be sent once after server starts using threading."""
        import threading
        import time

        def send_after_delay():
            time.sleep(3)
            logger.debug("Sending initial config event...")
            send_config_event()
            logger.debug("Initial config event sent")

        thread = threading.Thread(target=send_after_delay, daemon=True)
        thread.start()

    try:
        _schedule_config_send()
    except Exception as e:
        logger.debug(f"Failed to schedule config sender: {e}")

    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f"[ComfyUI-VoxCPM] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    if _COMFYUI_AVAILABLE:
        VOXCPM_SUBDIR_NAME = "VoxCPM"

        tts_path = os.path.join(folder_paths.models_dir, "tts")
        os.makedirs(tts_path, exist_ok=True)

        if "tts" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["tts"] = ([tts_path], folder_paths.supported_pt_extensions)
        else:
            if tts_path not in folder_paths.folder_names_and_paths["tts"][0]:
                folder_paths.folder_names_and_paths["tts"][0].append(tts_path)

        for model_name, config in MODEL_CONFIGS.items():
            AVAILABLE_VOXCPM_MODELS[model_name] = {
                "type": "official",
                **config
            }

        voxcpm_search_paths = []
        for tts_folder in folder_paths.get_folder_paths("tts"):
            potential_path = os.path.join(tts_folder, VOXCPM_SUBDIR_NAME)
            if os.path.isdir(potential_path) and potential_path not in voxcpm_search_paths:
                voxcpm_search_paths.append(potential_path)

        for search_path in voxcpm_search_paths:
            if not os.path.isdir(search_path):
                continue
            for item in os.listdir(search_path):
                item_path = os.path.join(search_path, item)
                if os.path.isdir(item_path) and item not in AVAILABLE_VOXCPM_MODELS:
                    config_exists = os.path.exists(os.path.join(item_path, "config.json"))
                    weights_exist = os.path.exists(os.path.join(item_path, "pytorch_model.bin")) or os.path.exists(os.path.join(item_path, "model.safetensors"))

                    if config_exists and weights_exist:
                        AVAILABLE_VOXCPM_MODELS[item] = {
                            "type": "local",
                            "path": item_path
                        }

        def scan_custom_model_path(path: str) -> list:
            models = []
            if not os.path.isdir(path):
                logger.warning(f"Custom model path does not exist: {path}")
                return models

            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    config_exists = os.path.exists(os.path.join(item_path, "config.json"))
                    weights_exist = (
                        os.path.exists(os.path.join(item_path, "pytorch_model.bin")) or
                        os.path.exists(os.path.join(item_path, "model.safetensors"))
                    )
                    if config_exists and weights_exist:
                        models.append(item)
                        logger.debug(f"Found model: {item} at {item_path}")

            return models

        try:
            from server import PromptServer
            from aiohttp import web
            import json

            @PromptServer.instance.routes.post("/voxcpm/models")
            async def voxcpm_models_handler(request):
                try:
                    data = await request.json()
                    path = data.get("path", "")

                    if not path:
                        return web.json_response({"error": "Path is required"}, status=400)
                    if not os.path.isabs(path):
                        return web.json_response({"error": "Path must be absolute"}, status=400)

                    models = scan_custom_model_path(path)
                    return web.json_response({
                        "path": path,
                        "models": models,
                        "count": len(models)
                    })
                except json.JSONDecodeError:
                    return web.json_response({"error": "Invalid JSON"}, status=400)
                except Exception as e:
                    logger.error(f"Error handling model scan request: {e}")
                    return web.json_response({"error": str(e)}, status=500)

            @PromptServer.instance.routes.get("/voxcpm/model_info")
            async def voxcpm_model_info_handler(request):
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
                            for tts_folder in folder_paths.get_folder_paths("tts"):
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

                    return web.json_response({"models": models_list})

                except Exception as e:
                    logger.error(f"Error handling model info request: {e}")
                    return web.json_response({"error": str(e)}, status=500)

            @PromptServer.instance.routes.post("/voxcpm/cancel_download")
            async def voxcpm_cancel_download_handler(request):
                """Cancel an active model download.
        
                Expects JSON body: {"model_name": "VoxCPM2"}
                Returns: {"success": true/false, "message": "..."}
                """
                try:
                    from .modules.downloader import get_download_manager
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
        
            @PromptServer.instance.routes.get("/voxcpm/tts_search_paths")
            async def voxcpm_tts_search_paths_handler(request):
                """Return all registered tts model search paths with model listings.

                Returns:
                {
                    "paths": [
                        {
                            "path": "/path/to/tts",
                            "is_default": true,
                            "voxcpm_subdir": "/path/to/tts/VoxCPM",
                            "models": ["VoxCPM2", "VoxCPM1.5"]
                        },
                        ...
                    ]
                }
                """
                try:
                    tts_paths = folder_paths.get_folder_paths("tts")
                    default_tts = os.path.join(folder_paths.models_dir, "tts")
                    paths_info = []

                    for tts_path in tts_paths:
                        voxcpm_subdir = os.path.join(tts_path, VOXCPM_SUBDIR_NAME)
                        models = []
                        if os.path.isdir(voxcpm_subdir):
                            for item in os.listdir(voxcpm_subdir):
                                item_path = os.path.join(voxcpm_subdir, item)
                                if os.path.isdir(item_path):
                                    config_exists = os.path.exists(os.path.join(item_path, "config.json"))
                                    weights_exist = (
                                        os.path.exists(os.path.join(item_path, "model.safetensors")) or
                                        os.path.exists(os.path.join(item_path, "pytorch_model.bin"))
                                    )
                                    if config_exists and weights_exist:
                                        models.append(item)

                        paths_info.append({
                            "path": tts_path,
                            "is_default": (tts_path == default_tts),
                            "voxcpm_subdir": voxcpm_subdir if os.path.isdir(voxcpm_subdir) else None,
                            "models": models,
                        })

                    return web.json_response({"paths": paths_info})

                except Exception as e:
                    logger.error(f"Error handling tts_search_paths request: {e}")
                    return web.json_response({"error": str(e)}, status=500)

            @PromptServer.instance.routes.get("/voxcpm/validate_directory")
            async def voxcpm_validate_directory_handler(request):
                """Validate whether a directory is within a registered tts search path.

                Query params:
                    path: Directory path to validate

                Returns:
                {
                    "valid": bool,
                    "is_registered": bool,  # True if path is within a registered tts directory
                    "registered_parent": str | null,  # The registered tts path that contains this path
                    "exists": bool,  # True if the directory exists on disk
                    "models": [],  # VoxCPM models found in this directory
                    "error": str | null
                }
                """
                dir_path = request.query.get("path", "")
                if not dir_path:
                    return web.json_response({"error": "path is required"}, status=400)

                try:
                    # Normalize the path for comparison
                    dir_path = os.path.normpath(dir_path)
                    tts_paths = folder_paths.get_folder_paths("tts")

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

                    # Scan for VoxCPM models
                    models = []
                    if exists:
                        # Check if this is a VoxCPM subdir itself
                        voxcpm_subdir = os.path.join(dir_path, VOXCPM_SUBDIR_NAME)
                        scan_dir = voxcpm_subdir if os.path.isdir(voxcpm_subdir) else dir_path

                        for item in os.listdir(scan_dir):
                            item_path = os.path.join(scan_dir, item)
                            if os.path.isdir(item_path):
                                config_exists = os.path.exists(os.path.join(item_path, "config.json"))
                                weights_exist = (
                                    os.path.exists(os.path.join(item_path, "model.safetensors")) or
                                    os.path.exists(os.path.join(item_path, "pytorch_model.bin"))
                                )
                                if config_exists and weights_exist:
                                    models.append(item)

                    is_valid = is_registered and exists
                    return web.json_response({
                        "valid": is_valid,
                        "is_registered": is_registered,
                        "registered_parent": registered_parent,
                        "exists": exists,
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

            @PromptServer.instance.routes.post("/voxcpm/register_model_path")
            async def voxcpm_register_model_path_handler(request):
                """Register a new directory as a tts model search path.

                This calls folder_paths.add_model_folder_path("tts", path) so
                ComfyUI's model discovery will find models in the new directory.

                Expects JSON body: {"path": "/absolute/path/to/directory"}
                Returns: {"success": bool, "path": str, "error": str | null}
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
                    tts_paths = folder_paths.get_folder_paths("tts")
                    norm_path = os.path.normpath(path)
                    for existing in tts_paths:
                        if os.path.normpath(existing) == norm_path:
                            return web.json_response({
                                "success": True,
                                "path": path,
                                "error": None,
                                "already_registered": True,
                            })

                    # Register the new path
                    folder_paths.add_model_folder_path("tts", path)
                    logger.info(f"Registered new tts model path: {path}")

                    return web.json_response({
                        "success": True,
                        "path": path,
                        "error": None,
                        "already_registered": False,
                    })

                except json.JSONDecodeError:
                    return web.json_response({"error": "Invalid JSON"}, status=400)
                except Exception as e:
                    logger.error(f"Error handling register_model_path request: {e}")
                    return web.json_response({"error": str(e)}, status=500)

            @PromptServer.instance.routes.get("/voxcpm/download_status")
            async def voxcpm_download_status_handler(request):
                """Get status of all active downloads.
        
                Returns: {"downloads": {"VoxCPM2": {...state...}}, ...}
                """
                try:
                    from .modules.downloader import get_download_manager
                    download_manager = get_download_manager()
                    statuses = download_manager.get_all_statuses()
                    return web.json_response({"downloads": statuses})
        
                except Exception as e:
                    logger.error(f"Error handling download status request: {e}")
                    return web.json_response({"error": str(e)}, status=500)
        
            # === Lazy-Load Heavy Extension Endpoint ===
            # The heavy JS (voxcpmHeavy.js + shared.js) is stored OUTSIDE the
            # WEB_DIRECTORY (js/) so ComfyUI's GET /extensions endpoint doesn't
            # discover it.  The lightweight stub (js/extension.js) dynamically
            # imports the heavy code via this endpoint only when a VoxCPM node
            # is actually used in the workflow.
            _HEAVY_JS_DIR = os.path.join(current_dir, "js_lazy")

            @PromptServer.instance.routes.get("/voxcpm/heavy_extension/{filename}")
            async def voxcpm_heavy_extension_handler(request):
                """Serve lazy-loaded JS files from outside WEB_DIRECTORY.

                Only voxcpmHeavy.js is allowed — prevents arbitrary file reads.
                Returns application/javascript with correct charset.

                NOTE: shared.js was removed in a previous architecture. The stub
                (extension.js) now inlines VOXCPM_NODE_TYPES to avoid a cross-directory
                chunk dependency that caused 404 errors. All shared constants are
                inlined into voxcpmHeavy.js at build time.
                """
                filename = request.match_info.get("filename", "")
                if filename not in ("voxcpmHeavy.js",):
                    return web.json_response(
                        {"error": f"File not allowed: {filename}"}, status=403
                    )

                file_path = os.path.join(_HEAVY_JS_DIR, filename)
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

        except Exception as e:
            logger.warning(f"Failed to register API routes: {e}")

        from .voxcpm_nodes import comfy_entrypoint

        WEB_DIRECTORY = "./js"
        __all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']

    else:
        # do we need this stuff here?
        WEB_DIRECTORY = "./js"
        __all__ = ['WEB_DIRECTORY', '_COMFYUI_AVAILABLE']