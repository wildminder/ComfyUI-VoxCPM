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

                Only voxcpmHeavy.js and shared.js are allowed — prevents arbitrary
                file reads.  Returns application/javascript with correct charset.
                """
                filename = request.match_info.get("filename", "")
                if filename not in ("voxcpmHeavy.js", "shared.js"):
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