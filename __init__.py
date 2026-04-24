import os
import sys
import logging
import folder_paths
from .modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS

# Late import for dependency check - avoid breaking import when dependencies are missing
try:
    from .src.voxcpm.utils.text_normalize import TEXT_NORMALIZATION_AVAILABLE
except Exception:
    TEXT_NORMALIZATION_AVAILABLE = False

# Import settings module
from .modules.settings import get_settings

logger = logging.getLogger(__name__)

# Track if config has been sent once
_config_sent_once = False

def send_config_to_client(client_id=None):
    """Send configuration to a specific client or broadcast to all.

    Args:
        client_id: Optional client ID to send to. If None, broadcasts to all.
    """
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
                logger.info("ℹ️ Text normalization packages (inflect, wetext) not found. Normalization will be disabled. Install them using: pip install inflect wetext")
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
        # Wait for server to be ready
        time.sleep(3)
        logger.debug("Sending initial config event...")
        send_config_event()
        logger.debug("Initial config event sent")

    thread = threading.Thread(target=send_after_delay, daemon=True)
    thread.start()

# Schedule config sender using threading
try:
    _schedule_config_send()
except Exception as e:
    logger.debug(f"Failed to schedule config sender: {e}")

# Configure logger
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

# ============================================================================
# API Routes for Custom Model Path
# ============================================================================

def scan_custom_model_path(path: str) -> list:
    """Scan a custom path for valid VoxCPM models.

    Args:
        path: Directory path to scan for models

    Returns:
        List of model names found in the path
    """
    models = []

    if not os.path.isdir(path):
        logger.warning(f"Custom model path does not exist: {path}")
        return models

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # Check for valid model files
            config_exists = os.path.exists(os.path.join(item_path, "config.json"))
            weights_exist = (
                os.path.exists(os.path.join(item_path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(item_path, "model.safetensors"))
            )

            if config_exists and weights_exist:
                models.append(item)
                logger.debug(f"Found model: {item} at {item_path}")

    return models

# Register API routes
try:
    from server import PromptServer
    from aiohttp import web
    import json

    @PromptServer.instance.routes.post("/voxcpm/models")
    async def voxcpm_models_handler(request):
        """Handle POST request to scan custom model path.

        Expects JSON body: {"path": "/path/to/models"}
        Returns: {"models": ["model1", "model2", ...]}
        """
        try:
            data = await request.json()
            path = data.get("path", "")

            if not path:
                return web.json_response(
                    {"error": "Path is required"},
                    status=400
                )

            # Security: validate path exists and is accessible
            if not os.path.isabs(path):
                return web.json_response(
                    {"error": "Path must be absolute"},
                    status=400
                )

            models = scan_custom_model_path(path)

            return web.json_response({
                "path": path,
                "models": models,
                "count": len(models)
            })

        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"Error handling model scan request: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    @PromptServer.instance.routes.get("/voxcpm/model_info")
    async def voxcpm_model_info_handler(request):
        """Handle GET request for rich model metadata.

        Returns model information including download status, architecture,
        sample rate, and size for all known VoxCPM models.

        Returns:
            JSON response: {"models": [{"name", "type", "architecture",
                           "sample_rate", "size_gb", "is_downloaded", ...}, ...]}
        """
        try:
            models_list = []

            for model_name, model_data in AVAILABLE_VOXCPM_MODELS.items():
                model_type = model_data.get("type", "local")
                architecture = model_data.get("architecture", "unknown")
                sample_rate = model_data.get("sample_rate", 0)
                size_gb = model_data.get("size_gb", 0)
                repo_id = model_data.get("repo_id", "")

                # Determine if model is downloaded by checking for config.json
                is_downloaded = False
                if model_type == "official":
                    # Official models live under models/tts/VoxCPM/<model_name>/
                    for tts_folder in folder_paths.get_folder_paths("tts"):
                        model_dir = os.path.join(tts_folder, VOXCPM_SUBDIR_NAME, model_name)
                        if os.path.isdir(model_dir):
                            config_path = os.path.join(model_dir, "config.json")
                            if os.path.exists(config_path):
                                is_downloaded = True
                                break
                elif model_type == "local":
                    # Local models: check the stored path
                    model_path = model_data.get("path", "")
                    if model_path and os.path.isdir(model_path):
                        config_path = os.path.join(model_path, "config.json")
                        if os.path.exists(config_path):
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
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    logger.debug("VoxCPM API routes registered")

except Exception as e:
    logger.warning(f"Failed to register API routes: {e}")

from .voxcpm_nodes import comfy_entrypoint

WEB_DIRECTORY = "./js"
__all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']
