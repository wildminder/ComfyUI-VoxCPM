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
            config_data = {
                "normalization_available": TEXT_NORMALIZATION_AVAILABLE
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

from .voxcpm_nodes import comfy_entrypoint

WEB_DIRECTORY = "./js"
__all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']
