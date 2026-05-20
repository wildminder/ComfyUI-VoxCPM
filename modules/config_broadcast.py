"""VoxCPM configuration broadcasting to frontend clients.

Handles sending configuration data (settings, feature flags) to the
ComfyUI frontend via WebSocket events. This runs after server startup
to ensure the frontend has the latest configuration.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Track if config has been sent once (for first-run notification)
_config_sent_once = False


def send_config_to_client(client_id=None):
    """Send configuration to a specific client or broadcast to all.

    Sends a 'voxcpm.config' event with:
    - normalization_available: whether text normalization deps are installed
    - settings: current user settings dict

    Args:
        client_id: Optional client SID to target. If None, broadcasts to all.
    """
    global _config_sent_once
    try:
        from server import PromptServer
        if PromptServer.instance is not None:
            from .settings import get_settings
            from .model_info import TEXT_NORMALIZATION_AVAILABLE
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
                logger.info(
                    "Text normalization packages (inflect, wetext) not found. "
                    "Normalization will be disabled. Install them using: pip install inflect wetext"
                )
    except Exception as e:
        logger.warning(f"Failed to send config: {e}")


def send_config_event():
    """Send configuration to frontend (legacy function for compatibility).

    Delegates to send_config_to_client() for backward compatibility.
    """
    send_config_to_client()


def schedule_config_send():
    """Schedule config to be sent once after server starts using threading.

    Uses a 3-second delay to allow the server to fully initialize before
    sending the configuration event. The thread is daemon-mode so it won't
    block process shutdown.
    """
    def send_after_delay():
        time.sleep(3)
        logger.debug("Sending initial config event...")
        send_config_event()
        logger.debug("Initial config event sent")

    thread = threading.Thread(target=send_after_delay, daemon=True)
    thread.start()
