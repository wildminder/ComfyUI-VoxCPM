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
    except ImportError as e:
        logger.warning(f"Failed to import required module for config broadcast: {e}")
    except AttributeError as e:
        logger.warning(f"Failed to access PromptServer for config broadcast: {e}")
    except (ConnectionError, OSError) as e:
        logger.warning(f"Network error sending config: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending config: {e}", exc_info=True)


def send_config_event():
    """Send configuration to frontend (legacy function for compatibility).

    Delegates to send_config_to_client() for backward compatibility.
    """
    send_config_to_client()


def schedule_config_send():
    """Schedule config to be sent once after server starts using threading.

    Uses a short delay (0.5s) for server startup, then sends.

    This is a best-effort one-shot broadcast: it only reaches tabs that
    are already connected when the server boots. Fresh tabs that connect
    later are covered by the frontend, which fetches /voxcpm/model_info
    (a superset of the ws payload) at extension load time and applies it
    via the same handler as the voxcpm.config event.

    The thread is daemon-mode so it won't block process shutdown.
    """
    def send_after_delay():
        time.sleep(0.5)  # Reduced from 3s — just enough for server startup
        logger.debug("Sending initial config event...")
        send_config_event()
        logger.debug("Initial config event sent")

    thread = threading.Thread(target=send_after_delay, daemon=True)
    thread.start()
