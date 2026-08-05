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

# Track if the client connect handler has been registered
_connect_handler_registered = False


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
    This is a fallback — the primary mechanism is register_config_on_connect()
    which sends config when a client connects via WebSocket.

    The thread is daemon-mode so it won't block process shutdown.
    """
    def send_after_delay():
        time.sleep(0.5)  # Reduced from 3s — just enough for server startup
        logger.debug("Sending initial config event...")
        send_config_event()
        logger.debug("Initial config event sent")

    thread = threading.Thread(target=send_after_delay, daemon=True)
    thread.start()


def register_config_on_connect():
    """Register a handler to send config when a client connects.

    This replaces the fragile time.sleep(3) approach with event-driven
    config broadcast. When a WebSocket client connects, we immediately
    send the config event to that specific client.

    This is the primary mechanism for config broadcast. The schedule_config_send()
    fallback is kept for backward compatibility but with a reduced delay.

    Safe to call multiple times — only registers the handler once.
    """
    global _connect_handler_registered
    if _connect_handler_registered:
        logger.debug("Config on-connect handler already registered")
        return

    try:
        from server import PromptServer
        if PromptServer.instance is None:
            logger.debug("PromptServer.instance is None, skipping connect handler registration")
            return

        # ComfyUI's PromptServer uses aiohttp WebSocket.
        # The on_client_connect callback is called when a new client connects.
        # We hook into this to send config immediately to the new client.
        async def on_client_connect(client_id):
            logger.debug(f"Client connected: {client_id}, sending config")
            send_config_to_client(client_id)

        # Set the callback on PromptServer.instance
        # Note: The exact attribute name may vary by ComfyUI version.
        # We try several common attribute names.
        for attr_name in ("on_client_connect", "on_connect", "client_connect_handler"):
            if hasattr(PromptServer.instance, attr_name):
                setattr(PromptServer.instance, attr_name, on_client_connect)
                logger.debug(f"Registered config on-connect handler via {attr_name}")
                _connect_handler_registered = True
                return

        # If no standard attribute found, log a warning
        logger.warning(
            "Could not find client connect handler attribute on PromptServer. "
            "Falling back to schedule_config_send() only."
        )

    except ImportError as e:
        logger.warning(f"Failed to import server module for connect handler: {e}")
    except AttributeError as e:
        logger.warning(f"Failed to access PromptServer for connect handler: {e}")
    except Exception as e:
        logger.error(f"Unexpected error registering connect handler: {e}", exc_info=True)
