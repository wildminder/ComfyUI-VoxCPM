"""
VoxCPM User Settings Management

Reads user preferences from ComfyUI's built-in comfy.settings.json file.
The frontend writes settings via app.api.storeSetting("voxcpm.<key>", value)
which stores them in user/default/comfy.settings.json. This module reads
those settings on the backend side.

Settings keys (all prefixed with "voxcpm."):
- voxcpm.use_custom_path (bool): Whether to use a custom model path
- voxcpm.custom_model_path (str): The custom model directory path

This module is READ-ONLY — the frontend is the single write path via
ComfyUI's settings API (POST /settings/voxcpm.<key>). The backend
only reads these settings to determine the effective model path.
"""

import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Legacy settings file name (used for migration cleanup)
SETTINGS_FILE = "voxcpm_settings.json"

# Settings key prefix in comfy.settings.json
SETTINGS_PREFIX = "voxcpm."


class VoxCPMSettings:
    """Reads VoxCPM settings from ComfyUI's comfy.settings.json file.

    This class is read-only. Settings are written by the frontend via
    ComfyUI's settings API (app.api.storeSetting). The backend reads
    them to determine the effective model path and broadcast config
    to the frontend.
    """

    def __init__(self):
        self._settings: Dict[str, Any] = {}
        self._load_settings()

    def _get_comfy_settings_path(self) -> Optional[str]:
        """Get the path to comfy.settings.json in the user directory."""
        try:
            import folder_paths

            user_dir = folder_paths.get_user_directory()
            if user_dir:
                return os.path.join(user_dir, "default", "comfy.settings.json")
        except ImportError:
            pass
        return None

    def _load_settings(self) -> None:
        """Load settings from comfy.settings.json."""
        try:
            settings_path = self._get_comfy_settings_path()
            if settings_path and os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    all_settings = json.load(f)
                # Filter to only voxcpm.* keys
                self._settings = {
                    k: v for k, v in all_settings.items()
                    if k.startswith(SETTINGS_PREFIX)
                }
                logger.debug(f"Loaded VoxCPM settings from comfy.settings.json: {list(self._settings.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load settings from comfy.settings.json: {e}")
            self._settings = {}

    def reload(self) -> None:
        """Re-read settings from comfy.settings.json.

        Call this after the frontend writes new settings via storeSetting
        to ensure the backend has fresh data.
        """
        self._load_settings()

    @property
    def use_custom_path(self) -> bool:
        """Check if user wants to use custom path."""
        return self._settings.get(f"{SETTINGS_PREFIX}use_custom_path", False)

    @property
    def custom_model_path(self) -> Optional[str]:
        """Get custom model path."""
        return self._settings.get(f"{SETTINGS_PREFIX}custom_model_path")

    def _validate_model_path(self, path: str) -> bool:
        """Validate that path exists and contains valid model structure."""
        if not os.path.isdir(path):
            return False
        # Check for at least one valid model
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    has_config = os.path.exists(
                        os.path.join(item_path, "config.json")
                    )
                    has_weights = (
                        os.path.exists(
                            os.path.join(item_path, "pytorch_model.bin")
                        )
                        or os.path.exists(
                            os.path.join(item_path, "model.safetensors")
                        )
                        or os.path.exists(
                            os.path.join(item_path, "model-00001-of-00001.safetensors")
                        )
                    )
                    if has_config and has_weights:
                        return True
        except OSError:
            return False
        return False

    def get_effective_model_path(self) -> str:
        """Get the effective model path based on settings."""
        if self.use_custom_path and self.custom_model_path:
            return self.custom_model_path
        # Default path
        try:
            import folder_paths

            tts_paths = folder_paths.get_folder_paths("tts")
            if tts_paths:
                return os.path.join(tts_paths[0], "VoxCPM")
            return os.path.join(folder_paths.models_dir, "tts", "VoxCPM")
        except ImportError:
            return os.path.join("models", "tts", "VoxCPM")

    def is_effective_path_valid(self) -> bool:
        """Check if the effective model path contains at least one valid model."""
        effective_path = self.get_effective_model_path()
        return self._validate_model_path(effective_path)

    def to_dict(self) -> Dict[str, Any]:
        """Export settings for frontend."""
        return {
            "use_custom_path": self.use_custom_path,
            "custom_model_path": self.custom_model_path,
            "effective_path": self.get_effective_model_path(),
            "effective_path_valid": self.is_effective_path_valid(),
        }


# Global settings instance
_settings_instance: Optional[VoxCPMSettings] = None


def get_settings() -> VoxCPMSettings:
    """Get the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = VoxCPMSettings()
    return _settings_instance
