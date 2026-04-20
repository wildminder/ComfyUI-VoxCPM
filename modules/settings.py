"""
VoxCPM User Settings Management

Handles persistence of user preferences including:
- Model path preference (default/custom)
- Custom model path
- First-run flag

Uses ComfyUI's folder_paths for proper integration.
"""

import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

SETTINGS_FILE = "voxcpm_settings.json"


class VoxCPMSettings:
    """Manages user settings for VoxCPM using ComfyUI's user directory."""

    def __init__(self):
        self._settings: Dict[str, Any] = {}
        self._settings_path: Optional[str] = None
        self._load_settings()

    def _get_settings_path(self) -> str:
        """Get the path to the settings file in ComfyUI's user directory."""
        try:
            import folder_paths

            user_dir = folder_paths.get_user_directory()
            if user_dir:
                # Store in user/default/VoxCPM/voxcpm_settings.json
                voxcpm_user_dir = os.path.join(user_dir, "default", "VoxCPM")
                os.makedirs(voxcpm_user_dir, exist_ok=True)
                return os.path.join(voxcpm_user_dir, SETTINGS_FILE)
        except ImportError:
            pass

        # Fallback to custom node directory
        return SETTINGS_FILE

    def _load_settings(self) -> None:
        """Load settings from disk."""
        try:
            self._settings_path = self._get_settings_path()
            if os.path.exists(self._settings_path):
                with open(self._settings_path, "r", encoding="utf-8") as f:
                    self._settings = json.load(f)
                logger.debug(f"Loaded settings from {self._settings_path}")
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
            self._settings = {}

    def _save_settings(self) -> bool:
        """Save settings to disk."""
        try:
            if self._settings_path:
                with open(self._settings_path, "w", encoding="utf-8") as f:
                    json.dump(self._settings, f, indent=2)
                logger.debug(f"Saved settings to {self._settings_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
        return False

    @property
    def is_first_run(self) -> bool:
        """Check if this is the first run."""
        return self._settings.get("first_run", True)

    def mark_first_run_complete(self) -> None:
        """Mark first run as complete."""
        self._settings["first_run"] = False
        self._save_settings()

    @property
    def use_custom_path(self) -> bool:
        """Check if user wants to use custom path."""
        return self._settings.get("use_custom_path", False)

    def set_use_custom_path(self, value: bool) -> None:
        """Set custom path preference."""
        self._settings["use_custom_path"] = value
        self._save_settings()

    @property
    def custom_model_path(self) -> Optional[str]:
        """Get custom model path."""
        return self._settings.get("custom_model_path")

    def set_custom_model_path(self, path: str) -> bool:
        """Set custom model path. Returns True if valid."""
        if self._validate_model_path(path):
            self._settings["custom_model_path"] = path
            self._save_settings()
            return True
        return False

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

    def to_dict(self) -> Dict[str, Any]:
        """Export settings for frontend."""
        return {
            "first_run": self.is_first_run,
            "use_custom_path": self.use_custom_path,
            "custom_model_path": self.custom_model_path,
            "effective_path": self.get_effective_model_path(),
        }


# Global settings instance
_settings_instance: Optional[VoxCPMSettings] = None


def get_settings() -> VoxCPMSettings:
    """Get the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = VoxCPMSettings()
    return _settings_instance
