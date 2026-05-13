import sys
import os
import importlib.util
from unittest.mock import MagicMock
import pytest

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this path is correct
COMFYUI_ROOT = r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"

# ====================================================================
# 1. PATH CONFIGURATION
# ====================================================================
if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ====================================================================
# 2. MOCK ONLY THE NETWORK SERVER
# ====================================================================
# We let comfy and folder_paths load normally from COMFYUI_ROOT.
# We only mock 'server' and 'aiohttp' to prevent network activity.
_mock_server = MagicMock()
_mock_prompt_server_instance = MagicMock()
_mock_server.PromptServer.instance = _mock_prompt_server_instance
sys.modules["server"] = _mock_server
sys.modules["aiohttp"] = MagicMock()
sys.modules["aiohttp.web"] = MagicMock()

# ====================================================================
# 3. NATIVE ALIAS FOR THE CUSTOM NODE PACKAGE
# ====================================================================
if "ComfyUI_VoxCPM" not in sys.modules:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_VoxCPM", 
        os.path.join(ROOT_DIR, "__init__.py"),
        submodule_search_locations=[ROOT_DIR]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_VoxCPM"] = pkg
    spec.loader.exec_module(pkg)

# ====================================================================
# 4. GLOBAL FIXTURES
# ====================================================================
@pytest.fixture
def mock_prompt_server():
    """Provide a shared PromptServer mock with reset send_sync.

    All downloader test files should use this fixture to get the
    PromptServer.instance mock. It ensures cross-file isolation
    by re-assigning the instance on the shared server mock.
    """
    _mock_server.PromptServer.instance = _mock_prompt_server_instance
    _mock_prompt_server_instance.send_sync.reset_mock()
    return _mock_prompt_server_instance


@pytest.fixture(autouse=True)
def temp_settings_dir(tmp_path):
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir

@pytest.fixture(autouse=True)
def temp_model_dir(tmp_path):
    model_dir = tmp_path / "models" / "tts" / "VoxCPM"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"arch": "voxcpm2"}')
    return model_dir

@pytest.fixture(autouse=True)
def comfyui_env():
    return True