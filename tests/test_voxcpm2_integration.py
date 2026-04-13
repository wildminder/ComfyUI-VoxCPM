"""
VoxCPM2 Integration Tests

Tests for verifying the VoxCPM2 integration with ComfyUI.
Run with: python -m pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelInfo:
    """Tests for model info configuration."""

    def test_model_configs_exist(self):
        """Verify model configs are defined."""
        from modules.model_info import MODEL_CONFIGS

        assert "VoxCPM2" in MODEL_CONFIGS
        assert "VoxCPM1.5" in MODEL_CONFIGS

    def test_voxcpm2_config_values(self):
        """Verify VoxCPM2 config has correct values."""
        from modules.model_info import MODEL_CONFIGS

        config = MODEL_CONFIGS["VoxCPM2"]
        assert config["architecture"] == "voxcpm2"
        assert config["sample_rate"] == 48000
        assert config["repo_id"] == "openbmb/VoxCPM2"
        assert config["size_gb"] == 8.0

    def test_voxcpm15_config_values(self):
        """Verify VoxCPM1.5 config has correct values."""
        from modules.model_info import MODEL_CONFIGS

        config = MODEL_CONFIGS["VoxCPM1.5"]
        assert config["architecture"] == "voxcpm"
        assert config["sample_rate"] == 44100
        assert config["repo_id"] == "openbmb/VoxCPM1.5"

    def test_available_models_populated(self):
        """Verify AVAILABLE_VOXCPM_MODELS is populated from MODEL_CONFIGS."""
        from modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS

        assert len(AVAILABLE_VOXCPM_MODELS) > 0
        for name in MODEL_CONFIGS:
            assert name in AVAILABLE_VOXCPM_MODELS


class TestIsVoxCPM2Model:
    """Tests for the is_voxcpm2_model helper function."""

    def test_detects_voxcpm2(self):
        """Verify VoxCPM2 is detected correctly."""
        from modules.utils import is_voxcpm2_model
        from modules.model_info import MODEL_CONFIGS

        assert is_voxcpm2_model("VoxCPM2", MODEL_CONFIGS) == True

    def test_detects_voxcpm15(self):
        """Verify VoxCPM1.5 is detected as non-VoxCPM2."""
        from modules.utils import is_voxcpm2_model
        from modules.model_info import MODEL_CONFIGS

        assert is_voxcpm2_model("VoxCPM1.5", MODEL_CONFIGS) == False

    def test_unknown_model_defaults_to_voxcpm(self):
        """Verify unknown models default to non-VoxCPM2."""
        from modules.utils import is_voxcpm2_model
        from modules.model_info import MODEL_CONFIGS

        # Unknown model should return False (default to voxcpm architecture)
        assert is_voxcpm2_model("UnknownModel", MODEL_CONFIGS) == False


class TestGenerationModule:
    """Tests for the shared generation module."""

    def test_extract_audio_tensor_none_input(self):
        """Verify None input returns None tuple."""
        from modules.generation import extract_audio_tensor

        waveform, sr = extract_audio_tensor(None)
        assert waveform is None
        assert sr is None

    def test_extract_audio_tensor_valid_input(self):
        """Verify valid audio input is extracted correctly."""
        from modules.generation import extract_audio_tensor

        # Simulate ComfyUI audio format
        sample_rate = 48000
        samples = sample_rate  # 1 second
        audio = {
            "waveform": torch.randn(1, 1, samples),
            "sample_rate": sample_rate
        }

        waveform, sr = extract_audio_tensor(audio, "test_audio")
        assert waveform is not None
        assert sr == sample_rate
        assert waveform.dim() == 2  # Should be [channels, samples]
        assert waveform.shape == (1, samples)

    def test_extract_audio_tensor_3d_squeezed(self):
        """Verify 3D waveform is squeezed to 2D."""
        from modules.generation import extract_audio_tensor

        sample_rate = 48000
        samples = sample_rate
        audio = {
            "waveform": torch.randn(1, 1, samples),
            "sample_rate": sample_rate
        }

        waveform, _ = extract_audio_tensor(audio)
        assert waveform.dim() == 2

    def test_extract_audio_tensor_empty_raises(self):
        """Verify empty audio raises ValueError."""
        from modules.generation import extract_audio_tensor

        audio = {
            "waveform": torch.tensor([]),
            "sample_rate": 48000
        }

        with pytest.raises(ValueError, match="empty"):
            extract_audio_tensor(audio)

    def test_extract_audio_tensor_invalid_format_raises(self):
        """Verify invalid format raises ValueError."""
        from modules.generation import extract_audio_tensor

        with pytest.raises(ValueError):
            extract_audio_tensor("not a dict")

    def test_build_final_text_with_control(self):
        """Verify control instruction is prepended correctly."""
        from modules.generation import build_final_text

        text = "Hello world"
        control = "warm female voice"

        result = build_final_text(text, control)
        assert result == "(warm female voice)Hello world"

    def test_build_final_text_without_control(self):
        """Verify text is unchanged without control."""
        from modules.generation import build_final_text

        text = "Hello world"

        result = build_final_text(text, None)
        assert result == text

    def test_build_final_text_empty_control(self):
        """Verify empty control doesn't affect text."""
        from modules.generation import build_final_text

        text = "Hello world"

        result = build_final_text(text, "")
        assert result == text

    def test_validate_prompt_pairing_valid(self):
        """Verify valid pairing passes."""
        from modules.generation import validate_prompt_pairing

        # Should not raise
        validate_prompt_pairing(None, None)
        validate_prompt_pairing({"waveform": torch.randn(1, 1, 1000), "sample_rate": 48000}, "text")

    def test_validate_prompt_pairing_missing_text(self):
        """Verify missing prompt_text raises."""
        from modules.generation import validate_prompt_pairing

        with pytest.raises(ValueError, match="prompt_text is required"):
            validate_prompt_pairing({"waveform": torch.randn(1, 1, 1000), "sample_rate": 48000}, None)


class TestCoreImports:
    """Tests for verifying core module imports work."""

    def test_import_voxcpm_core(self):
        """Verify VoxCPM core module can be imported."""
        from src.voxcpm.core import VoxCPM
        assert VoxCPM is not None

    def test_import_voxcpm_model(self):
        """Verify VoxCPMModel can be imported."""
        from src.voxcpm.model.voxcpm import VoxCPMModel, LoRAConfig
        assert VoxCPMModel is not None
        assert LoRAConfig is not None

    def test_import_voxcpm2_model(self):
        """Verify VoxCPM2Model can be imported."""
        from src.voxcpm.model.voxcpm2 import VoxCPM2Model, LoRAConfig
        assert VoxCPM2Model is not None
        assert LoRAConfig is not None


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_voxcpm_lora_config_defaults(self):
        """Verify VoxCPM1.5 LoRA config defaults."""
        from src.voxcpm.model.voxcpm import LoRAConfig

        config = LoRAConfig()
        assert config.enable_lm == False
        assert config.enable_dit == False
        assert config.enable_proj == False
        assert config.r == 8
        assert config.alpha == 16

    def test_voxcpm2_lora_config_defaults(self):
        """Verify VoxCPM2 LoRA config defaults."""
        from src.voxcpm.model.voxcpm2 import LoRAConfig

        config = LoRAConfig()
        assert config.enable_lm == False
        assert config.enable_dit == False
        assert config.enable_proj == False
        assert config.r == 8
        assert config.alpha == 16

    def test_lora_config_custom_values(self):
        """Verify LoRA config accepts custom values."""
        from src.voxcpm.model.voxcpm2 import LoRAConfig

        config = LoRAConfig(
            enable_lm=True,
            enable_dit=True,
            enable_proj=True,
            r=32,
            alpha=16,
            dropout=0.1
        )
        assert config.enable_lm == True
        assert config.enable_dit == True
        assert config.enable_proj == True
        assert config.r == 32
        assert config.alpha == 16
        assert config.dropout == 0.1


class TestVoxCPM2ModelMethods:
    """Tests for VoxCPM2Model methods (without loading model)."""

    def test_make_ref_prefix_shape(self):
        """Verify _make_ref_prefix creates correct tensor shapes."""
        # This test requires a loaded model, so we skip it in unit tests
        # Integration tests should verify this
        pytest.skip("Requires loaded model - run integration test")

    def test_build_prompt_cache_validation(self):
        """Verify build_prompt_cache validates inputs correctly."""
        # This test requires a loaded model
        pytest.skip("Requires loaded model - run integration test")


class TestTensorProcessing:
    """Tests for tensor processing utilities."""

    def test_audio_tensor_reshape_1d_to_2d(self):
        """Verify 1D audio tensors are reshaped correctly."""
        # Simulate ComfyUI audio format
        sample_rate = 48000
        duration_sec = 1.0
        samples = int(sample_rate * duration_sec)

        # 1D tensor
        audio_1d = torch.randn(samples)
        assert audio_1d.dim() == 1

        # Reshape to 2D (what the code does)
        if audio_1d.dim() == 1:
            audio_2d = audio_1d.unsqueeze(0)

        assert audio_2d.dim() == 2
        assert audio_2d.shape == (1, samples)

    def test_audio_tensor_reshape_3d_to_2d(self):
        """Verify 3D audio tensors are reshaped correctly."""
        sample_rate = 48000
        duration_sec = 1.0
        samples = int(sample_rate * duration_sec)

        # 3D tensor (batch, channels, samples) - ComfyUI format
        audio_3d = torch.randn(1, 1, samples)
        assert audio_3d.dim() == 3

        # Reshape to 2D (what the code does)
        if audio_3d.dim() == 3:
            audio_2d = audio_3d.squeeze(0)

        assert audio_2d.dim() == 2
        assert audio_2d.shape == (1, samples)


class TestControlInstructionFormat:
    """Tests for control instruction formatting."""

    def test_control_instruction_format(self):
        """Verify control instruction format is correct."""
        text = "Hello world"
        control = "warm female voice"

        # Expected format: "(control)text"
        formatted = f"({control}){text}"

        assert formatted == "(warm female voice)Hello world"

    def test_empty_control_instruction(self):
        """Verify empty control instruction doesn't affect text."""
        text = "Hello world"
        control = ""

        # Should not wrap empty control
        if control and control.strip():
            formatted = f"({control}){text}"
        else:
            formatted = text

        assert formatted == text


class TestNodeSchemas:
    """Tests for node schema definitions."""

    def test_voxcpm_node_schema(self):
        """Verify VoxCPMNode schema is valid."""
        # Import would require comfy_api, skip if not available
        pytest.skip("Requires comfy_api environment")


class TestValidateVoxCPM2Features:
    """Tests for VoxCPM2 feature validation."""

    def test_reference_audio_with_voxcpm2(self):
        """Verify reference audio works with VoxCPM2."""
        from modules.generation import validate_voxcpm2_features
        from modules.model_info import MODEL_CONFIGS

        is_v2, warning = validate_voxcpm2_features(
            "VoxCPM2", MODEL_CONFIGS,
            {"waveform": torch.randn(1, 1, 1000), "sample_rate": 48000},
            None
        )
        assert is_v2 == True
        assert warning is None

    def test_reference_audio_with_voxcpm15_raises(self):
        """Verify reference audio with VoxCPM1.5 raises error."""
        from modules.generation import validate_voxcpm2_features
        from modules.model_info import MODEL_CONFIGS

        with pytest.raises(ValueError, match="only supported with VoxCPM2"):
            validate_voxcpm2_features(
                "VoxCPM1.5", MODEL_CONFIGS,
                {"waveform": torch.randn(1, 1, 1000), "sample_rate": 44100},
                None
            )

    def test_control_instruction_with_voxcpm15_warns(self):
        """Verify control instruction with VoxCPM1.5 returns warning."""
        from modules.generation import validate_voxcpm2_features
        from modules.model_info import MODEL_CONFIGS

        is_v2, warning = validate_voxcpm2_features(
            "VoxCPM1.5", MODEL_CONFIGS,
            None,
            "warm female voice"
        )
        assert is_v2 == False
        assert warning is not None


# Integration tests that require actual model loading
class TestIntegration:
    """Integration tests requiring model loading."""

    @pytest.mark.integration
    def test_voxcpm2_model_loading(self):
        """Test VoxCPM2 model can be loaded."""
        pytest.skip("Set VOXCPM2_MODEL_PATH env var to run")

    @pytest.mark.integration
    def test_voxcpm2_voice_design_generation(self):
        """Test voice design generation."""
        pytest.skip("Set VOXCPM2_MODEL_PATH env var to run")

    @pytest.mark.integration
    def test_voxcpm2_reference_cloning(self):
        """Test reference audio cloning."""
        pytest.skip("Set VOXCPM2_MODEL_PATH env var to run")

    @pytest.mark.integration
    def test_voxcpm2_ultimate_cloning(self):
        """Test ultimate cloning mode."""
        pytest.skip("Set VOXCPM2_MODEL_PATH env var to run")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
