"""
Custom Types for VoxCPM Nodes.

This module defines custom ComfyUI types for passing configuration
between nodes in a modular architecture.

Types:
- VoiceCloningConfig: Configuration for audio-based voice cloning (prompt/reference audio)
- AdvancedParams: Advanced generation parameters

Note: Voice design (text-based) is now a direct parameter on the main TTS node,
not part of VoiceCloningConfig.
"""

import torch
from typing import Optional
from comfy_api.latest import io


# =============================================================================
# VoiceCloningConfig Type
# =============================================================================

# Create the custom type using io.Custom
VoiceCloningConfig = io.Custom("VOICE_CLONING_CONFIG")


# =============================================================================
# AdvancedParams Type
# =============================================================================

# Create the custom type using io.Custom
AdvancedParams = io.Custom("ADVANCED_PARAMS")


# =============================================================================
# Helper Functions
# =============================================================================

def create_default_voice_config() -> dict:
    """
    Create default voice cloning configuration.

    Note: voice_design is now a direct parameter on the main TTS node,
    not part of this configuration.

    Returns:
        dict: Default configuration with all values set to None/False
    """
    return {
        "prompt_text": None,
        "prompt_waveform": None,
        "prompt_sample_rate": None,
        "reference_waveform": None,
        "reference_sample_rate": None,
        "trim_silence": False
    }


def create_default_advanced_params() -> dict:
    """
    Create default advanced parameters.

    Returns:
        dict: Default parameters matching VoxCPMNode defaults
    """
    return {
        "min_tokens": 2,
        "max_tokens": 2048,
        "temperature": 1.0,
        "sway_sampling_coef": 1.0,
        "use_cfg_zero_star": True,
        "retry_max_attempts": 3,
        "retry_threshold": 6.0
    }


def validate_voice_config(config: dict) -> tuple[bool, str]:
    """
    Validate voice cloning configuration.

    Note: voice_design is now a direct parameter on the main TTS node,
    not part of this configuration.

    Supports two audio-based cloning modes:
    1. Controllable Cloning: reference_audio only (VoxCPM2)
    2. Ultimate Cloning: prompt_audio + prompt_text (optionally with reference_audio)

    Args:
        config: Voice cloning configuration dict

    Returns:
        tuple: (is_valid, error_message)
        - is_valid: True if configuration is valid
        - error_message: Error message if invalid, empty string if valid
    """
    has_prompt_audio = config.get("prompt_waveform") is not None
    has_prompt_text = config.get("prompt_text") is not None
    has_reference_audio = config.get("reference_waveform") is not None

    # Mode 1: Controllable Cloning - reference_audio only (no prompt_text needed)
    # Mode 2: Ultimate Cloning - prompt_audio + prompt_text required

    # If prompt_audio is provided, prompt_text is required
    if has_prompt_audio and not has_prompt_text:
        return False, "prompt_text is required when providing prompt_audio"

    # If prompt_text is provided without prompt_audio, that's an error
    # (user might have forgotten to connect audio)
    if has_prompt_text and not has_prompt_audio:
        return False, "prompt_audio is required when providing prompt_text"

    # Empty config is valid - just means basic TTS without voice cloning
    # The main TTS node will use voice_design parameter if provided

    return True, ""


def validate_advanced_params(params: dict) -> tuple[bool, str]:
    """
    Validate advanced parameters.

    Args:
        params: Advanced parameters dict

    Returns:
        tuple: (is_valid, error_message)
        - is_valid: True if parameters are valid
        - error_message: Error message if invalid, empty string if valid
    """
    # Validate min_tokens
    min_tokens = params.get("min_tokens", 2)
    if not isinstance(min_tokens, int) or min_tokens < 1:
        return False, "min_tokens must be a positive integer"

    # Validate max_tokens
    max_tokens = params.get("max_tokens", 2048)
    if not isinstance(max_tokens, int) or max_tokens < 64:
        return False, "max_tokens must be at least 64"

    if min_tokens > max_tokens:
        return False, "min_tokens cannot be greater than max_tokens"

    # Validate temperature
    temperature = params.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or temperature < 0.1 or temperature > 2.0:
        return False, "temperature must be between 0.1 and 2.0"

    # Validate sway_sampling_coef
    sway = params.get("sway_sampling_coef", 1.0)
    if not isinstance(sway, (int, float)) or sway < 0.0 or sway > 2.0:
        return False, "sway_sampling_coef must be between 0.0 and 2.0"

    # Validate retry_max_attempts
    retry_attempts = params.get("retry_max_attempts", 3)
    if not isinstance(retry_attempts, int) or retry_attempts < 0 or retry_attempts > 10:
        return False, "retry_max_attempts must be between 0 and 10"

    # Validate retry_threshold
    retry_threshold = params.get("retry_threshold", 6.0)
    if not isinstance(retry_threshold, (int, float)) or retry_threshold < 2.0 or retry_threshold > 20.0:
        return False, "retry_threshold must be between 2.0 and 20.0"

    return True, ""


def merge_voice_config_with_defaults(config: Optional[dict]) -> dict:
    """
    Merge voice config with defaults, handling None input.

    Args:
        config: Voice cloning configuration dict or None

    Returns:
        dict: Merged configuration with all keys present
    """
    defaults = create_default_voice_config()
    if config is None:
        return defaults

    # Merge with provided config (config takes precedence)
    result = defaults.copy()
    result.update({k: v for k, v in config.items() if v is not None})
    return result


def merge_advanced_params_with_defaults(params: Optional[dict]) -> dict:
    """
    Merge advanced params with defaults, handling None input.

    Args:
        params: Advanced parameters dict or None

    Returns:
        dict: Merged parameters with all keys present
    """
    defaults = create_default_advanced_params()
    if params is None:
        return defaults

    # Merge with provided params (params takes precedence)
    result = defaults.copy()
    result.update({k: v for k, v in params.items() if v is not None})
    return result


def merge_voice_config(
    config: Optional[dict],
    prompt_text: Optional[str] = None,
    prompt_waveform: Optional[torch.Tensor] = None,
    prompt_sample_rate: Optional[int] = None,
    reference_waveform: Optional[torch.Tensor] = None,
    reference_sample_rate: Optional[int] = None,
    trim_silence: Optional[bool] = None
) -> dict:
    """
    Merge voice config with direct parameters.

    Direct parameters take precedence over config values.
    This is used in the main TTS node to combine config node
    inputs with direct node inputs.

    Note: voice_design is now a direct parameter on the main TTS node,
    not part of this configuration.

    Args:
        config: Voice cloning configuration dict or None
        prompt_text: Direct prompt text parameter
        prompt_waveform: Direct prompt waveform parameter
        prompt_sample_rate: Direct prompt sample rate parameter
        reference_waveform: Direct reference waveform parameter
        reference_sample_rate: Direct reference sample rate parameter
        trim_silence: Direct trim silence parameter

    Returns:
        dict: Merged configuration with direct params taking precedence
    """
    # Start with defaults
    result = create_default_voice_config()

    # Apply config values first
    if config is not None:
        result.update({k: v for k, v in config.items() if v is not None})

    # Apply direct parameters (these take precedence)
    if prompt_text is not None:
        result['prompt_text'] = prompt_text
    if prompt_waveform is not None:
        result['prompt_waveform'] = prompt_waveform
    if prompt_sample_rate is not None:
        result['prompt_sample_rate'] = prompt_sample_rate
    if reference_waveform is not None:
        result['reference_waveform'] = reference_waveform
    if reference_sample_rate is not None:
        result['reference_sample_rate'] = reference_sample_rate
    if trim_silence is not None:
        result['trim_silence'] = trim_silence

    return result


def merge_advanced_params(
    config: Optional[dict],
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    sway_sampling_coef: Optional[float] = None,
    use_cfg_zero_star: Optional[bool] = None,
    retry_max_attempts: Optional[int] = None,
    retry_threshold: Optional[float] = None
) -> dict:
    """
    Merge advanced params with direct parameters.

    Direct parameters take precedence over config values.
    This is used in the main TTS node to combine config node
    inputs with direct node inputs.

    Args:
        config: Advanced parameters dict or None
        min_tokens: Direct min tokens parameter
        max_tokens: Direct max tokens parameter
        temperature: Direct temperature parameter
        sway_sampling_coef: Direct sway sampling coefficient parameter
        use_cfg_zero_star: Direct CFG-Zero* parameter
        retry_max_attempts: Direct retry max attempts parameter
        retry_threshold: Direct retry threshold parameter

    Returns:
        dict: Merged parameters with direct params taking precedence
    """
    # Start with defaults
    result = create_default_advanced_params()

    # Apply config values first
    if config is not None:
        result.update({k: v for k, v in config.items() if v is not None})

    # Apply direct parameters (these take precedence)
    if min_tokens is not None:
        result['min_tokens'] = min_tokens
    if max_tokens is not None:
        result['max_tokens'] = max_tokens
    if temperature is not None:
        result['temperature'] = temperature
    if sway_sampling_coef is not None:
        result['sway_sampling_coef'] = sway_sampling_coef
    if use_cfg_zero_star is not None:
        result['use_cfg_zero_star'] = use_cfg_zero_star
    if retry_max_attempts is not None:
        result['retry_max_attempts'] = retry_max_attempts
    if retry_threshold is not None:
        result['retry_threshold'] = retry_threshold

    return result
