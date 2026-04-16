"""
VoxCPM TTS Node - Main Text-to-Speech Node.

This is the main TTS node supporting both VoxCPM1.5 and VoxCPM2 models.
Features:
- Basic TTS synthesis (all models)
- LoRA support for style customization

Voice cloning and advanced parameters are configured via optional input nodes:
- VoxCPMVoiceCloning node (voice_config) - for voice design, prompt/reference audio
- VoxCPMAdvancedParams node (advanced_params) - for generation parameters
"""

import torch
import logging
from typing import Optional

import folder_paths
import comfy.model_management as model_management
from comfy_api.latest import io, ui

from ..modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS
from ..modules.generation import (
    load_voxcpm_model,
    generate_audio,
    build_final_text,
    validate_voxcpm2_features,
)
from ..modules.utils import get_available_devices, set_seed
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.custom_types import (
    VoiceCloningConfig,
    AdvancedParams,
)
from ..src.voxcpm.utils.text_normalize import TEXT_NORMALIZATION_AVAILABLE

logger = logging.getLogger(__name__)


class VoxCPMNode(io.ComfyNode):
    """
    Unified TTS node supporting both VoxCPM1.5 and VoxCPM2 models.

    Features:
    - Basic TTS synthesis (all models)
    - LoRA support for style customization

    Voice cloning and advanced parameters are configured via optional input nodes:
    - VoxCPMVoiceCloning node (voice_config) - for voice design, prompt/reference audio
    - VoxCPMAdvancedParams node (advanced_params) - for generation parameters
    """

    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls):
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())
        if not model_names:
            model_names.append("No models found. Please download a VoxCPM model.")

        available_devices = get_available_devices()
        default_device = available_devices[0]

        dtype_options = get_dtype_options()
        default_dtype = DTYPE_AUTO

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM_TTS",
            display_name="VoxCPM TTS",
            category=cls.CATEGORY,
            description="Generate speech using VoxCPM models. Connect VoxCPM Voice Cloning and Advanced Parameters nodes for voice cloning and fine-tuning.",
            inputs=[
                # Model selection
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM model to use. VoxCPM2 supports voice design and reference cloning."),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="Select a LoRA to apply for style/fine-tuning."),

            # Text input
            io.String.Input("text", multiline=True, default="VoxCPM is an innovative TTS model designed to generate highly expressive speech.", tooltip="Text to synthesize."),
            io.String.Input("voice_design", default="", tooltip="Voice design instruction for VoxCPM2 (e.g., 'warm female voice', 'deep male voice'). Only used in plain TTS mode (no reference audio). Ignored when reference audio is connected or for VoxCPM1.5."),

            # Generation parameters (basic)
                io.Float.Input("cfg", default=2.0, min=0.1, max=10.0, step=0.1, tooltip="Guidance scale. Higher values adhere more to the prompt but may sound less natural."),
                io.Int.Input("steps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps. Higher values may improve quality but are slower."),
                io.Boolean.Input(
                    "normalize_text",
                    default=TEXT_NORMALIZATION_AVAILABLE,
                    label_on="Normalize",
                    label_off="Raw",
                    tooltip="Enable text normalization (requires 'inflect' and 'wetext' packages)." if not TEXT_NORMALIZATION_AVAILABLE else "Enable text normalization (recommended for general text)."
                ),

                # System parameters
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Seed for reproducibility. -1 for random."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto-Manage", tooltip="Force the model to be offloaded from VRAM after generation."),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="Device to run inference on."),
                io.Combo.Input("dtype", options=dtype_options, default=default_dtype, tooltip="Data type for model precision. 'auto' selects optimal type for device."),

                # Optional configuration inputs (from config nodes)
                VoiceCloningConfig.Input("voice_config", optional=True, tooltip="Voice cloning configuration from VoxCPM Voice Cloning node. Provides voice_design, prompt_audio, reference_audio, trim_silence."),
                AdvancedParams.Input("advanced_params", optional=True, tooltip="Advanced parameters from VoxCPM Advanced Parameters node. Provides min/max_tokens, temperature, sway, use_cfg_zero_star, retry_*."),
            ],
            outputs=[
                io.Audio.Output(display_name="Generated Audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        lora_name: str,
        device: str,
        dtype: str,
        text: str,
        voice_design: str,
        cfg: float,
        steps: int,
        normalize_text: bool,
        seed: int,
        force_offload: bool,
        voice_config: Optional[dict] = None,
        advanced_params: Optional[dict] = None,
    ) -> io.NodeOutput:

        # =====================================================================
        # Extract parameters from config nodes or use defaults
        # =====================================================================

        # Voice design: direct parameter on this node (not from config)
        final_voice_design = voice_design.strip() if voice_design and voice_design.strip() else None

        # Extract voice cloning parameters from config
        if voice_config is not None:
            logger.debug("Using voice_config from config node")
            final_prompt_text = voice_config.get("prompt_text")
            final_trim_silence = voice_config.get("trim_silence", False)
            # Audio tensors from config
            prompt_waveform = voice_config.get("prompt_waveform")
            prompt_sample_rate = voice_config.get("prompt_sample_rate")
            ref_waveform = voice_config.get("reference_waveform")
            ref_sample_rate = voice_config.get("reference_sample_rate")
        else:
            # No voice config - basic TTS only
            final_prompt_text = None
            final_trim_silence = False
            prompt_waveform = None
            prompt_sample_rate = None
            ref_waveform = None
            ref_sample_rate = None

        # Extract advanced parameters from config or use defaults
        if advanced_params is not None:
            logger.debug("Using advanced_params from config node")
            final_min_tokens = advanced_params.get("min_tokens", 2)
            final_max_tokens = advanced_params.get("max_tokens", 2048)
            final_temperature = advanced_params.get("temperature", 1.0)
            final_sway = advanced_params.get("sway_sampling_coef", 1.0)
            final_cfg_zero = advanced_params.get("use_cfg_zero_star", True)
            final_retry_attempts = advanced_params.get("retry_max_attempts", 3)
            final_retry_threshold = advanced_params.get("retry_threshold", 6.0)
        else:
            # Use defaults
            final_min_tokens = 2
            final_max_tokens = 2048
            final_temperature = 1.0
            final_sway = 1.0
            final_cfg_zero = True
            final_retry_attempts = 3
            final_retry_threshold = 6.0

        # Send config event at execution time (fallback for browser refresh)
        # This ensures the frontend knows the normalization state even if WebSocket events were missed
        if not TEXT_NORMALIZATION_AVAILABLE:
            try:
                from server import PromptServer
                if PromptServer.instance is not None:
                    PromptServer.instance.send_sync("voxcpm.config", {
                        "normalization_available": False
                    })
            except Exception:
                pass

        # Send notification if text normalization is disabled and user tries to enable it
        if normalize_text and not TEXT_NORMALIZATION_AVAILABLE:
            try:
                from server import PromptServer
                if PromptServer.instance is not None:
                    PromptServer.instance.send_sync("voxcpm.status", {
                        "severity": "warn",
                        "summary": "VoxCPM Text Normalization Disabled",
                        "detail": "Optional packages 'inflect' and 'wetext' are not installed. Install with: pip install inflect wetext",
                        "life": 10000
                    })
            except Exception:
                pass

        # Validate VoxCPM2-only features (graceful degradation for non-fatal issues)
        is_voxcpm2, warning, ignore_reference = validate_voxcpm2_features(
            model_name, MODEL_CONFIGS,
            ref_waveform,
            final_voice_design
        )
        if warning:
            logger.warning(warning)
            final_voice_design = None  # Ignore for VoxCPM1.5

        # Handle reference audio for VoxCPM1.5 graceful degradation
        if ignore_reference:
            ref_waveform = None
            ref_sample_rate = None

        if (prompt_waveform is not None) or (ref_waveform is not None):
            final_text = text
            if final_voice_design:
                logger.info("Voice design ignored: reference audio takes precedence for voice cloning")
        else:
            final_text = build_final_text(text, final_voice_design)

        # Set seed for reproducibility
        set_seed(seed)

        # Load model with dtype
        patcher, model = load_voxcpm_model(model_name, device, dtype)

        # Apply LoRA if specified
        if lora_name and lora_name != "None":
            from ..modules.generation import apply_lora_if_needed
            apply_lora_if_needed(model, lora_name)

        try:
            # Generate audio
            output_tensor, sample_rate = generate_audio(
                model=model,
                text=final_text,
                cfg_value=cfg,
                inference_timesteps=steps,
                min_len=final_min_tokens,
                max_len=final_max_tokens,
                normalize=normalize_text,
                trim_silence_vad=final_trim_silence,
                temperature=final_temperature,
                sway_sampling_coef=final_sway,
                use_cfg_zero_star=final_cfg_zero,
                retry_badcase=final_retry_attempts > 0,
                retry_max_attempts=final_retry_attempts,
                retry_threshold=final_retry_threshold,
                prompt_text=final_prompt_text,
                prompt_waveform=prompt_waveform,
                prompt_sample_rate=prompt_sample_rate,
                reference_waveform=ref_waveform,
                reference_sample_rate=ref_sample_rate,
            )

            output_audio = {"waveform": output_tensor, "sample_rate": sample_rate}

            logger.info(f"Audio generation complete. Sample rate: {sample_rate}Hz")

            if force_offload:
                logger.info(f"Force offloading VoxCPM model '{model_name}' from VRAM...")
                patcher.unpatch_model(unpatch_weights=True)

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except model_management.InterruptProcessingException:
            # Clean interrupt - no logging needed
            raise
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
