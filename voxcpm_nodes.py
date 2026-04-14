"""
VoxCPM TTS Node for ComfyUI.

This module provides a unified TTS node supporting both VoxCPM1.5 and VoxCPM2 models.
- Basic TTS synthesis
- Voice Design (VoxCPM2): Create voices from natural language descriptions
- Voice Cloning: Clone from reference audio (VoxCPM2) or prompt audio + text
- Ultimate Cloning: Full nuance reproduction with both reference and prompt audio
"""

import torch
import logging
from typing import List, Optional

import folder_paths
import comfy.model_management as model_management
from comfy_api.latest import ComfyExtension, io, ui

from .modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS
from .modules.generation import (
    load_voxcpm_model,
    extract_audio_tensor,
    apply_lora_if_needed,
    generate_audio,
    build_final_text,
    validate_voxcpm2_features,
    validate_prompt_pairing,
)
from .modules.utils import get_available_devices, set_seed
from src.voxcpm.utils.text_normalize import TEXT_NORMALIZATION_AVAILABLE

from .voxcpm_train_nodes import VoxCPM_TrainConfig, VoxCPM_DatasetMaker, VoxCPM_LoraTrainer

logger = logging.getLogger(__name__)


class VoxCPMNode(io.ComfyNode):
    """
    Unified TTS node supporting both VoxCPM1.5 and VoxCPM2 models.

    Features:
    - Basic TTS synthesis (all models)
    - Voice Design: Create voices from descriptions (VoxCPM2)
    - Voice Cloning: Clone from reference audio (VoxCPM2) or prompt audio + text
    - Ultimate Cloning: Combine reference + prompt for maximum fidelity (VoxCPM2)
    - LoRA support for style customization
    """

    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls):
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())
        if not model_names:
            model_names.append("No models found. Please download a VoxCPM model.")

        available_devices = get_available_devices()
        default_device = available_devices[0]

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM_TTS",
            display_name="VoxCPM TTS",
            category=cls.CATEGORY,
            description="Generate speech or clone voices using VoxCPM models (1.5 or 2) with LoRA support. VoxCPM2 adds voice design via control instructions and reference audio cloning.",
            inputs=[
                # Model selection
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM model to use. VoxCPM2 supports voice design and reference cloning."),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="Select a LoRA to apply for style/fine-tuning."),

                # Text input
                io.String.Input("text", multiline=True, default="VoxCPM is an innovative TTS model designed to generate highly expressive speech.", tooltip="Text to synthesize."),

                # Voice design (VoxCPM2)
                io.String.Input("voice_design", multiline=False, optional=True, tooltip="(VoxCPM2 only) Voice design instruction (e.g., 'warm female voice', 'deep male voice')."),

                # Prompt audio + text (VoxCPM1.5 style voice cloning)
                io.String.Input("prompt_text", multiline=True, optional=True, tooltip="Transcript of the prompt audio. Required when using prompt_audio."),
                io.Audio.Input("prompt_audio", optional=True, tooltip="Prompt audio for voice cloning (continuation mode). Requires prompt_text."),

                # VoxCPM2 reference audio cloning
                io.Audio.Input("reference_audio", optional=True, tooltip="(VoxCPM2 only) Reference audio for voice cloning (isolated mode). No transcript needed."),

                # Generation parameters
                io.Float.Input("cfg", default=2.0, min=0.1, max=10.0, step=0.1, tooltip="Guidance scale. Higher values adhere more to the prompt but may sound less natural."),
                io.Int.Input("steps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps. Higher values may improve quality but are slower."),
                io.Int.Input("min_tokens", default=2, min=1, max=100, tooltip="Minimum length of generated audio tokens."),
                io.Int.Input("max_tokens", default=2048, min=64, max=8192, tooltip="Maximum length of generated audio tokens."),
                io.Boolean.Input(
                    "normalize_text",
                    default=TEXT_NORMALIZATION_AVAILABLE,
                    label_on="Normalize",
                    label_off="Raw",
                    tooltip="Enable text normalization (requires 'inflect' and 'wetext' packages)." if not TEXT_NORMALIZATION_AVAILABLE else "Enable text normalization (recommended for general text)."
                ),
                io.Boolean.Input("trim_silence", default=False, label_on="Trim", label_off="Keep", tooltip="(VoxCPM2 only) Trim silence from reference/prompt audio using VAD."),

                # Advanced generation parameters
                io.Float.Input("temperature", default=1.0, min=0.1, max=2.0, step=0.1, tooltip="Sampling temperature. Lower = more stable/consistent, Higher = more varied/expressive."),
                io.Float.Input("sway", default=1.0, min=0.0, max=2.0, step=0.1, tooltip="Sway sampling coefficient. Affects sampling trajectory in diffusion."),
                io.Boolean.Input("use_cfg_zero_star", default=True, label_on="CFG-Zero*", label_off="Standard CFG", tooltip="Use CFG-Zero* optimization for better quality."),

                # Retry parameters
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="Max retry attempts for bad cases (e.g., babbling). Set to 0 to disable."),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="Audio/text length ratio to trigger a retry."),

                # System parameters
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Seed for reproducibility. -1 for random."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto-Manage", tooltip="Force the model to be offloaded from VRAM after generation."),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="Device to run inference on."),
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
        text: str,
        voice_design: Optional[str],
        prompt_text: Optional[str],
        cfg: float,
        steps: int,
        min_tokens: int,
        max_tokens: int,
        normalize_text: bool,
        trim_silence: bool,
        temperature: float,
        sway: float,
        use_cfg_zero_star: bool,
        retry_max_attempts: int,
        retry_threshold: float,
        seed: int,
        force_offload: bool,
        prompt_audio: Optional[io.Audio.Type] = None,
        reference_audio: Optional[io.Audio.Type] = None,
    ) -> io.NodeOutput:

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
            model_name, MODEL_CONFIGS, reference_audio, voice_design
        )
        if warning:
            logger.warning(warning)
            voice_design = None # Ignore for VoxCPM1.5

        # Validate prompt audio/text pairing
        validate_prompt_pairing(prompt_audio, prompt_text)

        # Build final text with voice design instruction (VoxCPM2)
        final_text = build_final_text(text, voice_design)

        # Set seed for reproducibility
        set_seed(seed)

        # Load model
        patcher, model = load_voxcpm_model(model_name, device)

        # Apply LoRA if specified
        apply_lora_if_needed(model, lora_name)

        # Extract audio tensors
        prompt_waveform, prompt_sr = extract_audio_tensor(prompt_audio, "prompt_audio")
        # Only use reference_audio if not ignored (VoxCPM1.5 graceful degradation)
        if ignore_reference:
            ref_waveform, ref_sr = None, None
        else:
            ref_waveform, ref_sr = extract_audio_tensor(reference_audio, "reference_audio")

        try:
            # Generate audio
            output_tensor, sample_rate = generate_audio(
                model=model,
                text=final_text,
                cfg_value=cfg,
                inference_timesteps=steps,
                min_len=min_tokens,
                max_len=max_tokens,
                normalize=normalize_text,
                trim_silence_vad=trim_silence,
                temperature=temperature,
                sway_sampling_coef=sway,
                use_cfg_zero_star=use_cfg_zero_star,
                retry_badcase=retry_max_attempts > 0,
                retry_max_attempts=retry_max_attempts,
                retry_threshold=retry_threshold,
                prompt_text=prompt_text,
                prompt_waveform=prompt_waveform,
                prompt_sample_rate=prompt_sr,
                reference_waveform=ref_waveform,
                reference_sample_rate=ref_sr,
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


class VoxCPMExtension(ComfyExtension):
    """ComfyUI extension providing VoxCPM TTS nodes."""

    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        return [
            # Main unified TTS node
            VoxCPMNode,
            # Training nodes
            VoxCPM_TrainConfig,
            VoxCPM_DatasetMaker,
            VoxCPM_LoraTrainer
        ]


async def comfy_entrypoint() -> VoxCPMExtension:
    """Entry point for ComfyUI extension loading."""
    return VoxCPMExtension()
