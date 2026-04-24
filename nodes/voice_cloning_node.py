"""
VoxCPM Voice Cloning Configuration Node.

This node provides a dedicated interface for audio-based voice cloning:
- Prompt Audio Cloning: Clone from prompt audio + transcript (all models)
- Reference Audio Cloning: Clone from reference audio (VoxCPM2 only)

Note: Voice design (text-based) is now a direct parameter on the main TTS node.

Output connects to the main VoxCPM TTS node's voice_config input.
"""

import logging
from typing import Optional

from comfy_api.latest import io

from ..modules.custom_types import (
    VoiceCloningConfig,
    create_default_voice_config,
    validate_voice_config,
)
from ..modules.generation import extract_audio_tensor

logger = logging.getLogger(__name__)


class VoxCPMVoiceCloning(io.ComfyNode):
    """
    Configure voice cloning parameters for VoxCPM TTS.

    This node provides a dedicated interface for audio-based voice cloning:
    - Prompt Audio Cloning: Clone from prompt audio + transcript (all models)
    - Reference Audio Cloning: Clone from reference audio (VoxCPM2 only)

    Note: Voice design (text-based) is now a direct parameter on the main TTS node.

    Output connects to the main VoxCPM TTS node's voice_config input.
    """

    CATEGORY = "audio/tts/config"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxCPM_VoiceCloning",
            display_name="VoxCPM Voice Cloning",
            category=cls.CATEGORY,
            description="Configure voice cloning with prompt audio or reference audio. Connect output to VoxCPM TTS node. Note: voice_design is now a direct parameter on the main TTS node.",
            inputs=[
            # Prompt audio + text (VoxCPM1.5 style voice cloning)
            io.String.Input(
                "prompt_text",
                multiline=True,
                optional=True,
                tooltip="Transcript of the prompt audio. Required when using prompt_audio."
            ),
            io.Audio.Input(
                "prompt_audio",
                optional=True,
                tooltip="Prompt audio for voice cloning (continuation mode). Requires prompt_text."
            ),

            # VoxCPM2 reference audio cloning
            io.Audio.Input(
                "reference_audio",
                optional=True,
                tooltip="(VoxCPM2) Reference audio for voice cloning (isolated mode). No transcript needed."
            ),

            # Processing options
            io.Boolean.Input(
                "trim_silence",
                default=False,
                label_on="Trim",
                label_off="Keep",
                tooltip="(VoxCPM2) Trim silence from reference/prompt audio using VAD."
            ),
        ],
        outputs=[
            VoiceCloningConfig.Output("VOICE_CONFIG"),
        ],
    )

    @classmethod
    def execute(
        cls,
        prompt_text: Optional[str] = None,
        prompt_audio: Optional[io.Audio.Type] = None,
        reference_audio: Optional[io.Audio.Type] = None,
        trim_silence: bool = False
    ) -> io.NodeOutput:
        """
        Create voice cloning configuration.

        Args:
            prompt_text: Transcript for prompt audio
            prompt_audio: Prompt audio input
            reference_audio: Reference audio input (VoxCPM2)
            trim_silence: Whether to trim silence from audio

        Returns:
            NodeOutput containing VoiceCloningConfig dict
        """
        # Create default config
        config = create_default_voice_config()

        # Set prompt text
        config["prompt_text"] = prompt_text.strip() if prompt_text else None

        # Set trim_silence
        config["trim_silence"] = trim_silence

        # Extract prompt audio tensor
        if prompt_audio is not None:
            waveform, sr = extract_audio_tensor(prompt_audio, "prompt_audio")
            config["prompt_waveform"] = waveform
            config["prompt_sample_rate"] = sr

        # Extract reference audio tensor
        if reference_audio is not None:
            waveform, sr = extract_audio_tensor(reference_audio, "reference_audio")
            config["reference_waveform"] = waveform
            config["reference_sample_rate"] = sr

        # Validate configuration
        is_valid, error_msg = validate_voice_config(config)
        if not is_valid:
            raise ValueError(error_msg)

        logger.debug(f"VoiceCloningConfig created: "
            f"prompt={config['prompt_waveform'] is not None}, "
            f"reference={config['reference_waveform'] is not None}")

        return io.NodeOutput(config)
