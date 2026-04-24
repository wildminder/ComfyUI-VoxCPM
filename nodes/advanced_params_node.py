"""
VoxCPM Advanced Parameters Configuration Node.

This node provides fine-grained control over:
- Token limits (min/max length)
- Sampling parameters (temperature, sway)
- CFG optimization (CFG-Zero*)
- Retry logic for bad cases

Output connects to the main VoxCPM TTS node's advanced_params input.
"""

import logging

from comfy_api.latest import io

from ..modules.custom_types import (
    AdvancedParams,
    validate_advanced_params,
)

logger = logging.getLogger(__name__)


class VoxCPMAdvancedParams(io.ComfyNode):
    """
    Configure advanced generation parameters for VoxCPM TTS.

    This node provides fine-grained control over:
    - Token limits (min/max length)
    - Sampling parameters (temperature, sway)
    - CFG optimization (CFG-Zero*)
    - Retry logic for bad cases

    Output connects to the main VoxCPM TTS node's advanced_params input.
    """

    CATEGORY = "audio/tts/config"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxCPM_AdvancedParams",
            display_name="VoxCPM Advanced Parameters",
            category=cls.CATEGORY,
            description="Advanced generation parameters for fine-grained control over TTS output. Connect output to VoxCPM TTS node.",
            inputs=[
                # Token limits
                io.Int.Input(
                    "min_tokens",
                    default=2,
                    min=1,
                    max=100,
                    tooltip="Minimum length of generated audio tokens."
                ),
                io.Int.Input(
                    "max_tokens",
                    default=2048,
                    min=64,
                    max=8192,
                    tooltip="Maximum length of generated audio tokens."
                ),

                # Sampling parameters
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    tooltip="Sampling temperature. Lower = more stable/consistent, Higher = more varied/expressive."
                ),
                io.Float.Input(
                    "sway",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="Sway sampling coefficient. Affects sampling trajectory in diffusion."
                ),

                # Optimization
                io.Boolean.Input(
                    "use_cfg_zero_star",
                    default=True,
                    label_on="CFG-Zero*",
                    label_off="Standard CFG",
                    tooltip="Use CFG-Zero* optimization for better quality."
                ),

                # Retry logic
                io.Int.Input(
                    "retry_max_attempts",
                    default=3,
                    min=0,
                    max=10,
                    step=1,
                    tooltip="Max retry attempts for bad cases (e.g., babbling). Set to 0 to disable."
                ),
                io.Float.Input(
                    "retry_threshold",
                    default=6.0,
                    min=2.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Audio/text length ratio to trigger a retry."
                ),
            ],
            outputs=[
                AdvancedParams.Output("ADVANCED_PARAMS"),
            ],
        )

    @classmethod
    def execute(
        cls,
        min_tokens: int = 2,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        sway: float = 1.0,
        use_cfg_zero_star: bool = True,
        retry_max_attempts: int = 3,
        retry_threshold: float = 6.0
    ) -> io.NodeOutput:
        """
        Create advanced parameters configuration.

        Args:
            min_tokens: Minimum audio token length
            max_tokens: Maximum audio token length
            temperature: Sampling temperature
            sway: Sway sampling coefficient
            use_cfg_zero_star: Use CFG-Zero* optimization
            retry_max_attempts: Max retry attempts
            retry_threshold: Retry trigger threshold

        Returns:
            NodeOutput containing AdvancedParams dict
        """
        # Create params dict
        params = {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "sway_sampling_coef": sway,
            "use_cfg_zero_star": use_cfg_zero_star,
            "retry_max_attempts": retry_max_attempts,
            "retry_threshold": retry_threshold
        }

        # Validate parameters
        is_valid, error_msg = validate_advanced_params(params)
        if not is_valid:
            raise ValueError(error_msg)

        logger.debug(f"AdvancedParams created: min={min_tokens}, max={max_tokens}, "
                     f"temp={temperature}, sway={sway}, cfg_zero={use_cfg_zero_star}")

        return io.NodeOutput(params)
