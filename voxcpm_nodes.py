"""
VoxCPM TTS Nodes for ComfyUI.

This module provides the main entry point for VoxCPM nodes.
Nodes are organized in separate files in the 'nodes' package for easier maintenance.

Main Nodes:
- VoxCPMNode: Main TTS synthesis node
- VoxCPMVoiceCloning: Voice cloning configuration node
- VoxCPMAdvancedParams: Advanced parameters configuration node

Training Nodes:
- VoxCPM_TrainConfig: LoRA training configuration
- VoxCPM_DatasetMaker: Dataset preparation
- VoxCPM_LoraTrainer: LoRA training execution
"""

import logging
from typing import List

from comfy_api.latest import ComfyExtension, io

# Import main TTS nodes from the nodes package (relative import)
from .nodes import VoxCPMNode, VoxCPMVoiceCloning, VoxCPMAdvancedParams

# Import training nodes (relative import)
from .voxcpm_train_nodes import VoxCPM_TrainConfig, VoxCPM_DatasetMaker, VoxCPM_LoraTrainer

logger = logging.getLogger(__name__)


class VoxCPMExtension(ComfyExtension):
    """ComfyUI extension providing VoxCPM TTS nodes."""

    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        return [
            # Main TTS node
            VoxCPMNode,
            # Configuration nodes
            VoxCPMVoiceCloning,
            VoxCPMAdvancedParams,
            # Training nodes
            VoxCPM_TrainConfig,
            VoxCPM_DatasetMaker,
            VoxCPM_LoraTrainer
        ]


async def comfy_entrypoint() -> VoxCPMExtension:
    """Entry point for ComfyUI extension loading."""
    return VoxCPMExtension()
