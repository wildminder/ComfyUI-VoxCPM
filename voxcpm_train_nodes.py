import os
import logging
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui

from .modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS
from .modules.dataset_utils import create_jsonl_dataset

logger = logging.getLogger(__name__)
# The training module imports 'argbind' and 'datasets'.
# We wrap this so the main inference node works without them.
TRAINING_IMPORT_ERROR = None
try:
    from .modules.trainer import run_lora_training
except ImportError as e:
    run_lora_training = None
    TRAINING_IMPORT_ERROR = str(e)
    # Check specifically for the likely missing packages to give a better error
    missing = []
    try: import argbind
    except ImportError: missing.append("argbind")
    try: import datasets
    except ImportError: missing.append("datasets")

    if missing:
        TRAINING_IMPORT_ERROR = f"Missing required packages for training: {', '.join(missing)}. Please run: pip install {' '.join(missing)}"

class VoxCPM_TrainConfig(io.ComfyNode):
    CATEGORY = "audio/tts/training"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxCPM_TrainConfig",
            display_name="VoxCPM Train Config",
            category=cls.CATEGORY,
            description="Configuration parameters for VoxCPM LoRA training. Works with both VoxCPM1.5 and VoxCPM2.",
            inputs=[
                io.Float.Input("learning_rate", default=1e-4, min=1e-6, max=1e-2, step=1e-5, tooltip="Learning rate for the optimizer."),
                io.Int.Input("lora_rank", default=32, min=4, max=128, step=4, tooltip="Rank (dimension) of the LoRA adapter."),
                io.Int.Input("lora_alpha", default=16, min=1, max=128, step=1, tooltip="Alpha scaling factor for LoRA."),
                io.Float.Input("lora_dropout", default=0.0, min=0.0, max=0.5, step=0.05, tooltip="Dropout probability for LoRA layers."),
                io.Int.Input("warmup_steps", default=100, min=0, max=1000, tooltip="Number of warmup steps for learning rate scheduler."),
                io.Int.Input("grad_accum_steps", default=1, min=1, max=64, tooltip="Number of steps to accumulate gradients before updating weights."),
                io.Int.Input("max_batch_tokens", default=8192, min=1024, max=32768, tooltip="Maximum number of tokens per batch to manage VRAM usage."),
                io.Int.Input("sample_rate", default=48000, min=16000, max=48000, tooltip="Sample rate of the training audio. Use 48000 for VoxCPM2, 44100 for VoxCPM1.5."),
                io.Float.Input("weight_decay", default=0.01, min=0.0, max=0.1, tooltip="Weight decay for regularization."),
                io.Boolean.Input("enable_lm_lora", default=True, tooltip="Apply LoRA to the Language Model backbone."),
                io.Boolean.Input("enable_dit_lora", default=True, tooltip="Apply LoRA to the Diffusion Transformer."),
                io.Boolean.Input("enable_proj_lora", default=False, tooltip="Apply LoRA to projection layers."),
            ],
            outputs=[
                io.AnyType.Output(display_name="Train Config"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(kwargs)


class VoxCPM_DatasetMaker(io.ComfyNode):
    CATEGORY = "audio/tts/training"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxCPM_DatasetMaker",
            display_name="VoxCPM Dataset Maker",
            category=cls.CATEGORY,
            description="Creates a JSONL dataset from a folder of audio files and text transcripts.",
            inputs=[
                io.String.Input("audio_directory", default="", tooltip="Path to directory containing .wav and .txt files."),
                io.String.Input("output_filename", default="train.jsonl", tooltip="Name of the output JSONL file."),
            ],
            outputs=[
                io.String.Output(display_name="Dataset Path"),
            ],
        )

    @classmethod
    def execute(cls, audio_directory, output_filename):
        try:
            dataset_path = create_jsonl_dataset(audio_directory, output_filename)
            return io.NodeOutput(dataset_path)
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            raise e


class VoxCPM_LoraTrainer(io.ComfyNode):
    CATEGORY = "audio/tts/training"

    @classmethod
    def define_schema(cls):
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())
        if not model_names:
            model_names.append("No models found.")

        return io.Schema(
            node_id="VoxCPM_LoraTrainer",
            display_name="VoxCPM LoRA Trainer",
            category=cls.CATEGORY,
            description="Trains a LoRA adapter for VoxCPM (1.5 or 2). WARNING: This process takes time and blocks the UI.",
            inputs=[
                io.Combo.Input("base_model_name", options=model_names, default=model_names[0], tooltip="Base VoxCPM model to fine-tune. Supports both VoxCPM1.5 and VoxCPM2."),
                io.AnyType.Input("train_config", tooltip="Configuration dictionary from VoxCPM Train Config node."),
                io.String.Input("dataset_path", default="", tooltip="Path to the train.jsonl file."),
                io.String.Input("output_name", default="my_lora_v1", tooltip="Name of the subfolder in 'models/loras' to save results."),
                io.Int.Input("max_steps", default=1000, min=100, max=100000, tooltip="Total number of training steps."),
                io.Int.Input("save_every_steps", default=200, min=50, max=5000, tooltip="Save checkpoint every N steps."),
                io.Int.Input("num_workers", default=0, min=0, max=8, tooltip="Number of dataloader workers (0 for main thread)."),
            ],
            outputs=[
                io.String.Output(display_name="LoRA Output Path"),
            ],
        )

    @classmethod
    def execute(cls, base_model_name, train_config, dataset_path, output_name, max_steps, save_every_steps, num_workers):
        # Guard: Check if training module loaded successfully
        if run_lora_training is None:
            raise RuntimeError(f"Training functionality unavailable. {TRAINING_IMPORT_ERROR}")

        # Determine output directory using ComfyUI's standard paths
        lora_base_dir = folder_paths.get_folder_paths("loras")[0]
        output_dir = os.path.join(lora_base_dir, output_name)

        try:
            # Delegate to trainer module
            final_output_dir = run_lora_training(
                base_model_name=base_model_name,
                train_config=train_config,
                dataset_path=dataset_path,
                output_dir=output_dir,
                max_steps=max_steps,
                save_every_steps=save_every_steps,
                num_workers=num_workers,
                output_name=output_name,
                folder_paths_module=folder_paths # Pass folder_paths to resolve official models
            )
            return io.NodeOutput(final_output_dir)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e