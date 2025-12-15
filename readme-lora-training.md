# VoxCPM LoRA Training Guide

This guide details how to fine-tune VoxCPM v1.5 models using LoRA (Low-Rank Adaptation) directly within ComfyUI. This process allows you to clone voices or adapt the model's style using a small dataset of audio samples.

---

##  Prerequisites

1.  **Hardware**: NVIDIA GPU with at least 8GB VRAM (24GB recommended for higher batch sizes).
2.  **Dataset**: A collection of high-quality `.wav` audio files and corresponding text transcripts.
3.  **Base Model**: Ensure `VoxCPM1.5` is downloaded in `models/tts/VoxCPM`.

---

##  Data Preparation

Your training data must consist of pairs of audio files and text transcripts.

### 1. Folder Structure
Organize your data into a single directory. The audio and text files must share the same filename (excluding extension).

```
my_dataset/
├── voice_001.wav
├── voice_001.txt
├── voice_002.wav
├── voice_002.txt
└── ...
```

### 2. Audio Requirements (`.wav`)
*   **Format**: WAV (PCM)
*   **Sample Rate**: 44.1kHz is optimal for VoxCPM 1.5 (the node will resample automatically if needed, but native is better).
*   **Length**: Short clips between 3 to 10 seconds work best. Avoid clips longer than 15 seconds to prevent VRAM issues.
*   **Quality**: Clean, background-noise-free speech is critical.

### 3. Transcript Requirements (`.txt`)
*   **Content**: The exact spoken text corresponding to the audio file.
*   **Language**: Supports mixed English and Chinese.
*   **Normalization**: Raw text is accepted. The training pipeline handles basic tokenization.

---

##  Training Workflow

The training process involves three specific nodes connected in sequence.

### Step 1: Create Dataset Manifest (`VoxCPM Dataset Maker`)
This node scans your folder and generates a `train.jsonl` file required by the training engine.

*   **Inputs**:
    *   `audio_directory`: Absolute path to your dataset folder (e.g., `C:\AI\data\my_voice`).
    *   `output_filename`: Defaults to `train.jsonl`.
*   **Output**: Path string to the generated JSONL file.

### Step 2: Configure Training Parameters (`VoxCPM Train Config`)
This node aggregates all hyperparameters.

#### Key Parameters:
*   **`learning_rate`** (Default: `1e-4`):
    *   Controls how fast the model learns.
    *   *Recommendation*: Start with `1e-4`. If the loss explodes (NaN), reduce to `5e-5`.
*   **`lora_rank`** (Default: `32`):
    *   The dimension of the low-rank matrices. Higher values capture more detail but require more VRAM and data.
    *   *Recommendation*: `32` or `64`.
*   **`lora_alpha`** (Default: `16`):
    *   Scaling factor. A common rule of thumb is `alpha = rank / 2`.
*   **`grad_accum_steps`** (Default: `1`):
    *   Simulates a larger batch size. Since the physical batch size is locked to 1 for stability, increase this to 4 or 8 to stabilize gradients.
*   **`warmup_steps`**: Steps to ramp up the learning rate. Usually 5-10% of total steps.
*   **`max_batch_tokens`**: Limits the amount of audio processed at once. Lower this if you encounter Out-Of-Memory (OOM) errors.

### Step 3: Run Training (`VoxCPM LoRA Trainer`)
This is the execution node. **Warning**: Running this node will block the ComfyUI interface until training completes.

*   **Inputs**:
    *   `base_model_name`: Select `VoxCPM1.5`.
    *   `train_config`: Connect from the Config node.
    *   `dataset_path`: Connect from the Dataset Maker node.
    *   `output_name`: The name of the subfolder in `models/loras` where checkpoints will be saved.
    *   `max_steps`: Total training duration.
        *   *Rule of Thumb*: For a dataset of ~5 minutes, try 1000-2000 steps.
    *   `save_every_steps`: Checkpoint interval.

---

##  Monitoring & Results

### Console Output
Open the ComfyUI console window to see real-time logs:
```
Step 10/1000, Loss: 2.145, LR: 0.00001000
Step 20/1000, Loss: 1.892, LR: 0.00002000
```
*   **Loss**: Should generally decrease. If it stays at 0.0000, something is wrong with the setup.
*   **Loss Spike**: Sudden increases are normal but should recover.

### Output Files
After training, check `ComfyUI/models/loras/[output_name]/`:
1.  **`*.safetensors`**: The LoRA weight files.
2.  **`lora_config.json`**: Configuration metadata required for loading.

---

##  Using Your LoRA

1.  Refresh your ComfyUI browser page.
2.  In the standard **VoxCPM TTS** node:
    *   Set **`model_name`** to `VoxCPM1.5`.
    *   In the **`lora_name`** dropdown, select your newly trained LoRA (e.g., `my_voice_step_2000.safetensors`).
3.  Generate audio!

> **Tip**: If the effect is too strong or distorted, training might have overfitted. Try an earlier checkpoint or reduce the `learning_rate` and retrain.
