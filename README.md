<!-- Improved compatibility of back to top link -->

<div id="readme-top" align="center">
<h1 align="center">ComfyUI-VoxCPM</h1>

<a href="https://github.com/wildminder/ComfyUI-VoxCPM">
<img alt="ComfyUI-VoxCPM" src="https://github.com/user-attachments/assets/c3c1f87e-bc53-4a7a-8e69-d7a2f5832e04" />
</a>

<p align="center">
A custom node for ComfyUI that integrates <strong>VoxCPM</strong>, a novel tokenizer-free TTS system for context-aware speech generation and true-to-life voice cloning.
<br />
<br />
<a href="https://github.com/wildminder/ComfyUI-VoxCPM/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
·
<a href="https://github.com/wildminder/ComfyUI-VoxCPM/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
</p>
</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]

</div>

<br>

## ▷ About The Project

VoxCPM is a novel tokenizer-free Text-to-Speech system that redefines realism in speech synthesis by modeling speech in a continuous space. Built on the MiniCPM-4 backbone, it excels at generating highly expressive speech and performing accurate zero-shot voice cloning.

<div align="center">

<img alt="ComfyUI-VoxCPM example workflow" src="https://github.com/user-attachments/assets/9e5bc5a3-f5bb-4184-b5a9-d57db3acb6aa" />

</div>

This custom node handles everything from model downloading and memory management to audio processing, allowing you to generate high-quality speech directly from a text script and optional reference audio files.


### ❖ VoxCPM2
* **Voice Design:** Generate voices from natural language descriptions (e.g., "warm female voice", "deep male voice")
* **Controllable Voice Cloning:** Clone voices with style control instructions
* **Ultimate Cloning:** Combine reference audio (identity) with prompt audio (prosody) for most accurate reproduction
* **48kHz Audio:** Higher fidelity output compared to v1.5
* **30 Languages:** Multilingual support including English, Chinese, Japanese, Korean, and more
* **Reference Audio Mode:** Clone voices without needing transcripts

### ❖ VoxCPM1.5
* **High-Fidelity Audio:** Supports 44.1kHz sampling rate, preserving high-frequency details
* **LoRA Support:** Load fine-tuned LoRA checkpoints to apply specific voice styles
* **Native LoRA Training:** Train your own voice styles directly within ComfyUI
* **Context-Aware Expressive Speech:** Model understands text context for appropriate prosody
* **True-to-Life Voice Cloning:** Clone timbre, accent, and emotional tone from short samples
* **Zero-Shot TTS:** Generate high-quality speech without any reference audio
* **Automatic Model Management:** Models downloaded automatically and managed efficiently

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Getting Started

The easiest way to install is via **ComfyUI Manager**. Search for `ComfyUI-VoxCPM` and click "Install".

Alternatively, to install manually:

1. **Clone the Repository:**
Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
```sh
git clone https://github.com/wildminder/ComfyUI-VoxCPM.git
```

2. **Install Dependencies:**
Open a terminal or command prompt, navigate into the cloned `ComfyUI-VoxCPM` directory, and install the required Python packages:
```sh
cd ComfyUI-VoxCPM
pip install -r requirements.txt
```

3. **Start/Restart ComfyUI:**
Launch ComfyUI. The VoxCPM nodes will appear under the `audio/tts` category. The first time you use a node, it will automatically download the selected model to your `ComfyUI/models/tts/VoxCPM/` folder.

## ▓ Models

| Model | Parameters | Sampling Rate | Languages | Description | Hugging Face Link |
|:---|:---:|:---:|:---:|:---|:---|
| **VoxCPM2** | 2B | 48kHz | 30+ | **New!** Voice design, reference cloning, multilingual | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |
| **VoxCPM1.5** | 800M | 44.1kHz | 2 | Recommended for v1. LoRA support, improved fidelity | [openbmb/VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) |
| VoxCPM-0.5B | 640M | 16kHz | 2 | Original version | [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) |

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Available Nodes

### TTS Node
**`VoxCPM TTS`** - Single unified node supporting both VoxCPM1.5 and VoxCPM2 with all features:
- Basic TTS synthesis (zero-shot)
- Voice Design from natural language descriptions (VoxCPM2)
- Voice Cloning from reference audio (VoxCPM2) or prompt audio + text
- Ultimate Cloning combining reference + prompt audio (VoxCPM2)
- LoRA support for custom voice styles

### Training Nodes
* **`VoxCPM Train Config`** - Configure LoRA training parameters
* **`VoxCPM Dataset Maker`** - Create training datasets from audio files
* **`VoxCPM LoRA Trainer`** - Train custom LoRA models

### Configuration Nodes (Modular Architecture)
* **`VoxCPM Voice Cloning`** - Configure audio-based voice cloning settings (prompt/reference audio)
* **`VoxCPM Advanced Params`** - Configure advanced generation parameters

> **Note:** Voice design (`voice_design`) is a direct parameter on the main TTS node, not part of the Voice Cloning configuration.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Modular Node Architecture

The VoxCPM nodes support a modular architecture where configuration can be separated into dedicated nodes:

### Benefits
- **Cleaner Workflows**: Separate complex configuration from the main TTS node
- **Reusability**: Share the same configuration across multiple TTS nodes
- **Organization**: Group related parameters logically

### Using Configuration Nodes

#### Voice Cloning Configuration
Connect the `VoxCPM Voice Cloning` node to the main TTS node's `voice_config` input:

```
[VoxCPM Voice Cloning] ──voice_config──> [VoxCPM TTS]
```

Parameters in the Voice Cloning node:
- `prompt_text`: Transcript of prompt audio
- `prompt_audio`: Audio for continuation-style cloning
- `reference_audio`: Reference audio for identity cloning (VoxCPM2)
- `trim_silence`: Enable VAD silence trimming

> **Note:** `voice_design` is now a direct parameter on the main TTS node, not in this configuration node.

#### Advanced Parameters Configuration
Connect the `VoxCPM Advanced Params` node to the main TTS node's `advanced_params` input:

```
[VoxCPM Advanced Params] ──advanced_params──> [VoxCPM TTS]
```

Parameters in the Advanced Params node:
- `min_tokens` / `max_tokens`: Audio length constraints
- `temperature`: Sampling temperature (0.1-2.0)
- `sway_sampling_coef`: Sway sampling coefficient (0.0-2.0)
- `use_cfg_zero_star`: CFG-Zero* optimization toggle
- `retry_max_attempts`: Maximum retry attempts for bad cases
- `retry_threshold`: Threshold for triggering retries

### Configuration Precedence

When both configuration nodes and direct parameters are provided:
1. **Direct parameters** take precedence over config node values
2. **Config node values** take precedence over defaults

This allows you to set defaults via config nodes and override specific values when needed.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Usage

###  Basic TTS (Zero-Shot)
1. Add the `VoxCPM TTS` node to your graph
2. Select a model (VoxCPM2 recommended)
3. Enter text in the `text` field
4. Generate!

### Voice Cloning (VoxCPM1.5 style)
1. Add a `Load Audio` node with your reference audio
2. Connect to `prompt_audio` input
3. Provide the exact transcript in `prompt_text`
4. Generate!

### Voice Design (VoxCPM2 only)
1. Use the `VoxCPM TTS` node with a VoxCPM2 model
2. Enter voice description in `voice_design`: "warm female voice", "deep male voice with slight rasp"
3. Enter text to synthesize
4. Generate!

> **Note:** `voice_design` is only used in plain TTS mode (no reference audio). When reference audio is connected, the voice is cloned from that audio and `voice_design` is ignored.

### Reference Cloning (VoxCPM2 only)
1. Use the `VoxCPM TTS` node with a VoxCPM2 model
2. Connect reference audio to `reference_audio` (no transcript needed!)
3. Generate!

> **Note:** When using reference audio, `voice_design` is ignored - the voice is cloned directly from the reference audio.

### Ultimate Cloning (VoxCPM2 only)
For maximum fidelity, combine reference audio (identity) with prompt audio (prosody):
1. Use the `VoxCPM TTS` node with a VoxCPM2 model
2. Connect reference audio to `reference_audio`
3. Connect prompt audio to `prompt_audio` with transcript in `prompt_text`
4. Generate!

> [!NOTE]
> **Denoising:** The original VoxCPM library includes a built-in denoiser (ZipEnhancer). This feature is disabled by default to keep dependencies light.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Advanced Generation Parameters

All generation nodes expose advanced parameters for fine-tuning output quality:

### Diffusion Parameters

| Parameter | Default | Range | Description |
|:---|:---:|:---:|:---|
| `temperature` | 1.0 | 0.1-2.0 | Sampling temperature. Lower = more stable/consistent, Higher = more varied/expressive |
| `sway_sampling_coef` | 1.0 | 0.0-2.0 | Sway sampling coefficient. Affects sampling trajectory in diffusion |
| `use_cfg_zero_star` | True | - | CFG-Zero* optimization for better quality guidance |
| `cfg_value` | 2.0 | 0.1-10.0 | Guidance scale. Higher = more adherence to voice description/reference |
| `inference_timesteps` | 10 | 1-100 | Number of diffusion steps. More steps = higher quality but slower |

### Device & Precision Parameters

| Parameter | Default | Options | Description |
|:---|:---:|:---:|:---|
| `device` | auto | cuda, cpu, mps, xpu, npu | Device to run inference on. Auto-detects best available |
| `dtype` | auto | auto, bf16, fp16, fp32 | Model precision. 'auto' selects optimal type for device |

> [!TIP]
> - **CUDA GPUs**: Use `auto` (BF16 on Ampere+, FP16 on older GPUs)
> - **Apple Silicon (MPS)**: Use `auto` (FP16 native, BF16 on macOS 14+)
> - **CPU**: Use `auto` (FP32) or `bf16` (PyTorch 2.0+)
> - **FP16 on CPU**: Not recommended - may cause errors

> [!NOTE]
> The AudioVAE component always runs in FP32 for numerical stability, regardless of the selected dtype. This is by design - the model's decode operations expect float32 input.

### VAD Parameters (VoxCPM2 only)

Voice Activity Detection parameters for trimming silence from reference/prompt audio:

| Parameter | Default | Range | Description |
|:---|:---:|:---:|:---|
| `trim_silence` | False | - | Enable VAD-based silence trimming |
| `max_silence_ms` | 200.0 | 0-1000 | Maximum silence to keep at boundaries (ms) |
| `top_db` | 35.0 | 10-60 | Silence detection threshold (dB). Lower = more aggressive trimming |

> [!TIP]
> For noisy reference audio, enable `trim_silence` and reduce `top_db` (e.g., 25-30) for more aggressive silence removal.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

##  LoRA Support (Inference & Training)

VoxCPM fully supports LoRA technology for fine-tuning voice styles. Works with both VoxCPM1.5 and VoxCPM2.

### Inference
To use a pre-trained LoRA:
1. Place your `.safetensors` LoRA files in `ComfyUI/models/loras/`
2. Refresh the node, then select your file in the `lora_name` dropdown

### Training
Train custom LoRA models directly within ComfyUI using the training nodes.

👉 **[Click here for the full LoRA Training Guide](readme-lora-training.md)**

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Voice Design Examples (VoxCPM2)

The Voice Design feature allows you to create voices from natural language descriptions:

| Description | Result |
|:---|:---|
| `warm female voice` | Soft, gentle female voice |
| `deep male voice` | Low-pitched male voice |
| `cheerful young girl` | Energetic, high-pitched voice |
| `professional announcer` | Clear, authoritative voice |
| `whispering voice` | Quiet, intimate speech |

You can combine descriptions: `"warm female voice with slight British accent"`

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

##  Achieving High-Quality Voice Clones

For best voice cloning results with `prompt_audio` + `prompt_text`:

1. **Provide a Verbatim Transcript** - The `prompt_text` must be word-for-word match
2. **Punctuation Matters** - Use accurate punctuation for proper intonation
3. **Audio Length** - 5-15 seconds of continuous, clear speech works best

> [!Warning]
> `prompt_text` is the exact transcript, not a description of the voice.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ⟁ Risks and Limitations

* **Potential for Misuse:** Voice cloning could be misused for deepfakes. Users must not use it to infringe upon rights of individuals or for illegal purposes.
* **Technical Limitations:** May exhibit instability with very long or complex inputs.
* **Language Support:** VoxCPM1.5 is primarily Chinese and English. VoxCPM2 supports 30+ languages.
* This node is released for research and development purposes. Please use it responsibly.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<!-- LICENSE -->
## License

The VoxCPM model and its components are subject to the [Apache-2.0 License](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE) provided by OpenBMB.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* **OpenBMB & ModelBest** for creating and open-sourcing the incredible [VoxCPM](https://github.com/OpenBMB/VoxCPM) project.
* **The ComfyUI team** for their powerful and extensible platform.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>


<p align="center">══════════════════════════════════</p>

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-VoxCPM.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-VoxCPM/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-VoxCPM.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-VoxCPM/issues
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-VoxCPM.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-VoxCPM/network/members
