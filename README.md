<div id="readme-top" align="center">
<h1 align="center">ComfyUI-VoxCPM</h1>

<a href="https://github.com/wildminder/ComfyUI-VoxCPM">
<img alt="ComfyUI-VoxCPM" src="https://github.com/user-attachments/assets/c3c1f87e-bc53-4a7a-8e69-d7a2f5832e04" />
</a>

<p align="center">
ComfyUI custom node integrating <strong>VoxCPM</strong> — a tokenizer-free TTS system for expressive speech generation and voice cloning.
<br />
  
[![Report Bug][bug-shield]][bug-url] [![Request Feature][feature-shield]][feature-url]

</p>
</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Telegram][telegram-shield]][telegram-url]
[![X][x-shield]][x-url]

</div>

<br>

## ▷ About

VoxCPM models speech in a continuous space using a MiniCPM-4 backbone, producing highly expressive speech and accurate zero-shot voice cloning. This node handles model downloading, memory management, and audio processing end-to-end.

<div align="center">
<img alt="ComfyUI-VoxCPM example workflow" src="https://github.com/user-attachments/assets/bfb28fa2-c143-4542-97f7-936a67941125" />
</div>

### ❖ VoxCPM2
* **Voice Design** — generate voices from natural language descriptions
* **Controllable Voice Cloning** — clone with style control instructions
* **Ultimate Cloning** — combine reference audio (identity) + prompt audio (prosody)
* **48kHz** output, **30+ languages**

### ❖ VoxCPM1.5
* **44.1kHz** output, LoRA support, native LoRA training
* Context-aware expressive speech, zero-shot TTS
* Automatic model management

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Getting Started

**Via ComfyUI Manager:** Search `ComfyUI-VoxCPM` → Install.

**Manual install:**
```sh
cd ComfyUI/custom_nodes/
git clone https://github.com/wildminder/ComfyUI-VoxCPM.git
cd ComfyUI-VoxCPM
pip install -r requirements.txt
```

Restart ComfyUI. Nodes appear under `audio/tts`. Models auto-download to `ComfyUI/models/tts/VoxCPM/` on first use.

<p align="right"><a href="#readme-top">⟔ ▲ ⟓</a></p>

## ▓ Models

| Model | Params | Sample Rate | Languages | Link |
|:---|:---:|:---:|:---:|:---|
| **VoxCPM2** | 2B | 48kHz | 30+ | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |
| **VoxCPM1.5** | 800M | 44.1kHz | 2 | [openbmb/VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) |
| VoxCPM-0.5B | 640M | 16kHz | 2 | [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) |

<p align="right"><a href="#readme-top">⟔ ▲ ⟓</a></p>

## ▓ Nodes

### ❖ `VoxCPM TTS`
Unified TTS node supporting VoxCPM1.5 and VoxCPM2: zero-shot TTS, voice design, voice cloning, ultimate cloning, LoRA support.

### ❖ `VoxCPM Voice Cloning`
Configures audio-based cloning (prompt/reference audio, VAD trimming). Connect to TTS node's `voice_config` input.

### ❖ `VoxCPM Advanced Params`
Configures diffusion parameters (temperature, sway sampling, CFG, timesteps, retry). Connect to TTS node's `advanced_params` input.

### ❖ Training Nodes
* **`VoxCPM Train Config`** — LoRA training parameters
* **`VoxCPM Dataset Maker`** — create training datasets from audio
* **`VoxCPM LoRA Trainer`** — train custom LoRA models

> **Note:** `voice_design` is a direct parameter on the TTS node, not part of the Voice Cloning config.

**Config precedence:** Direct parameters > config node values > defaults.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Usage

### ❖ Basic TTS (Zero-Shot)
Add `VoxCPM TTS` → select model → enter text → generate.

### ❖ Voice Cloning (VoxCPM1.5 style)
Connect `Load Audio` → `prompt_audio`, provide exact transcript in `prompt_text` → generate.

### ❖ Voice Design (VoxCPM2)
Select VoxCPM2 model → enter description in `voice_design` (e.g., "warm female voice") → generate. Voice design is applied in plain TTS and reference cloning modes. Ignored when prompt audio is used (continuation cloning).

### ❖ Reference Cloning (VoxCPM2)
Connect reference audio to `reference_audio` (no transcript needed) → generate. Voice design instructions can be combined with reference audio for controllable cloning (e.g., style control).

### ❖ Ultimate Cloning (VoxCPM2)
Connect `reference_audio` (identity) + `prompt_audio` with transcript (prosody) → generate.

> [!NOTE]
> **Denoising:** The built-in ZipEnhancer denoiser is disabled by default to keep dependencies light.

<p align="right"><a href="#readme-top">⟔ ▲ ⟓</a></p>

## ▓ Advanced Parameters

### ❖ Diffusion

| Parameter | Default | Range | Description |
|:---|:---:|:---:|:---|
| `temperature` | 1.0 | 0.1-2.0 | Lower = stable, higher = expressive |
| `sway_sampling_coef` | 1.0 | 0.0-2.0 | Sway sampling trajectory |
| `use_cfg_zero_star` | True | — | CFG-Zero* optimization |
| `cfg_value` | 2.0 | 0.1-10.0 | Guidance scale |
| `inference_timesteps` | 10 | 1-100 | More steps = higher quality, slower |

### ❖ Device & Precision

| Parameter | Default | Options | Description |
|:---|:---:|:---:|:---|
| `device` | auto | cuda, cpu, mps, xpu, npu | Inference device |
| `dtype` | auto | auto, bf16, fp16, fp32 | Model precision |

> AudioVAE always runs in FP32 for numerical stability.

### ❖ VAD (VoxCPM2)

| Parameter | Default | Range | Description |
|:---|:---:|:---:|:---|
| `trim_silence` | False | — | VAD silence trimming |
| `max_silence_ms` | 200.0 | 0-1000 | Max silence at boundaries (ms) |
| `top_db` | 35.0 | 10-60 | Lower = more aggressive trimming |

<p align="right"><a href="#readme-top">⟔ ▲ ⟓</a></p>

## ▓ LoRA Support

**Inference:** Place `.safetensors` LoRA files in `ComfyUI/models/loras/`, refresh, select in `lora_name` dropdown.

**Training:** 👉 [Full LoRA Training Guide](readme-lora-training.md)

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Voice Design Examples

| Description | Result |
|:---|:---|
| `warm female voice` | Soft, gentle female voice |
| `deep male voice` | Low-pitched male voice |
| `cheerful young girl` | Energetic, high-pitched |
| `professional announcer` | Clear, authoritative |
| `whispering voice` | Quiet, intimate |

Combine descriptions: `"warm female voice with slight British accent"`

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Voice Cloning Tips

1. **Verbatim transcript** — `prompt_text` must match audio word-for-word
2. **Punctuation matters** — affects intonation
3. **5-15 seconds** of clear speech works best

> [!Warning]
> `prompt_text` is the exact transcript, not a description of the voice.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Risks and Limitations

* Voice cloning can be misused for deepfakes — use responsibly
* May exhibit instability with very long/complex inputs
* VoxCPM1.5: Chinese and English only; VoxCPM2: 30+ languages

<p align="right"><a href="#readme-top">⟔ ▲ ⟓</a></p>

## ▓ Changelog

### v2.4.0
- **Custom model selector UI** — replaced default LiteGraph dropdown with custom DOM widget featuring cyber-themed design, SVG model icons, and DEFAULT/CUSTOM badges
- **Real-time download progress** — live progress bar with cancel support, Xet dedup tracking, and file-level progress
- **Lazy-loaded frontend** — 95% startup payload reduction; heavy JS loads only when a VoxCPM node is placed
- **Model directory dialog** — browse and select custom model paths via a dedicated dialog instead of queue-time prompts
- **Architecture version tags** — model dropdown detects and displays architecture version (v1/v2) tags
- **ComfyUI settings API migration** — model path settings now use ComfyUI's native settings panel
- **Graceful download error handling** — automatic retry on transient network errors with toast notifications
- **BEM-styled UI** — modern CSS architecture with design tokens for consistent theming

### v2.3
- VoxCPM2 support (voice design, reference cloning, ultimate cloning, 48kHz, 30+ languages)
- Unified dtype/device handling delegating to ComfyUI
- Vite 5→8 upgrade, LiteGraph canvas widget fixes
- LoRA training nodes

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## License

VoxCPM model and components: [Apache-2.0 License](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE) by OpenBMB.

## Acknowledgments

* **OpenBMB & ModelBest** for [VoxCPM](https://github.com/OpenBMB/VoxCPM)
* **The ComfyUI team** for the platform

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<p align="center">══════════════════════════════════</p>

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-VoxCPM.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-VoxCPM/stargazers
[telegram-shield]: https://img.shields.io/badge/Telegram-TokenDiff-26A5E4?style=for-the-badge&logo=telegram&logoColor=white
[telegram-url]: https://t.me/TokenDiff
[x-shield]: https://img.shields.io/badge/X-@wildmindai-000000?style=for-the-badge&logo=x&logoColor=white
[x-url]: https://x.com/wildmindai
[bug-shield]: https://img.shields.io/badge/Report-Bug-red?style=flat-square&logo=github
[bug-url]: https://github.com/wildminder/ComfyUI-VoxCPM/issues/new?labels=bug&template=bug-report---.md
[feature-shield]: https://img.shields.io/badge/Request-Feature-blue?style=flat-square&logo=github
[feature-url]: https://github.com/wildminder/ComfyUI-VoxCPM/issues/new?labels=enhancement&template=feature-request---.md
