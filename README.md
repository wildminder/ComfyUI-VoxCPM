<!-- Improved compatibility of back to top link -->
<a id="readme-top"></a>

<div align="center">
  <h1 align="center">ComfyUI-VoxCPM</h1>

  <a href="https://github.com/wildminder/ComfyUI-VoxCPM">    
    <img src="https://github.com/user-attachments/assets/077f76c1-edd7-4bff-94c6-615037222913" alt="ComfyUI-VoxCPM" width="70%">
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

## About The Project

VoxCPM is a novel tokenizer-free Text-to-Speech system that redefines realism in speech synthesis by modeling speech in a continuous space. Built on the MiniCPM-4 backbone, it excels at generating highly expressive speech and performing accurate zero-shot voice cloning.

This custom node handles everything from model downloading and memory management to audio processing, allowing you to generate high-quality speech directly from a text script and optional reference audio files.

**✨ Key Features:**
*   **Context-Aware Expressive Speech:** The model understands text context to generate appropriate prosody and vocal expression.
*   **True-to-Life Voice Cloning:** Clone a voice's timbre, accent, and emotional tone from a short audio sample.
*   **Zero-Shot TTS:** Generate high-quality speech without any reference audio.
*   **Automatic Model Management:** The required VoxCPM model is downloaded automatically and managed efficiently by ComfyUI to save VRAM.
*   **Fine-Grained Control:** Adjust parameters like CFG scale and inference steps to tune the performance and style of the generated speech.
*   **High-Efficiency Synthesis:** VoxCPM is designed for speed, enabling fast generation even on consumer-grade hardware.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🚀 Getting Started

The easiest way to install is via **ComfyUI Manager**. Search for `ComfyUI-VoxCPM` and click "Install".

Alternatively, to install manually:

1.  **Clone the Repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```sh
    git clone https://github.com/wildminder/ComfyUI-VoxCPM.git
    ```

2.  **Install Dependencies:**
    Open a terminal or command prompt, navigate into the cloned `ComfyUI-VoxCPM` directory, and install the required Python packages:
    ```sh
    cd ComfyUI-VoxCPM
    pip install -r requirements.txt
    ```

3.  **Start/Restart ComfyUI:**
    Launch ComfyUI. The "VoxCPM TTS" node will appear under the `audio/tts` category. The first time you use the node, it will automatically download the selected model to your `ComfyUI/models/tts/VoxCPM/` folder.

## Models
This node automatically downloads the required model files.

| Model | Parameters | Hugging Face Link |
|:---|:---:|:---|
| VoxCPM-0.5B | 0.5B | [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🛠️ Usage

1.  **Add Nodes:** Add the `VoxCPM TTS` node to your graph. For voice cloning, add a `Load Audio` node to load your reference voice file.
2.  **Connect Voice (for Cloning):** Connect the `AUDIO` output from the `Load Audio` node to the `prompt_audio` input on the VoxCPM TTS node.
3.  **Write Text:**
    *   For **voice cloning**, provide the transcript of your reference audio in the `prompt_text` field.
    *   Enter the text you want to generate in the main `text` field.
4.  **Generate:** Queue the prompt. The node will process the text and generate a single audio file.

> [!NOTE]
> **Denoising:** The original VoxCPM library includes a built-in denoiser (ZipEnhancer). This feature has been intentionally removed from the node. The ComfyUI philosophy encourages modular, single-purpose nodes. For denoising, please use a dedicated audio processing node before passing the `prompt_audio` to this one. This keeps the workflow clean and flexible.

### Node Inputs

*   **`model_name`**: Select the VoxCPM model to use. Official models are downloaded automatically.
*   **`text`**: The target text to synthesize into speech.
*   **`prompt_audio` (Optional)**: A reference audio clip for voice cloning.
*   **`prompt_text` (Optional)**: The exact transcript of the `prompt_audio`. This is **required** for voice cloning.
*   **`cfg_value`**: Classifier-Free Guidance scale. Higher values increase adherence to the voice prompt but may reduce naturalness.
*   **`inference_timesteps`**: Number of diffusion steps for audio generation. More steps can improve quality but take longer.
*   **`normalize_text`**: Enable to automatically process numbers, abbreviations, and punctuation. Disable for precise control with phoneme inputs like `{ni3}{hao3}`.
*   **`seed`**: A seed for reproducibility. Set to -1 for a random seed on each run.
*   **`force_offload`**: Forces the model to be completely offloaded from VRAM after generation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 👩‍🍳 A Voice Chef's Guide

---

### 🥚 Step 1: Prepare Your Base Ingredients (Content)

First, choose how you’d like to input your text:
1.  **Regular Text (Classic Mode)**
    *   ✅ Keep **`normalize_text` ON**. Type naturally (e.g., "Hello, world! 123"). The system will automatically process numbers and punctuation.
2.  **Phoneme Input (Native Mode)**
    *   ❌ Turn **`normalize_text` OFF**. Enter phoneme text like `{HH AH0 L OW1}` (EN) or `{ni3}{hao3}` (ZH) for precise pronunciation control.

---
### 🍳 Step 2: Choose Your Flavor Profile (Voice Style)

This is the secret sauce that gives your audio its unique sound.
1.  **With a Prompt (Voice Cloning)**
    *   A `prompt_audio` file provides the desired acoustic characteristics. The speaker's timbre, speaking style, and even ambiance can be replicated.
    *   For best results, use a clean, high-quality audio recording as the prompt.
2.  **Without a Prompt (Zero-Shot Generation)**
    *   If no prompt is provided, VoxCPM becomes a creative chef! It will infer a fitting speaking style based on the text itself, thanks to its foundation model, MiniCPM-4.

---
### 🧂 Step 3: The Final Seasoning (Fine-Tuning)

For master chefs who want to tweak the flavor, here are two key spices:
*   **`cfg_value` (How Closely to Follow the Recipe)**
    *   **Default (2.0):** A great starting point.
    *   **Lower it:** If the cloned voice sounds strained or weird, lowering this value tells the model to be more relaxed and improvisational.
    *   **Raise it slightly:** To maximize clarity and adherence to the prompt voice or text.
*   **`inference_timesteps` (Simmering Time: Quality vs. Speed)**
    *   **Lower (e.g., 5-10):** For a quick snack. Perfect for fast drafts and experiments.
    *   **Higher (e.g., 15-25):** For a gourmet meal. This lets the model "simmer" longer, refining the audio for superior detail and naturalness.

---
Happy creating! 🎉 Start with the default settings and tweak from there. The kitchen is yours!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 📊 Performance Benchmarks
<details>
<summary>Click to view model performance highlights</summary>

VoxCPM achieves competitive results on public zero-shot TTS benchmarks:

### Seed-TTS-eval Benchmark

| Model | Parameters | Open-Source | test-EN | | test-ZH | | test-Hard | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | | WER/%⬇ | SIM/%⬆| CER/%⬇| SIM/%⬆ | CER/%⬇ | SIM/%⬆ |
| **VoxCPM** | 0.5B | ✅ | **1.85** | **72.9** | **0.93** | **77.2** | 8.87 | 73.0 |
| MegaTTS3 | 0.5B | ❌ | 2.79 | 77.1 | 1.52 | 79.0 | - | - |
| DiTAR | 0.6B | ❌ | 1.69 | 73.5 | 1.02 | 75.3 | - | - |

###  CV3-eval Benchmark

| Model | zh | en | hard-zh | | | hard-en | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | CER/%⬇ | WER/%⬇ | CER/%⬇ | SIM/%⬆ | DNSMOS⬆ | WER/%⬇ | SIM/%⬆ | DNSMOS⬆ |
| **VoxCPM** | **3.40** | **4.04** | 12.9 | 66.1 | 3.59 | **7.89** | 64.3 | 3.74 |
| CosyVoice2 | 4.08 | 6.32 | 12.58 | 72.6 | 3.81 | 11.96 | 66.7 | 3.95 |
| IndexTTS2 | 3.58 | 4.45 | 12.8 | 74.6 | 3.65 | - | - | - |

</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ⚠️ Risks and Limitations
*   **Potential for Misuse:** The voice cloning capability is powerful and could be misused for creating convincing deepfakes. Users of this node must not use it to create content that infringes upon the rights of individuals. It is strictly forbidden to use this for any illegal or unethical purposes.
*   **Technical Limitations:** The model may occasionally exhibit instability with very long or complex inputs.
*   **Bilingual Model:** VoxCPM is trained primarily on Chinese and English data. Performance on other languages is not guaranteed.
*   This node is released for research and development purposes. Please use it responsibly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

The VoxCPM model and its components are subject to the [Apache-2.0 License](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE) provided by OpenBMB.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

*   **OpenBMB & ModelBest** for creating and open-sourcing the incredible [VoxCPM](https://github.com/OpenBMB/VoxCPM) project.
*   **The ComfyUI team** for their powerful and extensible platform.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wildminder/ComfyUI-VoxCPM&type=Timeline)](https://www.star-history.com/#wildminder/ComfyUI-VoxCPM&Timeline)


<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-VoxCPM.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-VoxCPM/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-VoxCPM.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-VoxCPM/issues
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-VoxCPM.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-VoxCPM/network/members
