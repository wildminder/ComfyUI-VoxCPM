import torch
import os
import logging
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel

logger = logging.getLogger(__name__)

class VoxCPM:
    def __init__(self, voxcpm_model_path: str):
        """
        Initialize VoxCPM TTS pipeline. Denoiser is disabled for ComfyUI integration.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets.
        """
        logger.info(f"Initializing VoxCPM from path: {voxcpm_model_path}")
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path)
        self.text_normalizer = None
        self.denoiser = None # Denoiser is explicitly disabled.
        
        logger.info("Warming up VoxCPMModel...")
        self.tts_model.generate(
            target_text="Hello, this is a test sentence.",
            max_len=10,
        )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM-0.5B",
            cache_dir: str = None,
            local_files_only: bool = False,
        ):
        """
        Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Hugging Face repository id or local path.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files.

        Returns:
            VoxCPM: Initialized instance.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")
        
        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(voxcpm_model_path=local_path)

    def generate(
        self, 
        text : str,
        prompt_wav_path : str = None,
        prompt_waveform: torch.Tensor = None,
        prompt_sample_rate: int = None,
        prompt_text : str = None,
        cfg_value : float = 2.0,    
        inference_timesteps : int = 10,
        max_length : int = 4096,
        normalize : bool = True,
        retry_badcase : bool = True,
        retry_badcase_max_times : int = 3,
        retry_badcase_ratio_threshold : float = 6.0,
    ):
        """Synthesize speech for the given text and return a single waveform.

        This method optionally builds and reuses a prompt cache. If an external
        prompt (``prompt_wav_path`` + ``prompt_text``) is provided, it will be
        used for all sub-sentences. Otherwise, the prompt cache is built from
        the first generated result and reused for the remaining text chunks.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_length: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt audio if a denoiser is
                available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
        Returns:
            numpy.ndarray: 1D waveform array (float32) on CPU.
        """
        texts = text.split("\n")
        texts = [t.strip() for t in texts if t.strip()]
        final_wav = []
        
        # Check for either waveform or path for cloning
        is_cloning = prompt_waveform is not None or prompt_wav_path is not None
        if is_cloning and prompt_text:
            fixed_prompt_cache = self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
                prompt_waveform=prompt_waveform,
                prompt_sample_rate=prompt_sample_rate
            )
        else:
            # will be built from the first inference
            fixed_prompt_cache = None
        
        for i, sub_text in enumerate(texts):
            if sub_text.strip() == "":
                continue
            logger.info(f"Synthesizing chunk {i+1}/{len(texts)}: '{sub_text[:80]}...'")
            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer
                    self.text_normalizer = TextNormalizer()
                sub_text = self.text_normalizer.normalize(sub_text)
            wav, target_text_token, generated_audio_feat = self.tts_model.generate_with_prompt_cache(
                            target_text=sub_text,
                            prompt_cache=fixed_prompt_cache,
                            min_len=2,
                            max_len=max_length,
                            inference_timesteps=inference_timesteps,
                            cfg_value=cfg_value,
                            retry_badcase=retry_badcase,
                            retry_badcase_max_times=retry_badcase_max_times,
                            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                        )
            if fixed_prompt_cache is None:
                fixed_prompt_cache = self.tts_model.merge_prompt_cache(
                    original_cache=None,
                    new_text_token=target_text_token,
                    new_audio_feat=generated_audio_feat
                )
            final_wav.append(wav)
    
        return torch.cat(final_wav, dim=1).squeeze(0).cpu().numpy()
