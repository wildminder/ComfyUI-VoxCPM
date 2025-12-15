import os
import re
import tempfile
import torch
import numpy as np
from typing import Generator, Optional
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel, LoRAConfig

class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = None, # Defaulted to None for ComfyUI
            enable_denoiser : bool = False,      # Defaulted to False for ComfyUI
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
        ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets.
            zipenhancer_model_path: ModelScope acoustic noise suppression model path.
            enable_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile.
            lora_config: LoRA configuration for fine-tuning.
            lora_weights_path: Path to pre-trained LoRA weights.
        """
        # If lora_weights_path is provided but no lora_config, create a default one
        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )
            # print(f"[ComfyUI-VoxCPM] Auto-created default LoRAConfig for loading weights from: {lora_weights_path}")
        
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path, optimize=optimize, lora_config=lora_config)
        
        # Load LoRA weights if path is provided
        # todo: add logger
        if lora_weights_path is not None:
            # print(f"[ComfyUI-VoxCPM] Loading LoRA weights from: {lora_weights_path}")
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            # print(f"[ComfyUI-VoxCPM] Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}")
        
        self.text_normalizer = None
        
        # Denoiser handling: largely disabled for ComfyUI to keep dependencies light
        # unless specifically requested and available.
        self.denoiser = None
        if enable_denoiser and zipenhancer_model_path is not None:
            try:
                from .zipenhancer import ZipEnhancer
                self.denoiser = ZipEnhancer(zipenhancer_model_path)
            except ImportError:
                # print("[ComfyUI-VoxCPM] Warning: ZipEnhancer dependencies not found. Denoiser disabled.")
                self.denoiser = None

        if optimize:
            # We skip the warmup generation here for ComfyUI to avoid slowing down node loading.
            # The first generation will be slightly slower.
            pass

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM1.5",
            load_denoiser: bool = False,
            zipenhancer_model_id: str = None,
            cache_dir: str = None,
            local_files_only: bool = False,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
            **kwargs,
        ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot."""
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

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_waveform: torch.Tensor = None, # Added for ComfyUI
            prompt_sample_rate: int = None,       # Added for ComfyUI
            prompt_text : str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            min_len : int = 2,
            max_len : int = 4096,
            normalize : bool = False,
            denoise : bool = False,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")
        
        if prompt_text is not None and not prompt_text.strip():
            prompt_text = None

        has_prompt_audio = (prompt_wav_path is not None) or (prompt_waveform is not None)
        
        if has_prompt_audio and prompt_text is None:
            raise ValueError("prompt_text is required when providing prompt audio (path or waveform).")
        
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        
        temp_prompt_wav_path = None
        
        try:
            if has_prompt_audio and prompt_text is not None:
                if denoise and self.denoiser is not None and prompt_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_prompt_wav_path = tmp_file.name
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                    prompt_wav_path = temp_prompt_wav_path
                
                fixed_prompt_cache = self.tts_model.build_prompt_cache(
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_waveform=prompt_waveform,
                    prompt_sample_rate=prompt_sample_rate
                )
            else:
                fixed_prompt_cache = None  # will be built from the first inference
            
            # Text Normalization
            if normalize:
                if self.text_normalizer is None:
                    try:
                        from .utils.text_normalize import TextNormalizer
                        self.text_normalizer = TextNormalizer()
                    except ImportError:
                        print("[ComfyUI-VoxCPM] Warning: wetext dependency not found. Normalization skipped.")
                        # Mock the normalizer if import fails
                        class MockNormalizer:
                            def normalize(self, t): return t
                        self.text_normalizer = MockNormalizer()
                        
                text = self.text_normalizer.normalize(text)
            
            generate_result = self.tts_model._generate_with_prompt_cache(
                            target_text=text,
                            prompt_cache=fixed_prompt_cache,
                            min_len=min_len,
                            max_len=max_len,
                            inference_timesteps=inference_timesteps,
                            cfg_value=cfg_value,
                            retry_badcase=retry_badcase,
                            retry_badcase_max_times=retry_badcase_max_times,
                            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                            streaming=streaming,
                        )
        
            for wav, _, _ in generate_result:
                yield wav.squeeze(0).cpu().numpy()
        
        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # LoRA Interface
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        """Load LoRA weights from a checkpoint file."""
        if self.tts_model.lora_config is None:
            raise RuntimeError(
                "Cannot load LoRA weights: model was not initialized with LoRA config. "
            )
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        """Unload LoRA by resetting all LoRA weights to initial state."""
        self.tts_model.reset_lora_weights()
    
    def set_lora_enabled(self, enabled: bool):
        """Enable or disable LoRA layers."""
        self.tts_model.set_lora_enabled(enabled)
    
    def get_lora_state_dict(self) -> dict:
        """Get current LoRA parameters."""
        return self.tts_model.get_lora_state_dict()
    
    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA is currently configured."""
        return self.tts_model.lora_config is not None