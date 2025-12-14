"""
VoxCPM: A Tokenizer-free speech generation model

This module contains the main VoxCPM model implementation, including configuration classes
and the core VoxCPMModel for text-to-speech generation.

Copyright 2025 OpenBMB
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from typing import Tuple, Union, Generator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import warnings
from einops import rearrange
from pydantic import BaseModel

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
from tqdm import tqdm
from transformers import LlamaTokenizerFast

# ComfyUI imports
import comfy.model_management as model_management
from comfy.utils import ProgressBar

from ..modules.audiovae import AudioVAE, AudioVAEConfig
from ..modules.layers import ScalarQuantizationLayer
from ..modules.layers.lora import apply_lora_to_named_linear_modules
from ..modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiT
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import get_dtype, mask_multichar_chinese_tokens


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None


class VoxCPMDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None

    cfm_config: CfmConfig


class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 2
    feat_dim: int = 64
    residual_lm_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: Optional[AudioVAEConfig] = None

    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"
    dit_mean_mode: bool = False


class LoRAConfig(BaseModel):
    enable_lm: bool = False        # Apply LoRA to base_lm + residual_lm
    enable_dit: bool = False       # Apply LoRA to VoxCPMLocDiT
    enable_proj: bool = False      # Apply LoRA to projection Linear layers

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0

    # Target linear layer names for LM & DiT (matched by attribute name)
    target_modules_lm: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    target_modules_dit: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    # Projection layer attribute names to find on VoxCPMModel
    target_proj_modules: list[str] = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]


VoxCPMConfig.model_rebuild()


class VoxCPMModel(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
        lora_config: LoRAConfig = None,
    ):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.device = config.device
        
        # ComfyUI handles device management generally, but we keep this for initialization logic
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Text-Semantic LM
        self.base_lm = MiniCPMModel(config.lm_config)
        # Note: Cache setup happens in .to()

        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        self.residual_lm = MiniCPMModel(residual_lm_config)
        
        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiT(decoder_config, in_channels=config.feat_dim),
            mean_mode=config.dit_mean_mode,
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size, 
            config.lm_config.hidden_size, 
            config.scalar_quantization_latent_dim, 
            config.scalar_quantization_scale
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size
        self.sample_rate = audio_vae.sample_rate

        if self.lora_config is not None:
            self._apply_lora()

    def to(self, *args, **kwargs):
        # Override to setup cache when moving to device
        super().to(*args, **kwargs)
        # Identify the device and dtype from the first parameter
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Re-setup caches with correct device/dtype
        # This is critical for ComfyUI as it moves models between CPU and GPU
        self.base_lm.setup_cache(1, self.config.max_length, device, dtype)
        self.residual_lm.setup_cache(1, self.config.max_length, device, dtype)
        self.device = device
        return self

    def _apply_lora(self):
        """Inject LoRA into LM / DiT / Projection layers"""
        cfg = self.lora_config
        lora_kwargs = dict(r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)

        # LM: base_lm + residual_lm
        if cfg.enable_lm:
            for lm in [self.base_lm, self.residual_lm]:
                apply_lora_to_named_linear_modules(
                    lm, target_submodule_names=cfg.target_modules_lm, **lora_kwargs
                )

        # DiT: feat_decoder.estimator
        if cfg.enable_dit:
            apply_lora_to_named_linear_modules(
                self.feat_decoder.estimator, target_submodule_names=cfg.target_modules_dit, **lora_kwargs
            )

        # Projection layers
        if cfg.enable_proj:
            from ..modules.layers.lora import LoRALinear
            for attr_name in cfg.target_proj_modules:
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Linear):
                    setattr(self, attr_name, LoRALinear(base=module, **lora_kwargs))

    def optimize(self, disable: bool = False):
        if disable:
            return self
        
        # Check actual model device
        model_device = next(self.parameters()).device
        if model_device.type != "cuda":
            print("Skipping torch.compile optimization as model is not on a CUDA device.")
            return self

        try:
            import triton
            self.base_lm.forward_step = torch.compile(self.base_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.residual_lm.forward_step = torch.compile(self.residual_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.feat_encoder = torch.compile(self.feat_encoder, mode="reduce-overhead", fullgraph=True)
            self.feat_decoder.estimator = torch.compile(self.feat_decoder.estimator, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"Warning: torch.compile disabled - {e}")
        return self

    def _dtype(self):
        return get_dtype(self.config.dtype)

    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        prompt_waveform: torch.Tensor = None, # ComfyUI: Direct tensor input
        prompt_sample_rate: int = None,       # ComfyUI: Sample rate for tensor
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
            
        device = next(self.parameters()).device # Use actual device
        
        has_prompt = (len(prompt_wav_path) > 0) or (prompt_waveform is not None)

        if not has_prompt:
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor(
                        [self.audio_start_token],
                        dtype=torch.int32,
                        device=text_token.device,
                    ),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_mask = torch.ones(text_length).type(torch.int32).to(text_token.device)
            audio_mask = torch.zeros(text_length).type(torch.int32).to(text_token.device)

        else:
            # Need to test: Only use target_text to avoid repeating the prompt. Gives better output
            text = target_text 
            
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            # ComfyUI Modification: Load from tensor if available, else path
            # ... do not like path stuff
            if prompt_waveform is not None:
                audio = prompt_waveform.clone()
                sr = prompt_sample_rate
            else:
                audio, sr = torchaudio.load(prompt_wav_path)
            
            # Ensure audio is on correct device for resampling/processing if needed
            # But torchaudio resampling might prefer CPU or GPU depending on version. 
            # We'll keep it on original device (likely CPU) until VAE encode.
            
            # Mix down to mono if needed
            if audio.dim() > 1 and audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Note: If Mono [1, Samples], the above check fails (1 is not > 1), so it remains [1, Samples]. Correct.

            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            patch_len = self.patch_size * self.chunk_size

            if audio.size(-1) % patch_len != 0:
                # Left padding
                padding_size = patch_len - audio.size(-1) % patch_len
                audio = torch.nn.functional.pad(audio, (padding_size, 0))

            # (B, D, T)
            # Encode on device
            audio_feat = self.audio_vae.encode(audio.to(device), self.sample_rate).cpu()
            audio_feat = audio_feat.view(
                self.audio_vae.latent_dim,
                -1,
                self.patch_size,
            ).permute(1, 2, 0)
            audio_length = audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            text_token = torch.cat([text_token, text_pad_token])
            audio_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            audio_feat = torch.cat([audio_pad_feat, audio_feat], dim=0)
            text_mask = (
                torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
            )
            audio_mask = (
                torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)
            )

        text_token = text_token.unsqueeze(0).to(device)
        text_mask = text_mask.unsqueeze(0).to(device)
        # Handle dtype for feature
        feat_dtype = get_dtype(self.config.dtype)
        # MPS/DirectML check for bf16 fallback happens in utils.get_dtype or upstream
        audio_feat = audio_feat.unsqueeze(0).to(device).to(feat_dtype)
        audio_mask = audio_mask.unsqueeze(0).to(device)

        target_text_length = len(self.text_tokenizer(target_text))
        
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len), # avoid too long audio
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, _ in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...")
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break   
                
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1).cpu()  
            yield decode_audio        
    
    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_wav_path: str = None,
        prompt_waveform: torch.Tensor = None, # ComfyUI Addition
        prompt_sample_rate: int = None,       # ComfyUI Addition
    ):
        """
        Build prompt cache for subsequent fast generation.
        """
        if not prompt_text:
            raise ValueError("prompt_text is required")
        if not prompt_wav_path and prompt_waveform is None:
            raise ValueError("prompt_wav_path or prompt_waveform is required")

        # ComfyUI: Load from tensor or file
        if prompt_waveform is not None:
            audio = prompt_waveform.clone()
            sr = prompt_sample_rate
        else:
            audio, sr = torchaudio.load(prompt_wav_path)
            
        # Simplified standard mix logic here as well
        if audio.dim() > 1 and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        patch_len = self.patch_size * self.chunk_size

        if audio.size(-1) % patch_len != 0:
            # Left padding
            padding_size = patch_len - audio.size(-1) % patch_len
            audio = torch.nn.functional.pad(audio, (padding_size, 0))

        # extract audio features - ensure device correctness
        device = next(self.parameters()).device
        audio_feat = self.audio_vae.encode(audio.to(device), self.sample_rate).cpu()

        audio_feat = audio_feat.view(
            self.audio_vae.latent_dim,
            -1,
            self.patch_size,
        ).permute(1, 2, 0) # (D, T, P)
        
        prompt_cache = {
            "prompt_text": prompt_text,
            "audio_feat": audio_feat,
        }
        
        return prompt_cache

    # ... merge_prompt_cache remains same ...
    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text: str,
        new_audio_feat: torch.Tensor,
    ):
        if original_cache is None:
            return {
                "prompt_text": new_text,
                "audio_feat": new_audio_feat,
            }
        original_prompt_text = original_cache["prompt_text"]
        original_audio_feat = original_cache["audio_feat"]
        merged_prompt_text = original_prompt_text + new_text
        merged_audio_feat = torch.cat([original_audio_feat, new_audio_feat], dim=0)

        merged_cache = {
            "prompt_text": merged_prompt_text,
            "audio_feat": merged_audio_feat,
        }
        return merged_cache

    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))

    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
            
        device = next(self.parameters()).device

        # get prompt from cache
        if prompt_cache is None:
            prompt_audio_feat = torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
            text = target_text
        else:
            prompt_audio_feat = prompt_cache["audio_feat"]
            prompt_text = prompt_cache["prompt_text"]
            # again: Do not append prompt_text to target_text
            text = target_text 
        
        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor(
                    [self.audio_start_token],
                    dtype=torch.int32,
                    device=text_token.device,
                ),
            ],
            dim=-1,
        )
        
        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

        audio_length = prompt_audio_feat.size(0)
        text_length = text_token.shape[0]
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
        audio_pad_feat = torch.zeros(
            (text_token.shape[0], self.patch_size, self.audio_vae.latent_dim),
            dtype=torch.float32,
            device=text_token.device,
        )
        text_token = torch.cat([text_token, text_pad_token])
        audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
        text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
        audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)

        text_token = text_token.unsqueeze(0).to(device)
        text_mask = text_mask.unsqueeze(0).to(device)
        # Handle features on device with correct dtype
        feat_dtype = get_dtype(self.config.dtype)
        audio_feat = audio_feat.unsqueeze(0).to(device).to(feat_dtype)
        audio_mask = audio_mask.unsqueeze(0).to(device)
    
        # run inference
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len), # avoid too long audio
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, pred_audio_feat in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield (
                        decode_audio,
                        target_text_token,
                        pred_audio_feat
                    )
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...")
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1).cpu()

            yield (
                decode_audio,
                target_text_token,
                pred_audio_feat
            )

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._inference(*args, streaming=False, **kwargs))
    
    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        return self._inference(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """Core inference loop with ComfyUI progress reporting"""
        
        B, T, P, D = feat.shape

        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
       
        text_embed = self.base_lm.embed_tokens(text) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None

        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)
        
        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

         
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]

        # ComfyUI Progress Bar integration
        pbar_comfy = ProgressBar(max_len)
        pbar_tqdm = tqdm(range(max_len), desc="VoxCPM Sampling")

        try:
            for i in pbar_tqdm:
                # ComfyUI cancellation check
                model_management.throw_exception_if_processing_interrupted()

                dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
                dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
                dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit]

                pred_feat = self.feat_decoder(
                    mu=dit_hidden,
                    patch_size=self.patch_size,
                    cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                    n_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                ).transpose(
                    1, 2
                )  # [b, p, d]
                
                curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
                curr_embed = self.enc_to_lm_proj(curr_embed)
                
                pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
                prefix_feat_cond = pred_feat

                if streaming:
                    pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                    feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                    yield feat_pred, pred_feat_seq
                
                stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
                
                if i > min_len and stop_flag == 1:
                    pbar_comfy.update_absolute(max_len)
                    break
        
                lm_hidden = self.base_lm.forward_step(
                    curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
                ).clone()
            
                lm_hidden = self.fsq_layer(lm_hidden)
                residual_hidden = self.residual_lm.forward_step(
                    lm_hidden + curr_embed[:, 0, :], torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device)
                ).clone()
                
                pbar_comfy.update(1)
        finally:
            pbar_tqdm.close()
                
        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)  
            yield feat_pred, pred_feat_seq.squeeze(0).cpu()
            

    @classmethod
    def from_local(cls, path: str, optimize: bool = True, training: bool = False, lora_config: LoRAConfig = None):
        config = VoxCPMConfig.model_validate_json(open(os.path.join(path, "config.json")).read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        audio_vae_config = getattr(config, 'audio_vae_config', None)
        audio_vae = AudioVAE(config=audio_vae_config) if audio_vae_config else AudioVAE()
        vae_state_dict = torch.load(
            os.path.join(path, "audiovae.pth"),
            map_location="cpu",
            weights_only=True,
        )["state_dict"]
        model = cls(config, tokenizer, audio_vae, lora_config)
        if not training:
            lm_dtype = get_dtype(model.config.dtype)
            model = model.to(lm_dtype)
        else: # training mode
            for name, param in model.named_parameters():
                if "audio_vae" in name: # freeze VAE weights
                    param.requires_grad = False
                    continue
                if lora_config is not None:
                    if "lora" not in name: # freeze non-LoRA weights
                        param.requires_grad = False
        model.audio_vae = model.audio_vae.to(torch.float32)
        
        # Handle non-CUDA initialization for ComfyUI environment
        # If we are not on CUDA/CPU/MPS yet, force float32 to avoid half-precision errors on init
        current_dev_type = next(model.parameters()).device.type
        if current_dev_type in ["mps", "hip", "directml", "cpu"] and lm_dtype in [torch.float16, torch.bfloat16]:
             # Only convert if actually running on these devices during init
             pass 

        # Try to load from safetensors first, fallback to pytorch_model.bin
        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading model from safetensors: {safetensors_path}")
            model_state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_model_path):
            print(f"Loading model from pytorch_model.bin: {pytorch_model_path}")
            checkpoint = torch.load(
                pytorch_model_path,
                map_location="cpu",
                weights_only=True,
            )
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"Model file not found. Expected either {safetensors_path} or {pytorch_model_path}"
            )
            
        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val
        
        model.load_state_dict(model_state_dict, strict=False)
        if training:
            return model
        # Don't move to device here, let ComfyUI patcher handle it via .patch_model which will call .to()
        # But we do need to eval()
        return model.eval().optimize(disable=not optimize)

    # ------------------------------------------------------------------ #
    # LoRA Weight Management
    # ------------------------------------------------------------------ #
    def _iter_lora_modules(self):
        """Iterate over all LoRA modules."""
        from ..modules.layers.lora import LoRALinear
        for module in self.modules():
            if isinstance(module, LoRALinear):
                yield module

    def load_lora_weights(self, lora_path: str, device: str = None):
        """
        Load LoRA weights from file.
        """
        from pathlib import Path
        
        device = device or self.device
        lora_path = Path(lora_path)
        
        if lora_path.is_dir():
            safetensors_file = lora_path / "lora_weights.safetensors"
            ckpt_file = lora_path / "lora_weights.ckpt"
        else:
            safetensors_file = lora_path if lora_path.suffix == ".safetensors" else None
            ckpt_file = lora_path if lora_path.suffix in [".ckpt", ".pth"] else None
        
        if safetensors_file and safetensors_file.exists() and SAFETENSORS_AVAILABLE:
            # Load to CPU first to avoid device incompatibility in safetensors loading
            state_dict = load_file(str(safetensors_file), device="cpu")
        elif ckpt_file and ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            raise FileNotFoundError(
                f"LoRA checkpoint not found. Expected either {safetensors_file} or {ckpt_file}"
            )
        
        model_params = dict(self.named_parameters())
        key_mapping = {k.replace("._orig_mod.", "."): k for k in model_params if "._orig_mod." in k}
        
        loaded_keys, skipped_keys = [], []
        for key, value in state_dict.items():
            target_key = key if key in model_params else key_mapping.get(key)
            if target_key:
                model_params[target_key].data.copy_(value.to(device))
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        
        return loaded_keys, skipped_keys

    def set_lora_enabled(self, enabled: bool):
        for module in self._iter_lora_modules():
            module.set_enabled(enabled)

    def reset_lora_weights(self):
        for module in self._iter_lora_modules():
            module.reset_lora_parameters()

    def get_lora_state_dict(self) -> dict:
        return {name: param.data.clone() 
                for name, param in self.named_parameters() 
                if "lora_" in name}