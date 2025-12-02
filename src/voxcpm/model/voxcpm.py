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
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from pydantic import BaseModel
from tqdm import tqdm
from transformers import LlamaTokenizerFast

import comfy.model_management as model_management
from comfy.utils import ProgressBar

from ..modules.audiovae import AudioVAE
from ..modules.layers import ScalarQuantizationLayer
from ..modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiT
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import get_dtype, mask_multichar_chinese_tokens

logger = logging.getLogger(__name__)


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

    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"


class VoxCPMModel(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
    ):
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        # there was  hardcoded device logic.

        # Text-Semantic LM
        self.base_lm = MiniCPMModel(config.lm_config)
        # Defer cache setup until we know the device

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

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size
        self.sample_rate = audio_vae.sample_rate

    # For our custom device change
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.base_lm.setup_cache(1, self.config.max_length, device, dtype)
        self.residual_lm.setup_cache(1, self.config.max_length, device, dtype)
        return self

    def optimize(self):
        # Check the actual model device instead of torch.cuda.is_available()
        model_device = next(self.parameters()).device
        if model_device.type != "cuda":
            logger.info("Skipping torch.compile optimization as model is not on a CUDA device.")
            self.base_lm.forward_step = self.base_lm.forward_step
            self.residual_lm.forward_step = self.residual_lm.forward_step
            self.feat_encoder_step = self.feat_encoder
            self.feat_decoder.estimator = self.feat_decoder.estimator
            return self
        try:
            import triton
            self.base_lm.forward_step = torch.compile(self.base_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.residual_lm.forward_step = torch.compile(self.residual_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.feat_encoder_step = torch.compile(self.feat_encoder, mode="reduce-overhead", fullgraph=True)
            self.feat_decoder.estimator = torch.compile(self.feat_decoder.estimator, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            logger.error(f"Failed to optimize VoxCPMModel with torch.compile: {e}")
            logger.warning("Using original forward_step functions without compilation.")
            self.base_lm.forward_step = self.base_lm.forward_step
            self.residual_lm.forward_step = self.residual_lm.forward_step
            self.feat_encoder_step = self.feat_encoder
            self.feat_decoder.estimator = self.feat_decoder.estimator
        return self

    @torch.inference_mode()
    def generate(
	self, 
	target_text: str, 
	prompt_text: str = "", 
	prompt_wav_path: str = None, 
	prompt_waveform: torch.Tensor = None, 
	prompt_sample_rate: int = None, 
	min_len: int = 2, 
	max_len: int = 2000, 
	inference_timesteps: int = 10, 
	cfg_value: float = 2.0, 
	retry_badcase: bool = False, 
	retry_badcase_max_times: int = 3, 
	retry_badcase_ratio_threshold: float = 6.0
	):
	
        is_cloning = (prompt_waveform is not None or prompt_wav_path is not None) and prompt_text is not None
        if is_cloning:
            prompt_cache = self.build_prompt_cache(prompt_text, prompt_wav_path, prompt_waveform, prompt_sample_rate)
            result = self.generate_with_prompt_cache(target_text, prompt_cache, min_len, max_len, inference_timesteps, cfg_value, retry_badcase, retry_badcase_max_times, retry_badcase_ratio_threshold)
            return result[0]
        else:
            result = self.generate_with_prompt_cache(target_text, None, min_len, max_len, inference_timesteps, cfg_value, retry_badcase, retry_badcase_max_times, retry_badcase_ratio_threshold)
            return result[0]

    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_wav_path: str = None,
        prompt_waveform: torch.Tensor = None,
        prompt_sample_rate: int = None,
    ):
        """
        Build prompt cache for subsequent fast generation.
        
        Args:
            prompt_text: prompt text (required)
            prompt_wav_path: prompt audio path (required)
            
        Returns:
            prompt_cache: dict with text tokens and audio features
        """
        inference_device = next(self.parameters()).device
        
        if not prompt_text:
            raise ValueError("prompt_text is required to build a prompt cache.")
        if prompt_waveform is None and prompt_wav_path is None:
            raise ValueError("Either prompt_waveform or prompt_wav_path must be provided.")

        # Handle audio input directly from a tensor
        if prompt_waveform is not None:
            audio = prompt_waveform.clone() # Use a clone to avoid modifying the original
            sr = prompt_sample_rate
        else:
            # Fallback to loading from path if tensor is not provided
            audio, sr = torchaudio.load(prompt_wav_path)

        text_token = torch.LongTensor(self.text_tokenizer(prompt_text))
        
        # The audio tensor shape is [batch_size, channels, samples].
        # We check the channel dimension (dim=1).
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        patch_len = self.patch_size * self.chunk_size

        if audio.size(-1) % patch_len != 0:
            audio = torch.nn.functional.pad(audio, (0, patch_len - audio.size(-1) % patch_len))

        # (B, D, T)
        audio_feat = self.audio_vae.encode(audio.to(inference_device), self.sample_rate)

        audio_feat = audio_feat.view(
            self.audio_vae.latent_dim,
            -1,
            self.patch_size,
        ).permute(1, 2, 0) # (D, T, P)
        audio_feat = audio_feat[:-1, ...] # trick: remove the last padding token
        # build prompt cache
        prompt_cache = {
            "text_token": text_token,
            "audio_feat": audio_feat,
        }

        return prompt_cache

    
    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text_token: torch.Tensor,
        new_audio_feat: torch.Tensor,
    ):
        """
        Merge original prompt cache with newly generated content to stabilize voice.
        
        Args:
            original_cache: original prompt cache
            new_text_token: newly generated text tokens
            new_audio_feat: newly generated audio features
            
        Returns:
            merged_cache: merged cache
        """
        if original_cache is None:
            return {
                "text_token": new_text_token,
                "audio_feat": new_audio_feat,
            }
        original_text_token = original_cache["text_token"]
        original_audio_feat = original_cache["audio_feat"]
        merged_text_token = torch.cat([original_text_token, new_text_token], dim=0)
        merged_audio_feat = torch.cat([original_audio_feat, new_audio_feat], dim=0)

        # build new cache
        merged_cache = {
            "text_token": merged_text_token,
            "audio_feat": merged_audio_feat,
        }
        
        return merged_cache
    
    @torch.inference_mode()
    @torch.autocast(device_type="cuda")
    def generate_with_prompt_cache(
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
    ):
        """
        Generate audio using pre-built prompt cache.
        
        Args:
            target_text: Text to convert to speech
            prompt_cache: Cache built by build_prompt_cache (can be None)
            min_len: Minimum audio length to avoid very short audio
            max_len: Maximum audio length
            inference_timesteps: Number of diffusion sampling steps
            cfg_value: Classifier-free guidance value
            retry_badcase: Whether to retry on bad cases
            retry_badcase_max_times: Maximum retry attempts
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio
            
        Returns:
            tuple: (decoded audio tensor, new text tokens, new audio features)
        """
		
        # We need to define inference_device within this method's scope
        inference_device = next(self.parameters()).device
				
        # get prompt from cache
        if prompt_cache is None:
            # If no cache, create empty tensors directly on the target device
            prompt_text_token = torch.empty(0, dtype=torch.int32, device=inference_device)
            prompt_audio_feat = torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32, device=inference_device)
        else:
            # If cache exists, move its tensors to the target device
            prompt_text_token = prompt_cache["text_token"].to(inference_device)
            prompt_audio_feat = prompt_cache["audio_feat"].to(inference_device)
        
        # Create new text tokens on CPU first (as tokenizer does), then move to target device
        target_text_token = torch.LongTensor(self.text_tokenizer(target_text)).to(inference_device)
        
        # Concatenate text tokens on the target device
        text_token = torch.cat([prompt_text_token, target_text_token], dim=0)
        start_token = torch.tensor([self.audio_start_token], dtype=torch.int32, device=inference_device)
        text_token = torch.cat([text_token, start_token], dim=-1)

        audio_length = prompt_audio_feat.size(0)
        text_length = text_token.shape[0]
        
        # Create padding tensors directly on the target device
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=inference_device)
        audio_pad_feat = torch.zeros((text_length, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32, device=inference_device)
        
        # Now, all concatenations will succeed as all tensors are on the same device
        text_token = torch.cat([text_token, text_pad_token])
        audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
        text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(inference_device)
        audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(inference_device)

        text_token = text_token.unsqueeze(0)
        text_mask = text_mask.unsqueeze(0)
        audio_mask = audio_mask.unsqueeze(0)
        
        if inference_device.type in ["mps", "hip", "directml", "cpu"]:
            audio_feat_dtype = torch.float32
        else:
            audio_feat_dtype = torch.bfloat16
        audio_feat = audio_feat.unsqueeze(0).to(audio_feat_dtype)

        target_text_length = len(self.text_tokenizer(target_text))
        
        # Calculate a dynamic, safe maximum length based on the text.
        # This prevents runaway generation even if retries are disabled.
        dynamic_max_len = int(target_text_length * retry_badcase_ratio_threshold + 10)
        # Use the smaller of the hard limit (max_len) and our dynamic safety limit.
        effective_max_len = min(max_len, dynamic_max_len)

        retry_count = 0
        while True:
            latent_pred, pred_audio_feat = self.inference(
                text_token, 
				text_mask, 
				audio_feat, 
				audio_mask, 
                min_len=min_len, 
                max_len=effective_max_len,
                inference_timesteps=inference_timesteps, 
                cfg_value=cfg_value
            )
            retry_count += 1
            
            is_good_case = pred_audio_feat.shape[0] < target_text_length * retry_badcase_ratio_threshold
            
            if is_good_case:
                break
            
            if not retry_badcase:
                break
            
            if retry_count >= retry_badcase_max_times:
                break
            
            # If we reach here, it means the case was bad and we have more retries left.
            logger.warning(f"Bad case detected (audio/text ratio too high), retrying... ({retry_count}/{retry_badcase_max_times})")

        decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1).cpu()
        decode_audio = decode_audio[..., 640:-640]
        return (decode_audio, target_text_token.cpu(), pred_audio_feat)

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core inference method for audio generation.
        
        This is the main inference loop that generates audio features
        using the language model and diffusion transformer.
        
        Args:
            text: Input text tokens
            text_mask: Mask for text tokens
            feat: Input audio features
            feat_mask: Mask for audio features
            min_len: Minimum generation length
            max_len: Maximum generation length
            inference_timesteps: Number of diffusion steps
            cfg_value: Classifier-free guidance value
            
        Returns:
            Tuple containing:
                - Predicted latent features
                - Predicted audio feature sequence
        """
        B, T, P, D = feat.shape

        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
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
        
        pbar_comfy = ProgressBar(max_len)
        pbar_tqdm = tqdm(range(max_len), desc="VoxCPM Sampling")

        try:
            for i in pbar_tqdm:
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

                curr_embed = self.feat_encoder_step(pred_feat.unsqueeze(1))  # b, 1, c
                curr_embed = self.enc_to_lm_proj(curr_embed)

                pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
                prefix_feat_cond = pred_feat

                stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
                if i > min_len and stop_flag == 1:
                    pbar_comfy.update_absolute(max_len)
                    pbar_tqdm.update(max_len - pbar_tqdm.n)
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

        pred_feat_seq = torch.cat(pred_feat_seq, dim=1)

        feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
        feat_pred = feat_pred[..., 1:-1] # trick: remove the first and last token
        return feat_pred, pred_feat_seq.squeeze(0).cpu()

    @classmethod
    def from_local(cls, path: str):
        config = VoxCPMConfig.model_validate_json(open(os.path.join(path, "config.json")).read())

        tokenizer = LlamaTokenizerFast.from_pretrained(path)

        audio_vae = AudioVAE()
        vae_state_dict = torch.load(
            os.path.join(path, "audiovae.pth"),
            map_location="cpu",
            weights_only=True,
        )["state_dict"]

        model = cls(config, tokenizer, audio_vae)
        lm_dtype = get_dtype(config.dtype)
        model = model.to(lm_dtype)
        model.audio_vae = model.audio_vae.to(torch.float32)
        # Handle data type for non-CUDA devices on initial load
        if next(model.parameters()).device.type in ["mps", "hip", "directml"]:
            logger.warning(f"Converting model to float32 for {next(model.parameters()).device.type} compatibility.")
            model = model.to(torch.float32)

        model_state_dict = torch.load(
            os.path.join(path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )["state_dict"]

        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val
        model.load_state_dict(model_state_dict, strict=True)
        
        return model.eval().optimize()
