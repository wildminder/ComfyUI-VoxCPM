
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class AudioFeatureProcessingPacker:
    """
    Adapted from the minicpm-audio training utilities. It converts raw text and
    audio tokens into the packed multimodal representation required by VoxCPM.
    """

    def __init__(self, dataset_cnt: int, max_len: int, patch_size: int, feat_dim: int, audio_vae: nn.Module):
        self.audio_start_id = 101
        self.audio_end_id = 102
        # unused now 
        self.audio_prompt_start_id = 103
        self.audio_prompt_end_id = 104
        self.text_eos_token_id = 2

        self.patch_size = patch_size
        self.patch_len = audio_vae.hop_length * self.patch_size
        self.feat_dim = feat_dim
        self.dataset_cnt = max(dataset_cnt, 1)
        self.max_len = max_len

        self.audio_vae = audio_vae

        self.process_functions = {"tts": self.process_tts_data}
        self.task_id_map = {"tts": 1}
        self.id_to_task = {idx: usage for usage, idx in self.task_id_map.items()}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_pad_position(tokens: torch.Tensor):
        positions = (tokens == -100).nonzero(as_tuple=True)
        if positions[0].numel() == 0:
            return None
        return int(positions[0][0])

    def unpad_text_tokens(self, tokens: torch.Tensor):
        pad_pos = self._first_pad_position(tokens)
        return tokens if pad_pos is None else tokens[:pad_pos]

    def unpad_audio_tokens(self, tokens: torch.Tensor):
        pad_pos = self._first_pad_position(tokens)
        return tokens if pad_pos is None else tokens[:pad_pos]

    def encode_audio(self, wav: torch.Tensor):
        """
        Encode raw waveform into latent features using AudioVAE.

        AudioVAE.encode expects shape [B, 1, T'] and returns [B, D, T].
        We then transpose to [B, T, D] to match downstream expectations.
        """
        wav = wav.unsqueeze(0)  # [1, T]
        wav = wav.unsqueeze(1)  # [1, 1, T]
        wav_len = wav.size(-1)
        if wav_len % self.patch_len != 0:
            padding_size = self.patch_len - wav_len % self.patch_len
            wav = torch.nn.functional.pad(wav, (0, padding_size))

        with torch.no_grad():
            z = self.audio_vae.encode(wav, self.audio_vae.sample_rate)  # [1, D, T']
            feat = z.transpose(1, 2)  # [1, T', D]
        return feat

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        task_ids: torch.Tensor,
        dataset_ids: torch.Tensor,
        is_prompts: List[bool],
    ) -> Dict[str, torch.Tensor]:
        """
        Padding-based batching: each sample in the input batch is processed
        independently and then padded to a common length (capped by ``max_len``).
        The result tensors all have shape [B, T, ...].
        """
        device = audio_tokens.device
        max_dataset_id = int(dataset_ids.max().item()) if dataset_ids.numel() > 0 else -1
        dataset_cnt = max(self.dataset_cnt, max_dataset_id + 1)

        text_tokens_list: List[torch.Tensor] = []
        audio_feats_list: List[torch.Tensor] = []
        text_mask_list: List[torch.Tensor] = []
        audio_mask_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        audio_task_ids_list: List[torch.Tensor] = []
        audio_dataset_ids_list: List[torch.Tensor] = []
        lengths: List[int] = []

        audio_duration_consumed = torch.zeros(dataset_cnt, dtype=torch.float32, device=device)
        text_token_consumed = torch.zeros(dataset_cnt, dtype=torch.float32, device=device)

        for audio_token, text_token, task_id, dataset_idx, is_prompt in zip(
            audio_tokens, text_tokens, task_ids.tolist(), dataset_ids.tolist(), is_prompts
        ):
            unpad_audio_token = self.unpad_audio_tokens(audio_token).to(torch.float32)
            unpad_text_token = self.unpad_text_tokens(text_token)
            usage = self.id_to_task[task_id]

            (
                packed_text,
                audio_feat,
                text_mask,
                audio_mask,
                loss_mask,
                labels,
                audio_duration,
                text_token_count,
            ) = self.process_functions[usage](unpad_audio_token, unpad_text_token, is_prompt)

            audio_duration_consumed[dataset_idx] += audio_duration
            text_token_consumed[dataset_idx] += text_token_count

            audio_task_id = torch.zeros_like(audio_mask)
            audio_task_id[audio_mask == 1] = self.task_id_map[usage]

            audio_dataset_id = torch.zeros_like(audio_mask)
            audio_dataset_id[audio_mask == 1] = dataset_idx + 1

            text_tokens_list.append(packed_text)
            text_mask_list.append(text_mask)
            audio_feats_list.append(audio_feat)
            audio_mask_list.append(audio_mask)
            loss_mask_list.append(loss_mask)
            labels_list.append(labels)
            audio_task_ids_list.append(audio_task_id)
            audio_dataset_ids_list.append(audio_dataset_id)
            lengths.append(packed_text.shape[0])

        # Determine padded length per batch (cap by self.max_len)
        if lengths:
            max_len = min(self.max_len, max(lengths))
        else:
            max_len = self.max_len

        def pad_1d(x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
            if x.size(0) >= max_len:
                return x[: max_len]
            pad = torch.full((max_len - x.size(0),), pad_value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0)

        def pad_3d(x: torch.Tensor) -> torch.Tensor:
            # x: [T, P, D]
            if x.size(0) >= max_len:
                return x[: max_len]
            pad = torch.zeros(
                (max_len - x.size(0),) + x.shape[1:], dtype=x.dtype, device=x.device
            )
            return torch.cat([x, pad], dim=0)
        if lengths:
            text_tokens_batch = torch.stack([pad_1d(t, pad_value=0) for t in text_tokens_list], dim=0)
            text_mask_batch = torch.stack([pad_1d(m, pad_value=0) for m in text_mask_list], dim=0)
            audio_feats_batch = torch.stack([pad_3d(f) for f in audio_feats_list], dim=0)
            audio_mask_batch = torch.stack([pad_1d(m, pad_value=0) for m in audio_mask_list], dim=0)
            loss_mask_batch = torch.stack([pad_1d(m, pad_value=0) for m in loss_mask_list], dim=0)
            labels_batch = torch.stack([pad_1d(l, pad_value=0) for l in labels_list], dim=0)
            audio_task_ids_batch = torch.stack(
                [pad_1d(t, pad_value=0) for t in audio_task_ids_list], dim=0
            )
            audio_dataset_ids_batch = torch.stack(
                [pad_1d(d, pad_value=0) for d in audio_dataset_ids_list], dim=0
            )

            # Position ids: [B, T], simple 0..L_i-1 then padded with 0
            position_ids_list = []
            for L in lengths:
                L_clip = min(L, max_len)
                pos = torch.arange(0, L_clip, device=device)
                if L_clip < max_len:
                    pad = torch.zeros(max_len - L_clip, dtype=pos.dtype, device=device)
                    pos = torch.cat([pos, pad], dim=0)
                position_ids_list.append(pos)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            # Empty batch fallback (shouldn't really happen)
            text_tokens_batch = torch.zeros((0, self.max_len), dtype=torch.int32, device=device)
            text_mask_batch = torch.zeros_like(text_tokens_batch)
            audio_feats_batch = torch.zeros(
                (0, self.max_len, self.patch_size, self.feat_dim), dtype=torch.float32, device=device
            )
            audio_mask_batch = torch.zeros_like(text_tokens_batch)
            loss_mask_batch = torch.zeros_like(text_tokens_batch)
            labels_batch = torch.zeros_like(text_tokens_batch)
            audio_task_ids_batch = torch.zeros_like(text_tokens_batch)
            audio_dataset_ids_batch = torch.zeros_like(text_tokens_batch)
            position_ids = torch.zeros_like(text_tokens_batch)

        audio_duration_consumed = audio_duration_consumed.to(torch.long)
        text_token_consumed = text_token_consumed.to(torch.long)

        return {
            "text_tokens": text_tokens_batch,
            "audio_feats": audio_feats_batch,
            "text_mask": text_mask_batch,
            "audio_mask": audio_mask_batch,
            "loss_mask": loss_mask_batch,
            "position_ids": position_ids,
            "labels": labels_batch,
            "audio_task_ids": audio_task_ids_batch,
            "audio_dataset_ids": audio_dataset_ids_batch,
            "audio_duration_consumed": audio_duration_consumed,
            "text_token_consumed": text_token_consumed,
        }

    # ------------------------------------------------------------------ #
    # Feature extraction helpers
    # ------------------------------------------------------------------ #
    def extract_audio_feats(self, audio_data: torch.Tensor):
        audio_feats = self.encode_audio(audio_data)
        if audio_feats.size(1) % self.patch_size != 0:
            audio_feats_ = audio_feats.transpose(1, 2)
            padding = nn.functional.pad(audio_feats_, (0, self.patch_size - audio_feats.size(1) % self.patch_size))
            audio_feats = padding.transpose(1, 2)

        audio_duration = audio_feats.size(1) / 25
        audio_feats = rearrange(audio_feats, "b (t p) c -> b t p c", p=self.patch_size)
        return audio_feats, audio_duration

    def process_tts_data(self, audio_token: torch.Tensor, text_token: torch.Tensor, is_prompt: bool = False):
        text_token_info = torch.cat(
            [
                text_token,
                torch.tensor(
                    [self.audio_prompt_start_id if is_prompt else self.audio_start_id],
                    dtype=torch.int32,
                    device=text_token.device,
                ),
            ],
            dim=-1,
        )
        text_token_count = len(text_token)
        text_length = text_token_info.shape[0]
        audio_feat_info, audio_duration = self.extract_audio_feats(audio_token)
        audio_feat_info = audio_feat_info.squeeze(0)
        audio_length = audio_feat_info.shape[0]

        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
        text_token_info = torch.cat(
            [
                text_token_info,
                text_pad_token,
                torch.tensor(
                    [self.audio_prompt_end_id if is_prompt else self.audio_end_id],
                    dtype=torch.int32,
                    device=text_token.device,
                ),
            ]
        )
        audio_pad_feat = torch.zeros(
            (text_length, self.patch_size, audio_feat_info.size(-1)),
            dtype=torch.float32,
            device=text_token.device,
        )
        audio_feat_info = torch.cat([audio_pad_feat, audio_feat_info, audio_pad_feat[0:1, ...]], dim=0)

        text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length), torch.ones(1)]).type(torch.int32).to(
            text_token.device
        )
        audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length), torch.zeros(1)]).type(
            torch.int32
        ).to(text_token.device)
        loss_mask = torch.cat([torch.zeros(text_length), torch.zeros(audio_length) if is_prompt else torch.ones(audio_length), torch.zeros(1)]).type(torch.int32).to(text_token.device)

        labels = torch.zeros(text_length + audio_length + 1).type(torch.int32).to(text_token.device)
        labels[-2] = 1

        return (
            text_token_info,
            audio_feat_info,
            text_mask,
            audio_mask,
            loss_mask,
            labels,
            audio_duration,
            text_token_count,
        )

