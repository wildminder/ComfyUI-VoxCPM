import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import argbind
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from ..model.voxcpm import VoxCPMConfig
from ..modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker


DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_ID_COLUMN = "dataset_id"


@argbind.bind()
def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    dataset_id_column: str = DEFAULT_ID_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
) -> Tuple[Dataset, Optional[Dataset]]:
    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset) -> Dataset:
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        # We cast to Audio to ensure proper handling during training, 
        # but for length calculation we might need raw path or duration if available.
        # HF datasets usually don't compute duration automatically for 'Audio' column.
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)
        if dataset_id_column and dataset_id_column in ds.column_names:
            if dataset_id_column != DEFAULT_ID_COLUMN:
                ds = ds.rename_column(dataset_id_column, DEFAULT_ID_COLUMN)
        else:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"])
    val_ds = prepare(dataset_dict["validation"]) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: int = 25,
    patch_size: int = 1,
) -> List[int]:
    """
    预估每个样本经过 packer 之后的大致序列长度（text+audio），用于过滤超长样本。

    逻辑与 AudioFeatureProcessingPacker / AudioVAE 一致：
    - 文本长度: len(text_ids)
    - 音频长度:
        duration(s) * audio_vae_fps -> 近似 VAE 帧数 t_vae
        t_seq = ceil(t_vae / patch_size)
    - 序列总长约为: text_len + t_seq + 2
    """
    lengths: List[int] = []

    has_duration = "duration" in ds.column_names

    for i in range(len(ds)):
        item = ds[i]
        text_len = len(item["text_ids"])

        # 音频时长（尽量不解码；若 manifest 里已有 duration 列则优先使用）
        if has_duration:
            duration = float(item["duration"])
        else:
            audio = item[DEFAULT_AUDIO_COLUMN]
            duration = len(audio["array"]) / float(audio["sampling_rate"])

        t_vae = math.ceil(duration * audio_vae_fps)
        t_seq = math.ceil(t_vae / patch_size)

        total_len = text_len + t_seq + 2
        lengths.append(total_len)

    return lengths


class HFVoxCPMDataset(TorchDataset):
    """
    Thin wrapper around a tokenized HuggingFace dataset that returns
    PyTorch-friendly samples.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]
        return {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        return {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }


class BatchProcessor:
    """
    Wraps ``AudioFeatureProcessingPacker`` so the training loop can mirror
    the minicpm-audio mechanics.
    """

    def __init__(
        self,
        *,
        config: VoxCPMConfig,
        audio_vae: AudioVAE,
        dataset_cnt: int,
        device: torch.device,
    ):
        self.device = device
        self.dataset_cnt = dataset_cnt
        self.audio_vae = audio_vae
        self.audio_vae.to(device)
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_tokens = batch["audio_tokens"].to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        task_ids = batch["task_ids"].to(self.device)
        dataset_ids = batch["dataset_ids"].to(self.device)

        packed = self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
        )
        return packed


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)
    # Standard padding-based batching; Accelerator will attach DistributedSampler if needed.
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
    )

