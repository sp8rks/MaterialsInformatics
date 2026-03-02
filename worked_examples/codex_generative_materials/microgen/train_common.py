"""Shared training utilities for generative model demos."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from microgen.data import load_saved_dataset
from microgen.synth import generate_dataset as synth_generate_dataset


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> tuple[Path, Path]:
    ckpt_dir = Path("checkpoints")
    out_dir = Path("outputs")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, out_dir


def load_training_tensor(
    data_path: str | None,
    num_samples: int,
    image_size: int,
    kind: str,
    seed: int,
    normalize_to_neg1: bool = False,
) -> tuple[torch.Tensor, dict]:
    if data_path:
        images, _labels, metadata = load_saved_dataset(data_path)
    else:
        images, _labels, metadata = synth_generate_dataset(
            num_samples=num_samples,
            image_size=image_size,
            kind=kind,
            seed=seed,
        )

    x = torch.from_numpy(images).float().unsqueeze(1)
    if normalize_to_neg1:
        x = x * 2.0 - 1.0
    return x, metadata


def make_dataloader(x: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(x)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def save_grid(images: torch.Tensor, path: str | Path, normalize: bool = False, value_range=None, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, str(path), nrow=nrow, normalize=normalize, value_range=value_range)
