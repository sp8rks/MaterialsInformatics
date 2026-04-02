"""Dataset loading utilities for synthetic microstructures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_saved_dataset(npz_path: str | Path, metadata_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load images, labels, and metadata from disk."""
    npz_path = Path(npz_path)
    if metadata_path is None:
        metadata_path = npz_path.with_suffix(".json")
    metadata_path = Path(metadata_path)

    with np.load(npz_path) as data:
        images = data["images"].astype(np.float32)
        labels = data["labels"].astype(np.int64)

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return images, labels, metadata


class MicrostructureDataset(Dataset):
    """Torch dataset wrapper around saved microstructure arrays."""

    def __init__(self, npz_path: str | Path, normalize_to_neg1: bool = False) -> None:
        images, labels, metadata = load_saved_dataset(npz_path)
        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.normalize_to_neg1 = normalize_to_neg1

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        if self.normalize_to_neg1:
            image = image * 2.0 - 1.0
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label_tensor
