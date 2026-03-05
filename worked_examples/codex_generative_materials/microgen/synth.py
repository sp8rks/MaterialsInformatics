"""Procedural synthetic microstructure generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LabelMap = {"porous": 0, "precipitate": 1, "eutectic": 2}


@dataclass(frozen=True)
class SynthConfig:
    """Configuration for dataset synthesis."""

    image_size: int = 64
    porous_porosity: float = 0.45
    precipitate_count_range: tuple[int, int] = (20, 60)
    eutectic_frequency_range: tuple[float, float] = (5.0, 12.0)


def _ensure_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mean_blur(image: np.ndarray, passes: int = 2) -> np.ndarray:
    """Fast blur using a 3x3 neighborhood average."""
    out = image.astype(np.float32, copy=True)
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="reflect")
        out = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return out


def generate_porous(size: int = 64, porosity: float = 0.45, seed: int | None = None) -> np.ndarray:
    """Generate a porous binary-like microstructure image in [0, 1]."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if not 0.05 <= porosity <= 0.95:
        raise ValueError("porosity must be in [0.05, 0.95]")

    rng = _ensure_rng(seed)
    field = rng.normal(loc=0.0, scale=1.0, size=(size, size)).astype(np.float32)

    centers = rng.integers(0, size, size=(max(8, size // 2), 2))
    yy, xx = np.ogrid[:size, :size]
    for cy, cx in centers:
        radius = rng.uniform(size * 0.03, size * 0.12)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        field[mask] -= rng.uniform(0.5, 1.5)

    smooth = _mean_blur(field, passes=3)
    thresh = np.quantile(smooth, porosity)
    pores = (smooth < thresh).astype(np.float32)
    image = 1.0 - pores

    image = _mean_blur(image, passes=1)
    image += rng.normal(0.0, 0.04, size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def generate_precipitate(size: int = 64, count_range: tuple[int, int] = (20, 60), seed: int | None = None) -> np.ndarray:
    """Generate a precipitate-rich microstructure image in [0, 1]."""
    if size <= 0:
        raise ValueError("size must be > 0")
    lo, hi = count_range
    if lo <= 0 or hi < lo:
        raise ValueError("count_range must be positive and increasing")

    rng = _ensure_rng(seed)
    image = np.full((size, size), 0.25, dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    count = int(rng.integers(lo, hi + 1))

    for _ in range(count):
        cy = rng.uniform(0, size - 1)
        cx = rng.uniform(0, size - 1)
        sigma = rng.uniform(size * 0.015, size * 0.07)
        amplitude = rng.uniform(0.3, 1.0)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        image += amplitude * np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)

    image = _mean_blur(image, passes=1)
    image += rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    image -= image.min()
    image /= max(image.max(), 1e-6)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def generate_eutectic(
    size: int = 64,
    frequency_range: tuple[float, float] = (5.0, 12.0),
    seed: int | None = None,
) -> np.ndarray:
    """Generate a lamellar-like eutectic microstructure image in [0, 1]."""
    if size <= 0:
        raise ValueError("size must be > 0")
    lo, hi = frequency_range
    if lo <= 0.0 or hi < lo:
        raise ValueError("frequency_range must be positive and increasing")

    rng = _ensure_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    theta = float(rng.uniform(0.0, 2.0 * np.pi))

    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)

    domain_field = _mean_blur(rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32), passes=4)
    local_phase = 2.0 * np.pi * domain_field

    base_cycles = float(rng.uniform(lo, hi))
    wavelength = size / base_cycles
    lamellar = np.sin((2.0 * np.pi * xr / wavelength) + 0.25 * np.sin(2.0 * np.pi * yr / (2.0 * wavelength)) + local_phase)

    binary = (lamellar > 0.0).astype(np.float32)
    image = 0.15 + 0.7 * _mean_blur(binary, passes=1)
    image += rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def generate_dataset(
    num_samples: int,
    image_size: int = 64,
    kind: str = "mixed",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate synthetic dataset arrays and metadata."""
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if kind not in {"porous", "precipitate", "eutectic", "mixed"}:
        raise ValueError("kind must be one of: porous, precipitate, eutectic, mixed")

    rng = _ensure_rng(seed)
    images = np.empty((num_samples, image_size, image_size), dtype=np.float32)
    labels = np.empty((num_samples,), dtype=np.int64)

    for i in range(num_samples):
        if kind == "mixed":
            label_name = str(rng.choice(list(LabelMap.keys())))
        else:
            label_name = kind

        img_seed = int(rng.integers(0, 2**31 - 1))
        if label_name == "porous":
            images[i] = generate_porous(size=image_size, seed=img_seed)
        elif label_name == "precipitate":
            images[i] = generate_precipitate(size=image_size, seed=img_seed)
        else:
            images[i] = generate_eutectic(size=image_size, seed=img_seed)
        labels[i] = LabelMap[label_name]

    unique, counts = np.unique(labels, return_counts=True)
    id_to_name = {v: k for k, v in LabelMap.items()}
    label_counts = {id_to_name[int(k)]: int(v) for k, v in zip(unique, counts)}
    metadata = {
        "num_samples": int(num_samples),
        "image_size": int(image_size),
        "kind": kind,
        "seed": seed,
        "label_map": LabelMap,
        "label_counts": label_counts,
        "value_range": [float(images.min()), float(images.max())],
    }
    return images, labels, metadata
