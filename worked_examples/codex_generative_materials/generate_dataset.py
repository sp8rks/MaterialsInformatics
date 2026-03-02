"""CLI to generate synthetic microstructure datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from microgen.synth import generate_dataset as synth_generate_dataset


def build_and_save_dataset(
    out_path: Path,
    num_samples: int,
    image_size: int,
    kind: str,
    seed: int | None,
) -> tuple[Path, Path]:
    """Generate and persist dataset + metadata."""
    images, labels, metadata = synth_generate_dataset(
        num_samples=num_samples,
        image_size=image_size,
        kind=kind,
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=images, labels=labels)

    metadata_path = out_path.with_suffix(".json")
    metadata.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_file": out_path.name,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic microstructure datasets.")
    parser.add_argument("--out", type=Path, default=Path("data/microstructures.npz"))
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--kind", choices=["porous", "precipitate", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path, metadata_path = build_and_save_dataset(
        out_path=args.out,
        num_samples=args.num_samples,
        image_size=args.image_size,
        kind=args.kind,
        seed=args.seed,
    )
    print(f"Saved dataset: {data_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
