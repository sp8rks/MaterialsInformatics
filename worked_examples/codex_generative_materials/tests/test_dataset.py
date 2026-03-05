import json

import numpy as np

from generate_dataset import build_and_save_dataset
from microgen.data import MicrostructureDataset, load_saved_dataset


def test_build_and_save_dataset_outputs_and_metadata(tmp_path) -> None:
    out_path = tmp_path / "microstructures.npz"
    saved_data_path, metadata_path = build_and_save_dataset(
        out_path=out_path,
        num_samples=24,
        image_size=32,
        kind="mixed",
        seed=5,
    )

    assert saved_data_path.exists()
    assert metadata_path.exists()

    with np.load(saved_data_path) as data:
        assert data["images"].shape == (24, 32, 32)
        assert data["labels"].shape == (24,)
        assert data["images"].dtype == np.float32
        assert data["labels"].dtype == np.int64

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["num_samples"] == 24
    assert metadata["image_size"] == 32
    assert metadata["kind"] == "mixed"
    assert "created_at_utc" in metadata
    assert set(metadata["label_map"].keys()) == {"porous", "precipitate", "eutectic"}


def test_load_saved_dataset_and_torch_wrapper(tmp_path) -> None:
    out_path = tmp_path / "microstructures.npz"
    build_and_save_dataset(
        out_path=out_path,
        num_samples=12,
        image_size=16,
        kind="porous",
        seed=11,
    )

    images, labels, metadata = load_saved_dataset(out_path)
    assert images.shape == (12, 16, 16)
    assert labels.shape == (12,)
    assert metadata["kind"] == "porous"
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0

    ds = MicrostructureDataset(out_path)
    x, y = ds[0]
    assert tuple(x.shape) == (1, 16, 16)
    assert int(y.item()) == 0
