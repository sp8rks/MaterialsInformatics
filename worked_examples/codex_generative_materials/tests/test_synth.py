import numpy as np

from microgen.synth import generate_eutectic, generate_porous, generate_precipitate


def test_generate_porous_shape_and_range() -> None:
    image = generate_porous(size=64, porosity=0.4, seed=123)
    assert image.shape == (64, 64)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_generate_precipitate_shape_and_range() -> None:
    image = generate_precipitate(size=32, count_range=(8, 12), seed=123)
    assert image.shape == (32, 32)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_generate_eutectic_shape_and_range() -> None:
    image = generate_eutectic(size=40, frequency_range=(4.0, 8.0), seed=123)
    assert image.shape == (40, 40)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_generators_are_deterministic_with_seed() -> None:
    a = generate_porous(size=48, seed=77)
    b = generate_porous(size=48, seed=77)
    assert np.allclose(a, b)
    c = generate_eutectic(size=48, seed=77)
    d = generate_eutectic(size=48, seed=77)
    assert np.allclose(c, d)
