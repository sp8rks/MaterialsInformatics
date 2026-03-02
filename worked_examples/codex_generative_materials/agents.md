# Agent Instructions — microstructure-gen

## Mission

Build a **laptop-friendly generative modeling demo** in the materials science domain.

The project demonstrates three generative model families:

- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)
- Diffusion model (DDPM)

The dataset must be **procedurally generated synthetic microstructure images**, so the project runs fully offline.

This repository is designed for live, in-class coding demonstrations in a Materials Informatics course.

---

# Core Requirements

## Must Run On
- Python 3.10+
- CPU-only by default
- GPU optional (auto-detect CUDA)

## Must Work Offline
- No dataset downloads
- No external APIs
- Dataset generated procedurally

## Performance Targets (Fast Mode)
Using `--fast`, all training runs should complete on CPU in minutes:

- Dataset generation: < 30 seconds
- VAE training: ~1–2 minutes
- GAN training: ~1–2 minutes
- DDPM training: ~2–5 minutes

Full mode may take longer but must remain laptop-feasible.

---

# Environment Management (UV)

This project uses **uv**, not pip, conda, poetry, or requirements.txt.

## Required Files

- `pyproject.toml` (PEP 621 format)
- `.python-version` (set to 3.11 preferred)
- No requirements.txt
- No conda files

## Installation Flow (must work)

```bash
uv venv
source .venv/bin/activate  # or Windows equivalent
uv pip install -e ".[dev]"