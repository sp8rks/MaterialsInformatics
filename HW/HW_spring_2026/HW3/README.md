# HW3: Deep Learning - VAE, XRD Sequence Models, and CrabNet

## Setup with UV

### First Time Setup

1. **Install UV** (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Navigate to the HW3 folder**:
```bash
cd HW/HW_spring_2026/HW3
```

3. **Create a virtual environment and install dependencies**:
```bash
# Create virtual environment
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install all dependencies (includes CrabNet)
uv sync
```

4. **Read the assignment instructions**:
Open `HW3_instructions.md` for detailed assignment requirements.

5. **Write your code**:
Create your Python script(s) following the instructions. You can test your code by running:
```bash
python hw3_yourname.py
```

### Quick Start (After Initial Setup)

```bash
cd HW/HW_spring_2026/HW3
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python hw3_yourname.py
```

## Assignment Structure

See `HW3_instructions.md` for complete instructions.

### Part 1: VAE + MLP Surrogate

Build a convolutional VAE for microstructure images, and an MLP surrogate model for structural toughness:
- **VAE** (Tasks 1.1–1.8): Encoder/decoder on 32×32 grayscale images, latent dim 8, generate similar microstructures
- **MLP Surrogate** (Task 1.9): Feedforward network predicting toughness from crossed-barrel geometry parameters (n, θ, r, t)

### Part 2: RRUFF Powder XRD (GAN, RNN, Transformer Decoder)

Build sequence models on powder XRD data:
- GAN to generate synthetic patterns for augmentation
- RNN classifier on mineral labels (use augmented data)
- Decoder-only transformer for autoregressive generation

### Part 3: CrabNet (Transformer Model)

Train CrabNet on two materials property datasets:
- **Shear modulus** (Tasks 3.1–3.6): Train, evaluate, predict on 10 new compositions
- **Heat capacity at 298 K** (Task 3.7): Filter `cp_data_cleaned.csv` to room temperature, train a second CrabNet model, compare performance

## Submission

Submit your Python code file(s):
- `hw3_yourname.py` (or split into multiple files as needed)
- Saved model weights (`vae_model.pth`, CrabNet models)
- All generated plots and figures
- Any output files requested in the instructions

## Data Requirements

- **Microstructure images**: `data/micrographs/`
- **Crossed-barrel simulation**: `data/crossed_barrel_dataset.csv` (n, θ, r, t → toughness)
- **RRUFF powder XRD**: `data/powderXRD/XY_Processed.zip` (manually unzip to `data/powderXRD/XY_Processed/`)
- **Shear modulus dataset**: `data/shearmodulus_aflow.csv` (columns: `formula`, `target`)
- **Heat capacity dataset**: `data/cp_data_cleaned.csv` (columns: `formula`, `T`, `Cp`; filter to T = 298 K for Task 3.7)

## GPU Acceleration (Optional but Recommended)

PyTorch will automatically use CUDA if available:
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

For Apple Silicon (M1/M2):
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## Notes

- Training VAEs can take time. Start with fewer epochs for testing.
- CrabNet (Part 3): shear modulus uses `data/shearmodulus_aflow.csv`; heat capacity uses `data/cp_data_cleaned.csv` filtered to T = 298 K.
- The micrographs folder should be in the HW3 directory before running Part 1.
- RRUFF powder XRD data is in `data/powderXRD/XY_Processed.zip`.
