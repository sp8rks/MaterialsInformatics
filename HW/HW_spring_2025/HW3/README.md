# HW3: Deep Learning - VAE & CrabNet

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
cd HW/HW_spring_2025/HW3
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

# Install all dependencies
uv pip install -e .
```

4. **For CrabNet (Part 2)**:
```bash
# Install CrabNet from GitHub
uv pip install git+https://github.com/anthony-wang/CrabNet.git
```

5. **Read the assignment instructions**:
Open `HW3_instructions.md` for detailed assignment requirements.

6. **Write your code**:
Create your Python script(s) following the instructions. You can test your code by running:
```bash
python hw3_yourname.py
```

### Quick Start (After Initial Setup)

```bash
cd HW/HW_spring_2025/HW3
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python hw3_yourname.py
```

## Assignment Structure

See `HW3_instructions.md` for complete instructions.

### Part 1: Variational Autoencoder (VAE)

Build a convolutional VAE for microstructure images:
- **Architecture**:
  - Encoder: 2 conv layers → latent dim 8
  - Decoder: 2 transposed conv layers
- **Tasks**:
  - Implement VAE loss (reconstruction + KL divergence, β=0.1)
  - Train on microstructure images
  - Generate similar images by sampling latent space

### Part 2: CrabNet (Transformer Model)

Train CrabNet on shear modulus dataset:
- Use CBFV for featurization
- Evaluate on 10% test set
- Predict on 10 new material compositions

## Submission

Submit your Python code file(s):
- `hw3_yourname.py` (or split into multiple files as needed)
- Saved model weights (`vae_model.pth`, CrabNet models)
- All generated plots and figures
- Any output files requested in the instructions

## Data Requirements

- **Microstructure images**: Download the micrographs.zip folder (provided separately)
- **Shear modulus dataset**: Download from Canvas

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
- CrabNet requires the shear modulus dataset from Canvas.
- The micrographs folder should be in the HW3 directory before running Part 1.
