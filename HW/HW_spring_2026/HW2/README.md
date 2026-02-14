# HW2: Molecular Featurization, SMILES in ML, and Data Format Structures

## Setup with UV

### First Time Setup

1. **Install UV** (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Navigate to the HW2 folder**:
```bash
cd HW/HW_spring_2026/HW2
```

3. **Create a virtual environment and install dependencies**:
```bash
# This creates the .venv and installs all dependencies in one step
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

4. **Read the assignment instructions**:
Open `HW2_instructions.md` for detailed assignment requirements.

5. **Write your code**:
Create your Python script(s) following the instructions. You can test your code by running:
```bash
python hw2_yourname.py
```

### Quick Start (After Initial Setup)

```bash
cd HW/HW_spring_2026/HW2
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python hw2_yourname.py
```

## Assignment Structure

See `HW2_instructions.md` for complete instructions.

- **Part 1**: Molecular Featurization and Machine Learning
  - Basic molecular descriptors
  - Morgan fingerprints
  - Topological fingerprints
  - Clustering (KMeans) and dimensionality reduction (PCA, UMAP)
  - ML models: Ridge, Random Forest, SVR, Ensemble
  - Molecular similarity using Tanimoto

- **Part 2**: Data Format Structures
  - Convert SMILES to 2D molecular structure graphs
  - Generate 3D molecular models
  - Save to file formats (SDF, MOL, PDB)
  - Create graph data structures
  - Discussion on pros/cons of each format

## Dataset

- `smiles_tg.csv`: SMILES strings with glass transition temperature (Tg) values

## Submission

Submit your Python code file(s):
- `hw2_yourname.py` (or split into multiple files as needed)
- All generated plots and figures
- Molecule files (SDF, MOL, PDB)
- Any output files requested in the instructions

## Notes on RDKit

RDKit can be tricky to install. UV should handle it automatically, but if you encounter issues:

```bash
# Alternative: Install via conda-forge (if UV fails)
conda install -c conda-forge rdkit
```
