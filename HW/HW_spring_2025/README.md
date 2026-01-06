# MSE 6640 Homework Assignments - Spring 2026

## Using UV Package Manager

All homework assignments use [UV](https://docs.astral.sh/uv/) for fast and reliable Python package management. UV is significantly faster than pip and handles dependency resolution more efficiently.

### Why UV?

- **Fast**: 10-100x faster than pip
- **Reliable**: Better dependency resolution
- **Simple**: Drop-in replacement for pip
- **No conda conflicts**: Works independently of conda environments

### Installing UV

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

### Basic UV Commands

```bash
# Create a virtual environment
uv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install from pyproject.toml
uv pip install -e .

# Install a specific package
uv pip install pandas

# Install from requirements.txt
uv pip install -r requirements.txt

# List installed packages
uv pip list

# Sync environment (install/update all dependencies)
uv pip sync
```

## Homework Overview

### HW1: Materials Data Extraction & Featurization
**Focus**: Extracting crystalline materials data from multiple sources

**Key Components**:
1. Materials Project API extraction
2. Literature extraction using KnowMat2 agent
3. Featurization and ML model training

**Setup**:
```bash
cd HW_spring_2025/HW1
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
uv pip install -e ".[knowmat2]"  # For Part 2
jupyter notebook HW1_spring2025.ipynb
```

### HW2: Molecular Featurization, SMILES in ML, and Data Format Structures
**Focus**: Working with molecular data using RDKit and SMILES strings

**Key Components**:
1. Molecular featurization (basic descriptors, Morgan fingerprints, topological fingerprints)
2. Machine learning models (Ridge, Random Forest, SVR, Ensemble)
3. Data format structures (2D graphs, 3D models, file formats, graph data types)

**Setup**:
```bash
cd HW_spring_2025/HW2
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
jupyter notebook HW2_spring2025_final.ipynb
```

### HW3: Deep Learning - VAE & CrabNet
**Focus**: Generative models and transformer-based architectures

**Key Components**:
1. Variational Autoencoder (VAE) for microstructure images
2. CrabNet transformer model for materials property prediction

**Setup**:
```bash
cd HW_spring_2025/HW3
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
uv pip install git+https://github.com/anthony-wang/CrabNet.git
jupyter notebook HW3_spring2025_final.ipynb
```

## General Workflow

For each homework assignment:

1. **Navigate to the homework folder**:
   ```bash
   cd HW_spring_2025/HWX  # where X is 1, 2, or 3
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

4. **Read the assignment instructions**:
   Open `HWX_instructions.md` for detailed requirements and tasks.

5. **Write your Python code**:
   Create your script(s) following the instructions (e.g., `hw1_yourname.py`).

6. **Test your code**:
   ```bash
   python hwX_yourname.py
   ```

7. **Submit your work**:
   - Python code files
   - Generated plots and figures
   - Any required output files
   - Model weights (for HW3)

## Troubleshooting

### Virtual Environment Issues
If you have issues with virtual environments, remove and recreate:
```bash
rm -rf .venv  # or rmdir /s .venv on Windows
uv venv
```

### Package Installation Failures
Try upgrading UV:
```bash
uv self update
```

### RDKit Installation (HW2)
If RDKit installation fails with UV, you can use conda:
```bash
conda install -c conda-forge rdkit
```

### GPU Support (HW3)
For CUDA support with PyTorch:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2), the default PyTorch installation includes MPS support.

## Additional Resources

- **UV Documentation**: https://docs.astral.sh/uv/
- **Materials Project API**: https://next-gen.materialsproject.org/api
- **KnowMat2**: https://github.com/hasan-sayeed/KnowMat2
- **CrabNet**: https://github.com/anthony-wang/CrabNet
- **Course Materials**: See main repository README

## Need Help?

1. Check the README in each homework folder
2. Review the HW_Summary.md for assignment details
3. Consult UV documentation
4. Ask on the course discussion board
