# UV Quick Reference Guide

UV is a fast, modern Python package manager that we use for all course assignments. This guide provides quick reference for common UV commands.

## Installation

### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Verify Installation
```bash
uv --version
```

---

## Essential Commands

### Create Virtual Environment
```bash
# Create a new virtual environment in .venv directory
uv venv

# Create with specific Python version
uv venv --python 3.11
```

### Activate Virtual Environment
```bash
# Windows (PowerShell)
.venv\Scripts\activate

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Install Packages
```bash
# Install a package
uv pip install package-name

# Install specific version
uv pip install package-name==1.2.3

# Install from requirements.txt
uv pip install -r requirements.txt

# Install in editable mode (for development)
uv pip install -e .

# Install multiple packages
uv pip install numpy pandas matplotlib
```

### List Installed Packages
```bash
uv pip list
```

### Show Package Information
```bash
uv pip show package-name
```

### Uninstall Package
```bash
uv pip uninstall package-name
```

### Update Package
```bash
# Uninstall and reinstall with latest version
uv pip uninstall package-name
uv pip install package-name
```

---

## Common Workflows

### Starting a New Project
```bash
# Navigate to project directory
cd my-project

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt

# Or install project in editable mode
uv pip install -e .
```

### For HW1 (KnowMat2 Setup)
```bash
# Navigate to KnowMat2 directory
cd HW1/KnowMat2

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install KnowMat2 and dependencies
uv pip install -e .

# Install Ollama support
uv pip install langchain-ollama

# Verify installation
python -c "from knowmat.app_config import settings; print('Success!')"
```

### Switching Between Projects
```bash
# Deactivate current environment
deactivate

# Navigate to new project
cd /path/to/other-project

# Activate that project's environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

---

## Troubleshooting

### "uv: command not found"

**Solution**: UV is not in your PATH.

**Windows**: Restart your terminal after installation.

**macOS/Linux**: Add to PATH:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
# Add this line to ~/.bashrc or ~/.zshrc to make permanent
```

### Virtual environment not activating

**Solution**: Make sure you're using the correct activation command for your OS:
- **Windows PowerShell**: `.venv\Scripts\activate`
- **Windows CMD**: `.venv\Scripts\activate.bat`
- **macOS/Linux**: `source .venv/bin/activate`

### "No such file or directory: .venv"

**Solution**: You need to create the virtual environment first:
```bash
uv venv
```

### Packages not found after installation

**Solution**: Make sure your virtual environment is activated. You should see `(.venv)` in your terminal prompt.

### "Permission denied" error

**Windows**: Run PowerShell as Administrator.

**macOS/Linux**: Don't use sudo with UV. If you get permission errors:
```bash
# Fix ownership of .cargo directory
sudo chown -R $USER:$USER ~/.cargo
```

---

## UV vs pip vs conda

| Feature | UV | pip | conda |
|---------|-----|-----|-------|
| **Speed** | ⚡ Very Fast (10-100x faster) | Medium | Slow |
| **Dependency Resolution** | ✅ Deterministic | ❌ Can have conflicts | ✅ Good |
| **Virtual Environments** | ✅ Built-in (`uv venv`) | ⚠️ Separate tool (`venv`) | ✅ Built-in |
| **Binary Packages** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Disk Space** | 💚 Efficient | 💛 Medium | ❌ Large |
| **Learning Curve** | 💚 Easy | 💚 Easy | 💛 Medium |

---

## Why UV for This Course?

1. **Speed**: UV is 10-100x faster than pip, saving you time during development
2. **Reliability**: Deterministic dependency resolution means fewer conflicts
3. **Modern**: Built with Rust for performance and reliability
4. **Simple**: Easy-to-use commands similar to pip
5. **Recommended**: Industry-standard tool for modern Python development

---

## Additional Resources

- **UV Documentation**: https://github.com/astral-sh/uv
- **UV Installation Guide**: https://astral.sh/uv
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html

---

## Quick Command Cheat Sheet

```bash
# Installation
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Virtual Environment
uv venv                              # Create
source .venv/bin/activate            # Activate (macOS/Linux)
.venv\Scripts\activate               # Activate (Windows)
deactivate                           # Deactivate

# Package Management
uv pip install package-name          # Install
uv pip install -e .                  # Install editable
uv pip install -r requirements.txt   # Install from file
uv pip list                          # List installed
uv pip uninstall package-name        # Uninstall

# Common Combinations
uv venv && source .venv/bin/activate && uv pip install -e .  # Full setup (Unix)
uv venv; .venv\Scripts\activate; uv pip install -e .         # Full setup (Windows)
```

---

## For HW1 Specifically

### Complete Setup Commands

**macOS/Linux:**
```bash
cd HW1/KnowMat2
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install langchain-ollama
```

**Windows (PowerShell):**
```powershell
cd HW1\KnowMat2
uv venv
.venv\Scripts\activate
uv pip install -e .
uv pip install langchain-ollama
```

### Verify Everything Works
```bash
python -c "import os; os.environ['USE_OLLAMA']='true'; from knowmat.app_config import settings; print(f'✓ Ollama mode: {settings.use_ollama}')"
```

---

**Remember**: Always activate your virtual environment before running Python code or installing packages!
