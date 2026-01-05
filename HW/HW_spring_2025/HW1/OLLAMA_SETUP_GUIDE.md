# KnowMat2 with Ollama Setup Guide

This guide explains how to set up and use KnowMat2 with Ollama for HW1. Using Ollama allows you to run large language models locally without incurring API costs from OpenAI.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Part 1: Installing Ollama](#part-1-installing-ollama)
4. [Part 2: Installing Required Models](#part-2-installing-required-models)
5. [Part 3: Configuring KnowMat2](#part-3-configuring-knowmat2)
6. [Part 4: Running LangChain Locally](#part-4-running-langchain-locally)
7. [Part 5: Using KnowMat2 with Ollama](#part-5-using-knowmat2-with-ollama)
8. [Part 6: Switching to OpenAI for Grading](#part-6-switching-to-openai-for-grading)
9. [Troubleshooting](#troubleshooting)
10. [Performance Comparison](#performance-comparison)

---

## Overview

**What is Ollama?**
Ollama is a tool that allows you to run large language models (LLMs) locally on your computer. Instead of sending requests to OpenAI's servers (which costs money), you can run models directly on your machine.

**Why use Ollama for this homework?**
- **Cost**: No API fees - completely free to use
- **Privacy**: Your data stays on your machine
- **Learning**: Understand how to work with both cloud and local LLMs
- **Flexibility**: Test and iterate without worrying about API costs

**Models we'll use:**
- `gpt-oss-20b`: A 20 billion parameter model - faster, good for testing and development
- `gpt-oss-120b`: A 120 billion parameter model - slower but better quality, recommended for final results

---

## Prerequisites

**System Requirements:**
- **RAM**: Minimum 16GB (32GB+ recommended for gpt-oss-120b)
- **Storage**: At least 100GB free space
- **GPU**: Optional but recommended for faster inference (NVIDIA GPU with 8GB+ VRAM)
- **OS**: Windows 10/11, macOS 11+, or Linux

**Software Requirements:**
- Python 3.9 or higher
- UV package manager (recommended) or pip
- Internet connection (for initial model download)

---

## Part 1: Installing Ollama

### Windows Installation

1. **Download Ollama:**
   - Visit https://ollama.ai/download
   - Click "Download for Windows"
   - Run the installer (OllamaSetup.exe)

2. **Install:**
   - Follow the installation wizard
   - Ollama will install and start automatically
   - You should see an Ollama icon in your system tray

3. **Verify Installation:**
   Open PowerShell or Command Prompt:
   ```powershell
   ollama --version
   ```
   You should see the version number (e.g., `ollama version 0.1.x`)

### macOS Installation

1. **Download Ollama:**
   - Visit https://ollama.ai/download
   - Click "Download for Mac"
   - Open the downloaded DMG file

2. **Install:**
   - Drag Ollama to Applications folder
   - Open Ollama from Applications
   - Ollama will start and appear in your menu bar

3. **Verify Installation:**
   Open Terminal:
   ```bash
   ollama --version
   ```

### Linux Installation

1. **Install via script:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama service:**
   ```bash
   sudo systemctl start ollama
   sudo systemctl enable ollama
   ```

3. **Verify Installation:**
   ```bash
   ollama --version
   ```

---

## Part 2: Installing Required Models

### Understanding Model Sizes

| Model | Size | RAM Required | Speed | Quality | Use Case |
|-------|------|--------------|-------|---------|----------|
| gpt-oss-20b | ~12GB | 16GB+ | Fast | Good | Testing, development |
| gpt-oss-120b | ~70GB | 32GB+ | Slow | Excellent | Final submission |

### Installing the Models

1. **Install gpt-oss-20b (Recommended for testing):**
   ```bash
   ollama pull gpt-oss-20b
   ```
   This will download ~12GB. It may take 10-30 minutes depending on your internet speed.

2. **Install gpt-oss-120b (Recommended for final submission):**
   ```bash
   ollama pull gpt-oss-120b
   ```
   This will download ~70GB. It may take 1-3 hours depending on your internet speed.

   **Note:** If you have limited RAM (<32GB), you may experience slow performance with the 120B model. Start with the 20B model.

3. **Verify Installation:**
   ```bash
   ollama list
   ```
   You should see both models listed:
   ```
   NAME              ID              SIZE    MODIFIED
   gpt-oss-20b      abc123def456    12 GB   2 minutes ago
   gpt-oss-120b     def456ghi789    70 GB   5 minutes ago
   ```

4. **Test a Model:**
   ```bash
   ollama run gpt-oss-20b "What is materials science?"
   ```
   You should see a response from the model. Type `/bye` to exit.

---

## Part 3: Configuring KnowMat2

### Step 0: Install UV Package Manager

UV is a fast, modern Python package manager that we'll use for this course.

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify installation:**
```bash
uv --version
# Should output: uv 0.x.x
```

**Why UV?**
- **Fast**: 10-100x faster than pip
- **Reliable**: Deterministic dependency resolution
- **Simple**: Easy to use and understand
- **Modern**: Built with Rust for performance

---

### Step 1: Install KnowMat2 Dependencies

1. **Navigate to the KnowMat2 directory:**
   ```bash
   cd path/to/HW1/KnowMat2
   ```

2. **Create virtual environment and install dependencies using UV:**
   ```bash
   # Create virtual environment
   uv venv

   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   # Install KnowMat2 in editable mode
   uv pip install -e .

   # Install Ollama integration
   uv pip install langchain-ollama
   ```

3. **Alternative: Using pip (if UV is not available):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   pip install langchain-ollama
   ```

### Step 2: Verify Virtual Environment

Make sure your virtual environment is activated (you should see `(.venv)` in your terminal prompt):
```bash
# Check Python location - should be in .venv
which python  # macOS/Linux
where python  # Windows

# Should show: /path/to/HW1/KnowMat2/.venv/bin/python (or similar)
```

---

### Step 3: Create Environment Configuration

Create a `.env` file in the `KnowMat2` directory:

```bash
# KnowMat2/.env

# Use Ollama instead of OpenAI
USE_OLLAMA=true

# Ollama server URL (default is localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Disable LangSmith tracing (optional, to avoid setup)
LANGCHAIN_TRACING_V2=false

# Optional: Override specific models
# KNOWMAT2_EXTRACTION_MODEL=gpt-oss-120b
# KNOWMAT2_SUBFIELD_MODEL=gpt-oss-20b
# KNOWMAT2_EVALUATION_MODEL=gpt-oss-120b
```

### Step 4: Verify Configuration

Create a test script `test_ollama.py`:

```python
import os

# Set environment before importing KnowMat2
os.environ['USE_OLLAMA'] = 'true'
os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'

# Import KnowMat2 modules
from knowmat.app_config import settings
from knowmat.extractors import get_llm

# Print configuration
print("=" * 50)
print("KnowMat2 Configuration")
print("=" * 50)
print(f"Using Ollama: {settings.use_ollama}")
print(f"Ollama URL: {settings.ollama_base_url}")
print(f"Extraction Model: {settings.extraction_model}")
print(f"Subfield Model: {settings.subfield_model}")
print(f"Evaluation Model: {settings.evaluation_model}")
print("=" * 50)

# Test LLM connection
print("\nTesting LLM connection...")
llm = get_llm("extraction")
print(f"LLM Type: {type(llm).__name__}")

# Try a simple query
response = llm.invoke("Say 'KnowMat2 with Ollama is working!'")
print(f"Response: {response.content}")
print("\n✓ Configuration successful!")
```

Run the test (make sure your virtual environment is activated):
```bash
# Make sure .venv is activated
python test_ollama.py
```

Expected output:
```
KnowMat2 configured to use Ollama at: http://localhost:11434
==================================================
KnowMat2 Configuration
==================================================
Using Ollama: True
Ollama URL: http://localhost:11434
Extraction Model: gpt-oss-120b
Subfield Model: gpt-oss-20b
Evaluation Model: gpt-oss-120b
==================================================

Testing LLM connection...
LLM Type: ChatOllama
Response: KnowMat2 with Ollama is working!

✓ Configuration successful!
```

---

## Part 4: Running LangChain Locally

KnowMat2 uses LangChain as its LLM framework. By default, LangChain can send telemetry data to LangSmith (a cloud tracing service). For this homework, we'll run LangChain **completely locally** without any cloud services.

### What is LangSmith and Why Disable It?

**LangSmith** is a cloud-based monitoring and debugging tool for LangChain applications. While useful for production applications, it:
- Requires creating an account and API key
- Sends your prompts and responses to LangSmith servers
- Is not necessary for this homework assignment

**Running Locally** means:
- No data is sent to external servers
- No API keys needed (except OpenAI if using that mode)
- Everything runs on your machine
- Faster execution (no network overhead)

### Configuration for Local LangChain

The KnowMat2 code has been modified to run LangChain locally by default. Here's what's configured:

**In your `.env` file:**
```bash
# Disable LangSmith cloud tracing - run everything locally
LANGCHAIN_TRACING_V2=false

# Not needed when running locally, but set for compatibility
LANGCHAIN_PROJECT=KnowMat2_Local
```

**In your Python code:**
```python
import os

# Ensure LangChain runs locally
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

# No LANGCHAIN_API_KEY needed!
# The code has been modified to not require this when running locally
```

### Verifying Local Operation

When you run KnowMat2, you should see:
```
KnowMat2 configured to use Ollama at: http://localhost:11434
LangChain tracing: false
```

The `LangChain tracing: false` confirms everything is running locally.

### Optional: Enable LangSmith Tracing

If you want to use LangSmith for debugging (optional, not required):

1. Create an account at https://smith.langchain.com/
2. Get your API key
3. Update your `.env`:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-langsmith-key-here
   LANGCHAIN_PROJECT=KnowMat2
   ```

**For this homework, keeping tracing disabled (local mode) is recommended and sufficient.**

### Benefits of Local Operation

| Aspect | Local Mode | Cloud Tracing |
|--------|------------|---------------|
| **Setup** | Simple | Requires account + API key |
| **Privacy** | All data stays local | Sent to LangSmith servers |
| **Speed** | Faster | Network overhead |
| **Cost** | Free | Free tier, then paid |
| **Required for HW** | Yes | No |

---

## Part 5: Using KnowMat2 with Ollama

### Basic Usage Example

Here's a complete example for extracting data from a PDF:

```python
import os
from pathlib import Path

# ============================================
# CONFIGURATION - Set this at the top
# ============================================
USE_OLLAMA = True  # Change to False to use OpenAI

if USE_OLLAMA:
    os.environ['USE_OLLAMA'] = 'true'
    os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'
    print("Using Ollama for local testing")
else:
    os.environ['USE_OLLAMA'] = 'false'
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-here')
    print("Using OpenAI API")

# Import AFTER setting environment variables
from knowmat.orchestrator import run_extraction
from knowmat.app_config import settings

# ============================================
# EXTRACTION
# ============================================
pdf_path = Path("path/to/your/paper.pdf")
output_dir = Path("output")

print(f"\nExtracting data from: {pdf_path}")
print(f"Using model: {settings.extraction_model}")

# Run extraction
results = run_extraction(
    pdf_path=str(pdf_path),
    output_dir=str(output_dir)
)

print(f"\nExtraction complete! Results saved to: {output_dir}")
```

### Working with Multiple PDFs

```python
import os
from pathlib import Path

# Configuration
os.environ['USE_OLLAMA'] = 'true'

from knowmat.orchestrator import run_extraction

# Find all PDFs in a directory
pdf_dir = Path("HW1/papers")
pdf_files = list(pdf_dir.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files")

# Process each PDF
for pdf_file in pdf_files:
    print(f"\nProcessing: {pdf_file.name}")

    try:
        results = run_extraction(
            pdf_path=str(pdf_file),
            output_dir=f"output/{pdf_file.stem}"
        )
        print(f"✓ Success: {pdf_file.name}")
    except Exception as e:
        print(f"✗ Error processing {pdf_file.name}: {e}")
```

---

## Part 6: Switching to OpenAI for Grading

Your submitted code must support both Ollama (for your testing) and OpenAI (for grading).

### Recommended Code Structure

```python
import os

# ============================================
# CONFIGURATION SECTION
# ============================================
# IMPORTANT: Set USE_OLLAMA = False before submitting for grading
USE_OLLAMA = True  # Set to False for OpenAI API (grading)

if USE_OLLAMA:
    # Ollama configuration (for local testing)
    os.environ['USE_OLLAMA'] = 'true'
    os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'
    print("✓ Using Ollama for local testing")
    print("  Models: gpt-oss-20b / gpt-oss-120b")
else:
    # OpenAI configuration (for grading)
    os.environ['USE_OLLAMA'] = 'false'
    # API key will be set by grading environment
    # If testing locally with OpenAI, uncomment:
    # os.environ['OPENAI_API_KEY'] = 'your-openai-key-here'
    print("✓ Using OpenAI API for grading")
    print("  Models: gpt-4o / gpt-4o-mini")

# ============================================
# IMPORTS - Must come AFTER configuration
# ============================================
from knowmat.orchestrator import run_extraction
from knowmat.app_config import settings

# ... rest of your homework code ...
```

### Before Submitting

**Checklist:**
- [ ] Your code has a single `USE_OLLAMA` variable at the top
- [ ] Switching between Ollama and OpenAI only requires changing one line
- [ ] You've tested with both `USE_OLLAMA = True` and `USE_OLLAMA = False`
- [ ] All imports come AFTER setting environment variables
- [ ] You've included comments explaining how to switch modes
- [ ] Your code works with both Ollama and OpenAI

---

## Troubleshooting

### Problem: "Connection refused" or "Cannot connect to Ollama"

**Solution:**
1. Check if Ollama is running:
   ```bash
   ollama list
   ```
2. If not running, start Ollama:
   - **Windows**: Open Ollama from Start menu
   - **Mac**: Open Ollama from Applications
   - **Linux**: `sudo systemctl start ollama`

### Problem: "Model not found"

**Solution:**
```bash
# List installed models
ollama list

# If model is missing, pull it
ollama pull gpt-oss-20b
```

### Problem: Very slow performance

**Solutions:**
1. Use the smaller model: `gpt-oss-20b` instead of `gpt-oss-120b`
2. Close other applications to free up RAM
3. If you have an NVIDIA GPU, ensure CUDA drivers are installed
4. Check RAM usage: The model needs to fit in RAM

### Problem: "Out of memory" error

**Solutions:**
1. Switch to the smaller model (gpt-oss-20b)
2. Close other applications
3. Restart your computer to clear memory
4. If persistent, you may need more RAM - consider using OpenAI API for testing

### Problem: Import error for langchain_ollama

**Solution:**
```bash
# Make sure virtual environment is activated
uv pip install langchain-ollama
# Or with pip:
pip install langchain-ollama
```

### Problem: Environment variables not being recognized

**Solution:**
Make sure you set environment variables BEFORE importing KnowMat2:
```python
# CORRECT order:
import os
os.environ['USE_OLLAMA'] = 'true'
from knowmat import ...

# WRONG order:
from knowmat import ...
import os
os.environ['USE_OLLAMA'] = 'true'  # Too late!
```

---

## Performance Comparison

### Expected Performance Differences

| Aspect | Ollama (20B) | Ollama (120B) | OpenAI (GPT-4o) |
|--------|--------------|---------------|-----------------|
| **Speed** | Medium | Slow | Fast |
| **Quality** | Good | Excellent | Excellent |
| **Cost** | Free | Free | $$ |
| **Setup Time** | Medium | Long | Instant |
| **RAM Required** | 16GB | 32GB+ | N/A |

### Discussion Points for Task 2.6

When comparing Ollama to OpenAI, consider:

1. **Accuracy**:
   - The 120B model should produce results comparable to GPT-4o
   - The 20B model may miss some details or make more errors
   - Document any differences you observe

2. **Speed**:
   - Ollama will be slower, especially on CPU
   - OpenAI API is usually faster due to cloud infrastructure
   - Time your extractions and compare

3. **Usability**:
   - Ollama: No API key needed, but requires setup
   - OpenAI: Instant to use, but costs money

4. **Reliability**:
   - Both should extract the same core information
   - OpenAI may have better formatting consistency
   - Ollama may require more prompt engineering

---

## Additional Resources

- **Ollama Documentation**: https://github.com/ollama/ollama
- **LangChain Ollama**: https://python.langchain.com/docs/integrations/chat/ollama
- **KnowMat2 Repository**: https://github.com/hasan-sayeed/KnowMat2

---

## Summary

You've now learned how to:
1. ✓ Install and configure Ollama
2. ✓ Download and use GPT-OSS models (20B and 120B)
3. ✓ Configure KnowMat2 to work with Ollama
4. ✓ Switch between Ollama and OpenAI for testing vs. grading
5. ✓ Troubleshoot common issues

**For the homework:**
- Use Ollama (gpt-oss-20b) for initial testing and development
- Use Ollama (gpt-oss-120b) for better quality results
- Ensure your code can switch to OpenAI for grading
- Document your experience comparing Ollama vs OpenAI in Task 2.6

Good luck with your assignment!
