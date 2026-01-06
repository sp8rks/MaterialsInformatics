# HW1 Modifications Summary: Ollama Integration for KnowMat2

This document summarizes all modifications made to integrate Ollama support into KnowMat2 for HW1.

## Date Modified
2026-01-01

## Objective
Modify KnowMat2 to support both Ollama (local LLM) and OpenAI API, allowing students to:
1. Test and develop locally using Ollama with gpt-oss-20b or gpt-oss-120b models (no API costs)
2. Submit code that can be graded using OpenAI API
3. Run LangChain completely locally without cloud services

---

## Files Modified

### 1. KnowMat2/src/knowmat/config.py

**Purpose**: Add environment configuration for Ollama vs OpenAI selection and run LangChain locally by default.

**Changes Made**:
- Added `USE_OLLAMA` environment variable detection
- Made `OPENAI_API_KEY` optional when using Ollama
- Made `LANGCHAIN_API_KEY` optional (not required for local operation)
- Added `OLLAMA_BASE_URL` configuration with default `http://localhost:11434`
- Disabled LangSmith tracing by default (`LANGCHAIN_TRACING_V2=false`)
- Added print statements to show configuration status

**Key Features**:
```python
# Check if using Ollama or OpenAI
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

if use_ollama:
    # No OpenAI key needed
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    print("KnowMat2 configured to use Ollama at:", os.getenv("OLLAMA_BASE_URL"))
else:
    # Require OpenAI key
    _set_env("OPENAI_API_KEY")

# Disable cloud tracing - run locally
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
```

**Impact**: Students can now switch between Ollama and OpenAI by setting a single environment variable.

---

### 2. KnowMat2/src/knowmat/app_config.py

**Purpose**: Configure default models for Ollama vs OpenAI modes.

**Changes Made**:
- Added `use_ollama` and `ollama_base_url` settings
- Changed default OpenAI models from `gpt-5` to `gpt-4o` (for compatibility)
- Added conditional defaults for all agent models:
  - **Ollama mode**: Uses `gpt-oss-120b` for main tasks, `gpt-oss-20b` for lighter tasks
  - **OpenAI mode**: Uses `gpt-4o` and `gpt-4o-mini`
- Added comprehensive documentation for Ollama model names

**Default Models**:

| Agent Type | Ollama Default | OpenAI Default |
|------------|----------------|----------------|
| Fallback | gpt-oss-120b | gpt-4o |
| Subfield Detection | gpt-oss-20b | gpt-4o-mini |
| Extraction | gpt-oss-120b | gpt-4o |
| Evaluation | gpt-oss-120b | gpt-4o |
| Manager (Validation) | gpt-oss-120b | gpt-4o |
| Flagging | gpt-oss-20b | gpt-4o-mini |

**Key Code**:
```python
use_ollama: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
model_name: str = "gpt-oss-120b" if use_ollama else "gpt-4o"
extraction_model: str = "gpt-oss-120b" if use_ollama else "gpt-4o"
subfield_model: str = "gpt-oss-20b" if use_ollama else "gpt-4o-mini"
```

**Impact**: Appropriate models are automatically selected based on mode, optimized for performance and quality.

---

### 3. KnowMat2/src/knowmat/extractors.py

**Purpose**: Support both ChatOpenAI and ChatOllama for LLM inference.

**Changes Made**:
- Added import for `ChatOllama` from `langchain_ollama`
- Modified `get_llm()` function to return `Union[ChatOpenAI, ChatOllama]`
- Added Ollama instantiation logic:
  ```python
  if settings.use_ollama:
      return ChatOllama(
          model=model,
          base_url=settings.ollama_base_url,
          temperature=settings.temperature,
      )
  ```
- Maintained existing OpenAI logic for backward compatibility

**Key Features**:
- Seamless switching between LLM providers
- All extractors (subfield, extraction, evaluation, manager, flagging) work with both providers
- Same interface regardless of provider (TrustCall compatibility)

**Impact**: KnowMat2 pipeline works identically with both Ollama and OpenAI, allowing transparent switching.

---

### 4. KnowMat2/environment.yml

**Purpose**: Add required dependency for Ollama support.

**Changes Made**:
- Added `langchain-ollama==0.2.2` to pip dependencies

**Before**:
```yaml
- langchain==0.3.26
- langchain-community==0.3.27
- langchain-openai==0.3.27
- langgraph==0.6.10
```

**After**:
```yaml
- langchain==0.3.26
- langchain-community==0.3.27
- langchain-openai==0.3.27
- langchain-ollama==0.2.2  # NEW
- langgraph==0.6.10
```

**Impact**: Students installing from environment.yml will automatically get Ollama support.

---

### 5. HW1_instructions.md

**Purpose**: Update homework instructions with detailed Ollama setup and usage instructions.

**Major Changes**:

#### Task 2.1: Setup Ollama for Local Testing
- **Added**: Step-by-step installation instructions for Windows, Mac, and Linux
- **Added**: Instructions to pull `gpt-oss-20b` and `gpt-oss-120b` models
- **Added**: Verification commands and expected outputs
- **Added**: Server connectivity testing instructions

#### Task 2.2: Setup KnowMat2 with Ollama Support
- **Replaced**: Generic configuration example with detailed setup steps
- **Added**: Step 1 - Install Python dependencies with both pip and conda options
- **Added**: Step 2 - Create `.env` file with Ollama configuration
- **Added**: Step 3 - Test script to verify Ollama + KnowMat2 integration
- **Added**: Step 4 - Instructions for switching between Ollama and OpenAI
- **Added**: Recommended code structure with single-variable switching
- **Added**: Important environment variables documentation

#### Updated Deliverables
- Screenshots of Ollama installation
- Configuration verification output
- Code demonstrating mode switching
- Comments explaining grading setup

**Impact**: Students have comprehensive, step-by-step guidance for setup and usage.

---

## New Files Created

### 1. OLLAMA_SETUP_GUIDE.md

**Purpose**: Comprehensive reference guide for using KnowMat2 with Ollama.

**Contents** (500+ lines):
1. **Overview**: What is Ollama, why use it, which models to use
2. **Prerequisites**: System and software requirements
3. **Part 1: Installing Ollama**: OS-specific installation instructions
4. **Part 2: Installing Required Models**: Model sizes, download instructions, testing
5. **Part 3: Configuring KnowMat2**: Dependency installation, environment setup, verification
6. **Part 4: Running LangChain Locally**: Explanation of LangSmith, local vs cloud operation
7. **Part 5: Using KnowMat2 with Ollama**: Code examples, basic usage, batch processing
8. **Part 6: Switching to OpenAI for Grading**: Code structure, submission checklist
9. **Troubleshooting**: Common issues and solutions
10. **Performance Comparison**: Expected differences between models and modes

**Key Sections**:
- Detailed model comparison table (20B vs 120B vs OpenAI)
- Complete working code examples
- Troubleshooting for 10+ common issues
- Performance comparison table
- Discussion points for homework Task 2.6

**Impact**: Students have a complete reference for all Ollama-related questions and issues.

---

### 2. CHANGES_SUMMARY.md (This Document)

**Purpose**: Document all modifications for instructor and student reference.

**Contents**:
- List of all modified files with detailed changes
- New files created with descriptions
- Usage instructions
- Testing recommendations
- Grading considerations

---

## Usage Instructions

### For Students (Testing with Ollama)

1. **Install UV Package Manager** (if not already installed):
   ```bash
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Ollama and Models**:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull gpt-oss-20b
   ollama pull gpt-oss-120b
   ```

3. **Setup KnowMat2 with UV**:
   ```bash
   cd HW1/KnowMat2

   # Create virtual environment
   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

   # Install dependencies
   uv pip install -e .
   uv pip install langchain-ollama
   ```

4. **Create `.env` file**:
   ```bash
   # In HW1/KnowMat2/.env
   USE_OLLAMA=true
   OLLAMA_BASE_URL=http://localhost:11434
   LANGCHAIN_TRACING_V2=false
   ```

5. **Use in Code**:
   ```python
   import os
   os.environ['USE_OLLAMA'] = 'true'

   # Import AFTER setting environment
   from knowmat.orchestrator import run_extraction
   ```

### For Students (Preparing for Grading)

```python
import os

# IMPORTANT: Set to False for grading
USE_OLLAMA = False  # Change this single variable

if USE_OLLAMA:
    os.environ['USE_OLLAMA'] = 'true'
else:
    os.environ['USE_OLLAMA'] = 'false'
    # OpenAI key will be provided by grading environment

# Rest of code...
```

### For Instructors (Grading)

1. **Set Environment**:
   ```bash
   export USE_OLLAMA=false
   export OPENAI_API_KEY=your-grading-key
   ```

2. **Run Student Code**:
   - The code should automatically detect `USE_OLLAMA=false`
   - KnowMat2 will use OpenAI API with the provided key
   - All functionality should work identically to Ollama mode

---

## Testing Performed

### Configuration Tests
- ✓ Environment variable detection (`USE_OLLAMA=true/false`)
- ✓ Ollama base URL configuration
- ✓ Model selection (Ollama vs OpenAI models)
- ✓ LangChain tracing disabled (local operation)

### Functionality Tests
- ✓ KnowMat2 imports work in both modes
- ✓ LLM instantiation (ChatOllama vs ChatOpenAI)
- ✓ All extractors work with both providers
- ✓ Mode switching without code changes (env vars only)

### Integration Tests
- ✓ Ollama server connectivity
- ✓ Model loading (gpt-oss-20b, gpt-oss-120b)
- ✓ Simple query/response with ChatOllama
- ✓ Configuration verification script

---

## Benefits of These Modifications

### For Students
1. **Cost Savings**: No OpenAI API costs during development
2. **Privacy**: All data stays local
3. **Learning**: Experience with both cloud and local LLMs
4. **Flexibility**: Test extensively without budget concerns
5. **Speed**: Faster iteration with local models

### For Instructors
1. **Reduced Support**: Comprehensive documentation reduces questions
2. **Flexible Grading**: Can grade with OpenAI while students use Ollama
3. **Consistency**: Same code works in both modes
4. **Learning Objectives**: Students learn about LLM deployment options

### Technical Benefits
1. **Clean Abstraction**: Mode selection through environment variables
2. **No Code Duplication**: Single codebase for both modes
3. **Backward Compatible**: Existing OpenAI code still works
4. **Well Documented**: Every change is explained

---

## Grading Considerations

### What to Check
1. **Mode Switching**: Does the code support both Ollama and OpenAI?
2. **Environment Setup**: Is there a clear way to switch modes?
3. **Documentation**: Did the student explain how to switch to OpenAI?
4. **Functionality**: Does extraction work correctly with OpenAI API?

### Recommended Grading Setup
```bash
# Create grading environment
export USE_OLLAMA=false
export OPENAI_API_KEY=your-key-here

# Run student code
python hw1_studentname.py
```

### Common Issues to Watch For
- Student didn't set environment variables before imports
- Hardcoded `USE_OLLAMA=true` in submission
- Missing instructions for switching to OpenAI
- Code only works with Ollama, not OpenAI

---

## Troubleshooting Quick Reference

### "Connection refused" - Ollama not running
```bash
# Windows: Open Ollama from Start Menu
# Mac: Open Ollama from Applications
# Linux: sudo systemctl start ollama
```

### "Model not found"
```bash
ollama pull gpt-oss-20b
```

### "Import error: langchain_ollama"
```bash
pip install langchain-ollama
```

### "Environment variable not recognized"
```python
# Make sure to set BEFORE imports
os.environ['USE_OLLAMA'] = 'true'
from knowmat import ...  # Import AFTER
```

---

## Future Enhancements (Optional)

If further improvements are needed:

1. **Add More Models**: Support for other Ollama models (llama3.2, mistral, etc.)
2. **Performance Metrics**: Built-in timing and quality comparisons
3. **Hybrid Mode**: Use Ollama for some agents, OpenAI for others
4. **GUI Configuration**: Streamlit app for easy mode switching
5. **Docker Support**: Containerized Ollama + KnowMat2 setup

---

## Package Manager: UV

**IMPORTANT**: All instructions have been updated to use **UV** instead of conda/pip.

**Why UV?**
- 10-100x faster than pip
- Deterministic dependency resolution
- Modern tool built with Rust
- Recommended for all course assignments

**Installation**:
- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Usage**:
```bash
uv venv                    # Create virtual environment
source .venv/bin/activate  # Activate (Windows: .venv\Scripts\activate)
uv pip install -e .        # Install package in editable mode
```

---

## Summary

All required modifications have been completed:

✅ **KnowMat2 Code Modified**:
- config.py: Environment-based mode selection, local LangChain
- app_config.py: Model defaults for both modes
- extractors.py: Support for ChatOllama and ChatOpenAI
- environment.yml: Added langchain-ollama dependency

✅ **Documentation Created**:
- HW1_instructions.md: Updated with detailed setup steps
- OLLAMA_SETUP_GUIDE.md: Comprehensive reference guide (500+ lines)
- CHANGES_SUMMARY.md: This document

✅ **Testing Completed**:
- Configuration switching works
- Both modes functional
- Local LangChain operation verified

✅ **Student Experience**:
- Clear setup instructions
- Easy mode switching (single variable)
- Comprehensive troubleshooting guide
- Example code provided

✅ **Grading Ready**:
- OpenAI API mode works
- Environment variable switching
- Backward compatible
- Well documented

The homework is now ready for students to use Ollama for local testing while maintaining OpenAI API compatibility for grading.
