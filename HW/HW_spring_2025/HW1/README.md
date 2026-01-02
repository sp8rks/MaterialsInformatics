# HW1: Materials Data Extraction & Featurization

## Setup with UV

### First Time Setup

1. **Install UV** (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Navigate to the HW1 folder**:
```bash
cd HW/HW_spring_2025/HW1
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

4. **For optional dependencies**:
```bash
# For KnowMat2 (Part 2)
uv pip install -e ".[knowmat2]"

# For NOMAD (Part 4)
uv pip install -e ".[nomad]"

# Or install all optional dependencies
uv pip install -e ".[knowmat2,nomad]"
```

5. **Read the assignment instructions**:
Open `HW1_instructions.md` for detailed assignment requirements.

6. **Write your code**:
Create your Python script(s) following the instructions. You can test your code by running:
```bash
python hw1_yourname.py
```

### Quick Start (After Initial Setup)

```bash
cd HW/HW_spring_2025/HW1
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python hw1_yourname.py
```

## API Keys and Account Setup

### Materials Project API Key (Part 1)

You'll need a Materials Project API key. Get one at:
https://next-gen.materialsproject.org/api

Set it as an environment variable:
```bash
# Linux/macOS
export MP_API_KEY="your_api_key_here"

# Windows
set MP_API_KEY=your_api_key_here
```

### Ollama Setup (Part 2 - Local Testing)

For Part 2, you'll use Ollama for local testing of KnowMat2:

1. **Install Ollama**:
   - Download from https://ollama.ai/
   - Follow installation instructions for your platform

2. **Pull a model** (recommended: llama3.2 or mistral):
   ```bash
   ollama pull llama3.2
   # or
   ollama pull mistral
   ```

3. **Set environment variable for testing**:
   ```bash
   # Linux/macOS
   export USE_OLLAMA=true

   # Windows
   set USE_OLLAMA=true
   ```

4. **For grading, we will use OpenAI API**:
   - Your code must support both Ollama and OpenAI API
   - We will set `USE_OLLAMA=false` and provide `OPENAI_API_KEY` when grading
   - No need to get your own OpenAI API key for testing

### NOMAD Access (Part 4)

For downloading data from NOMAD, no API key is required for public data access.
For uploading data, you can:
- Use NOMAD's staging/test server (recommended for this assignment)
- Create a free account at https://nomad-lab.eu/ if you want to upload to production

### Materials Commons Access (Part 4)

If using Materials Commons instead of NOMAD:
- Create a free account at https://materialscommons.org/
- No special setup required for testing uploads

## Assignment Structure

See `HW1_instructions.md` for complete instructions.

- **Part 1**: Extract data via Materials Project API
- **Part 2**: Extract data from literature using KnowMat2 agent
- **Part 3**: Featurization and Machine Learning
- **Part 4**: Download and upload data using NOMAD or Materials Commons

## Submission

Submit your Python code file(s):
- `hw1_yourname.py` (or split into multiple files as needed)
- All generated plots and figures
- Any output files requested in the instructions

## Papers Included

The following papers are provided for Part 2 (LiFeP2O7 extraction):
- bergerhoff1983.pdf
- bih2009.pdf
- dan47204.pdf
- Hautier_2011_Phosphates.pdf
- jain2011.pdf
- ledain1995.pdf
- riou1990.pdf
- rousse2002.pdf

Note: Not all papers contain LiFeP2O7 data - identifying which ones do is part of the assignment.
