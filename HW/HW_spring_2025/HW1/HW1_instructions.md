# HW1: Materials Data Extraction & Featurization (Spring 2026)

## Prerequisites: Environment Setup

Before you begin, please set up your Python environment. All required packages for this assignment are listed in the `pyproject.toml` file.

**Setup Steps:**
1.  Navigate to the `HW1` directory in your terminal.
2.  Create a virtual environment: `uv venv`
3.  Activate the environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).
4.  Install all dependencies, including optional ones for Part 2 and Part 4:
    ```bash
    uv pip install -e .[knowmat2,nomad]
    ```
This single command will install `pymatgen`, `cbfv`, `knowmat2`, `nomad-lab`, and all other necessary packages for this assignment.

---

## Overview
**Jupyter Notebook Provided:** A starter Jupyter Notebook, `HW1_spring2026.ipynb`, is provided in the assignment folder. You are encouraged to use it to develop and test your code for each task.

The goal of this assignment is to extract crystalline materials data from multiple sources, clean the data, identify specific information within it, and train simple ML models.

**Assignment Structure:**
- **Part 1:** Extract data via Materials Project API (~2-3 hours)
- **Part 2:** Extract data from literature via KnowMat2 agent (~2-3 hours)
- **Part 3:** Use featurizers and run simple machine learning models (~1.5-2 hours)
- **Part 4:** Download and upload data using NOMAD or Materials Commons (~2-3 hours)

**Total Estimated Time:** 8-11 hours

**Submission:** Submit your Python code file(s) that complete all tasks below.

---

## Part 1: Materials Project API Data Extraction

### Task 1.1: Setup and Import Libraries
Import appropriate libraries for the new pymatgen API and initialize the API key via MPRester.

**Requirements:**
- Import necessary libraries (pymatgen, mp_api, pandas, numpy, matplotlib, etc.)
- Initialize Materials Project API with your API key
- Verify connection to the API

---

### Task 1.2: Query Li-Containing Materials
Let's explore energy storage materials containing Li. Create a search that finds all quaternary lithium-containing materials with:
- Band gap between 0.5 - 3.0 eV
- Density between 2.0 - 4.0 g/cm³

**Note on API Data:** The number of materials returned by an API query can change over time as the database is updated. If your count doesn't exactly match a specific number, don't worry. As long as your query is formulated correctly, you can proceed with the data you have.

**Deliverables:**
- Code to query the Materials Project
- Print the total number of materials found

---

### Task 1.3: Identify Highest Band Gap Material
Identify which composition within this dataset has the highest band gap.

**Deliverables:**
- Print the formula and the band gap value of the material with the highest band gap

---

### Task 1.4: Check Material Stability
Is the optimal material (from Task 1.3) stable?

**Hint:** You might get a few materials with the same formula. Identify only the one of interest using other properties we identified so far.

**Deliverables:**
- Code to check stability
- Print whether the material is stable

---

### Task 1.5: Visualize Band Gap Distribution
Using matplotlib, create a histogram of the band gap values for all materials in the dataset.

**Requirements:**
- Set bins=50
- Set density=True
- Include axis labels and title
- Save the plot as `bandgap_distribution.png`

---

### Task 1.6: Create DataFrame
Create a dataframe of the dataset containing a column for the formula and band gap.

**Deliverables:**
- Create the dataframe
- Print the first 10 rows
- Print the shape of the dataframe

---

### Task 1.7: Copy and Explore Data
Once the dataframe is created, make a copy of it. We will begin identifying features within this dataset.

**Deliverables:**
- Create a copy of the dataframe
- Store it in a new variable

---

### Task 1.8: Identify Duplicates
Inspect the dataframe for duplicated rows (meaning these rows will have identical entries in both the formula and band_gap columns).

**Deliverables:**
- Report what formulas have duplicate entries
- Print the number of duplicate rows

---

### Task 1.9: Remove Duplicate Rows
Print the shape of the dataframe before and after removing the duplicate rows.

**Deliverables:**
- Print shape before removing duplicates
- Remove duplicate rows
- Print shape after removing duplicates
- Verify you removed the correct number of rows

---

### Task 1.10: Handle Formula Duplicates
Next we want to identify all unique formulas within the dataset. These formulas that are duplicated within the formula column are most likely measured at different conditions, leading to different band gap values. For now we will ignore this fact and proceed to condensing these formulas. Let's take the average value of all duplicate formulas and replace the duplicate formulas with the average value.

**Example:** If SiO₂ has 3 different entries, replace the 3 values with the mean value, so now we will only see SiO₂ once, and the associated band gap will be the average of the 3 values.

**Deliverables:**
- Code to calculate mean values for duplicate formulas
- Replace duplicate entries with mean values

---

### Task 1.11: Validate Averaging
You can now validate that you did this correctly by filtering the dataframe for a single composition that is duplicated and making sure that you replaced this value with the average of the duplicates.

**Deliverables:**
- Filter the dataframe for a composition that has duplicates
- Verify that the value is the mean value
- Print the verification results

---

### Task 1.12: Drop Duplicates
Once you have validated that you have done this correctly, you can now proceed to the next step. Let's drop the duplicated rows in the dataframe. Make sure to print the shape of the dataframe before and after dropping the duplicates to validate that you have done this correctly.

**Deliverables:**
- Print shape before drop
- Drop duplicated rows
- Print shape after drop
- Verify the shapes match the expected values

---

## Part 2: Extract Data from Literature using KnowMat2 Agent

This part of the assignment requires using the `KnowMat2` agent with a local LLM (Ollama). Follow the setup steps below carefully.

### Step-by-Step Setup Guide

#### Step 1: Install Ollama (Local LLM Server)

Ollama runs language models locally on your machine, allowing KnowMat2 to extract data from PDFs.

1. **Download Ollama:**
   - Go to [https://ollama.ai/download](https://ollama.ai/download)
   - Download the installer for your operating system (Windows, Mac, or Linux)
   - Run the installer and follow the installation prompts

2. **Verify Ollama Installation:**
   - Open a **new** terminal window (important: close and reopen to refresh the PATH)
   - Run: `ollama --version`
   - You should see a version number (if not, see Troubleshooting below)

3. **Pull the Required Model:**
   - In your terminal, run:
     ```bash
     ollama pull gpt-oss-20b
     ```
   - This downloads a ~12GB model - it may take 10-30 minutes depending on your internet speed
   - **Optional:** For better quality (but requires 70+ GB of disk space):
     ```bash
     ollama pull gpt-oss-120b
     ```

4. **Verify Ollama is Running:**
   - Check installed models: `ollama list`
   - You should see `gpt-oss-20b` in the list
   - Test the server is running:
     - **Mac/Linux:** `curl http://localhost:11434/api/tags`
     - **Windows PowerShell:** `Test-NetConnection -ComputerName localhost -Port 11434`
   - If you get a connection error, start the Ollama application from your applications menu

#### Step 2: Install KnowMat2 Python Package

1. **Activate your virtual environment** (if not already active):
   - **Mac/Linux:** `source .venv/bin/activate`
   - **Windows:** `.venv\Scripts\activate`
   - You should see `(.venv)` in your terminal prompt

2. **Install KnowMat2 and dependencies:**
   ```bash
   uv pip install -e .[knowmat2,nomad]
   ```
   This installs KnowMat2 and its dependencies (this was in the Prerequisites section)

3. **Install the Ollama integration:**
   ```bash
   uv pip install langchain-ollama
   ```
   This allows KnowMat2 to communicate with your local Ollama server

#### Step 3: Configure KnowMat2 to Use Ollama

1. **Navigate to the KnowMat2 directory:**
   ```bash
   cd HW1/KnowMat2
   ```

2. **Create the `.env` file:**
   - Inside the `HW1/KnowMat2` directory, create a file named **exactly** `.env` (note the dot at the beginning)
   - **Important:** The file must be in `HW1/KnowMat2/`, NOT in `HW1/` or anywhere else

3. **Add the following configuration to `.env`:**
   ```
   USE_OLLAMA=true
   OLLAMA_BASE_URL=http://localhost:11434
   LANGCHAIN_TRACING_V2=false
   # For grading, you will change USE_OLLAMA to false and add your OPENAI_API_KEY
   # OPENAI_API_KEY="your-key-here"
   ```

### Troubleshooting KnowMat2 & Ollama

**Q: `ollama` command not found.**
- **A:** Close and reopen your terminal after installing Ollama to refresh your system's PATH. If still not working, restart your computer.

**Q: `Connection refused` error when running my script.**
- **A:** Your Ollama server isn't running.
  - **Fix:** Start the Ollama application from your applications menu
  - Check that you can run `ollama list` successfully
  - Verify the model is installed with `ollama list` (should show `gpt-oss-20b`)

**Q: `ModuleNotFoundError: No module named 'knowmat'`**
- **A:** Your virtual environment is not active.
  - Look for `(.venv)` in your terminal prompt
  - If missing, run `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows) from the `HW1` directory

**Q: My code is trying to use OpenAI, not Ollama.**
- **A:** Your `.env` file is in the wrong place or configured incorrectly.
  - **Check:** The `.env` file MUST be in `HW1/KnowMat2/`, not `HW1/`
  - **Check:** The file is named `.env` (with a dot), not `env.txt` or `.env.txt`
  - **Check:** `USE_OLLAMA=true` is set correctly (no quotes, true is lowercase)

**Q: Ollama is using too much RAM/CPU.**
- **A:** This is normal - LLMs are resource-intensive. Close other applications while running KnowMat2. The model runs locally for privacy and to avoid API costs.

---

### Task 2.1: Verify Your Setup
After following the setup guide, run the script below to verify that KnowMat2 is configured correctly. This script should run without errors and confirm that it's using Ollama.

**Deliverables:**
- A screenshot showing the successful output of the verification script below.

```python
# Verification Script
# Make sure you are running this script from the HW1 directory,
# or that the KnowMat2 package is otherwise in your Python path.
from knowmat.app_config import settings
from knowmat.extractors import get_llm

print("--- Verifying KnowMat2 Configuration ---")
print(f"Using Ollama: {settings.use_ollama}")
print(f"Extraction model: {settings.extraction_model}")
print(f"Ollama URL: {settings.ollama_base_url}")

if settings.use_ollama:
    try:
        llm = get_llm("extraction")
        print(f"LLM type: {type(llm).__name__}")
        print("\nAttempting a test query...")
        response = llm.invoke("Why is the sky blue?")
        print("Test query successful!")
        print("KnowMat2 is configured correctly for Ollama!")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred: {e}")
        print("Please check that your Ollama server is running and you have pulled a model (e.g., 'ollama pull gpt-oss-20b').")
else:
    print("\nKnowMat2 is configured to use OpenAI.")
    print("Ensure your OPENAI_API_KEY is set in the .env file.")

print("--- Verification Complete ---")
```

---

### Task 2.2: Identify Relevant Papers
List all PDF files in the HW1 folder and use KnowMat2 to identify which papers contain information about LiFeP₂O₇.

**Hint:** The folder contains 9 PDFs - not all will have LiFeP₂O₇ data.

**Deliverables:**
- Code to list all PDF files.
- Code that uses the KnowMat2 `run_extraction` orchestrator on the PDF paths.
- A printout of which papers were identified as containing the relevant data.

---

### Task 2.3: Extract Crystallographic Data
Extract the following crystallographic information for LiFeP₂O₇ from the papers:
- Space group
- Lattice parameters (a, b, c, alpha, beta, gamma)
- Atomic coordinates
- Volume

**Deliverables:**
- Create a dataframe or dictionary to store this information from each paper source
- Print the extracted data

---

### Task 2.4: Compare with Materials Project
Now let's compare the literature data with Materials Project API data.

**Requirements:**
- Use the Materials Project API to search for LiFeP₂O₇
- Extract the same crystallographic information (space group, lattice parameters, atomic coordinates)
- Compare the results from the literature (KnowMat2) with the Materials Project API

**Deliverables:**
- Code to query Materials Project for LiFeP₂O₇
- Code to compare literature data with Materials Project data
- Print the comparison results

---

### Task 2.5: Discussion - Literature vs Database
In a few sentences, discuss:
1. Which papers contained useful LiFeP₂O₇ crystallographic data?
2. How did the literature data compare to Materials Project?
3. What are the advantages and limitations of extracting data from literature vs. using databases like Materials Project?

**Deliverables:**
- Write your discussion as comments in your code or in a separate text section
- Minimum 3-5 sentences per question

---

## Part 3: Featurization and Machine Learning

### Task 3.1: Import CBFV
Import a featurizer that has been developed by our group: CBFV (Composition-Based Feature Vector).

There are a few different featurizers that can be used:
- oliynyk (default)
- onehot
- mat2vec

**Deliverables:**
- Import CBFV library
- Verify it's working correctly

---

### Task 3.2: Prepare Datasets
Let's make 3 copies of our dataset and use the oliynyk, onehot, and mat2vec featurizers to create features.

**Requirements:**
- Call each dataset df_oli, df_onehot, df_mat2vec
- Convert the band_gap column name to 'target' for each dataset

**Deliverables:**
- Create three copies of the dataset
- Rename the target column appropriately

---

### Task 3.3: Create Features
Use CBFV to create features for each of the 3 datasets.

**Deliverables:**
- Apply oliynyk featurizer to df_oli
- Apply onehot featurizer to df_onehot
- Apply mat2vec featurizer to df_mat2vec
- Print the shape of each featurized dataset

---

### Task 3.4: Train and Evaluate Models
Once you have created the features, you can now proceed to the next step.

**Requirements:**
- Split the data into training and testing sets using train_test_split from sklearn
- Use 80% training and 20% testing
- Set random_state to 13 so that we can reproduce the results
- Do this for all 3 featurized datasets
- Train 3 random forest regressors (one for each featurized dataset)
- Evaluate each model on the test set
- Report the mean squared error for each model

**Deliverables:**
- Code to split data for all three datasets
- Code to train three Random Forest models
- Print MSE for each model (oliynyk, onehot, mat2vec)

---

### Task 3.5: Model Comparison
Which model performed the best? In a few sentences, explain why you think this model performed the best.

**Deliverables:**
- Identify which featurizer produced the best model
- Write 3-5 sentences explaining why you think this model performed the best
- Include this as comments in your code or in a separate text section

---

## Part 4: Data Repository Integration - NOMAD and Materials Commons

In this part, you will learn how to interact with open materials science data repositories. You'll download data from NOMAD and upload your processed data to either NOMAD or Materials Commons.

**⚠️ IMPORTANT - Learning Objective:**
This part is designed to expose you to the real-world challenges of integrating data from multiple sources. **You are NOT expected to successfully complete all integration tasks.** Many students will encounter difficulties with data format incompatibilities, API limitations, or conflicting metadata. This is intentional!

**Your goal is to:**
1. Attempt each task and document your process
2. Identify and articulate the challenges you encounter
3. Reflect on why data integration is difficult in materials science
4. **Partial completion with thoughtful reflection is acceptable and will receive full credit**

If you encounter errors or roadblocks, document them clearly and explain what you learned about the limitations of different data sources.

### Background
- **NOMAD** (Novel Materials Discovery): A large-scale repository and archive for materials science data
  - Website: https://nomad-lab.eu/
  - Documentation: https://nomad-lab.eu/prod/v1/docs/
- **Materials Commons**: A platform for the materials community to share and collaborate on data
  - Website: https://materialscommons.org/
  - Documentation: https://materials.org/

---

### Task 4.1: Setup NOMAD Access
Set up access to the NOMAD repository.

**Requirements:**
- Install the NOMAD Python client (should be in pyproject.toml)
- Import the necessary libraries
- Familiarize yourself with NOMAD's API structure

**Deliverables:**
- Code to import NOMAD libraries
- Print NOMAD client version or verify connection

---

### Task 4.2: Browse NOMAD Database
Explore the NOMAD repository to understand what data is available.

**Requirements:**
- Search for datasets related to lithium-containing materials (similar to Part 1)
- Browse available entries and their metadata
- Identify at least 3 different datasets of interest

**Deliverables:**
- Code to query NOMAD for lithium materials
- Print the number of available entries
- Display metadata for 3 selected entries (entry ID, formula, properties available)

---

### Task 4.3: Download Data from NOMAD
Download crystallographic or computational data from NOMAD for a specific material or set of materials.

**Requirements:**
- Choose one of the following approaches:
  1. Download data for a specific composition (e.g., LiFePO₄)
  2. Download a dataset of similar materials (e.g., Li-containing phosphates)
- Extract relevant properties (structure, energy, band gap, etc.)
- Store the data in a structured format (DataFrame or dictionary)

**Deliverables:**
- Code to download data from NOMAD
- Save the downloaded data as a CSV or JSON file
- Print summary statistics of the downloaded data (number of entries, property ranges)
- Save as `nomad_downloaded_data.csv` or `nomad_downloaded_data.json`

---

### Task 4.4: Prepare Data for Upload
Prepare a dataset from your previous work for upload to a repository.

**Requirements:**
- Use the processed data from Part 1 or Part 3 (the cleaned Li-containing materials dataset with predictions)
- Format the data according to repository requirements
- Include relevant metadata:
  - Data source (Materials Project)
  - Processing steps performed
  - ML model information (if including predictions)
  - Date created
  - Your name/affiliation

**Deliverables:**
- Code to format your data for upload
- Create metadata dictionary or file
- Save formatted dataset as `upload_ready_data.csv`

---

### Task 4.5: Upload to NOMAD or Materials Commons
Upload your prepared dataset to either NOMAD (staging/test server) or Materials Commons.

**Important:** Use test/staging servers only, not production repositories!

**Option A: NOMAD Upload**
- Use NOMAD's staging server for testing uploads
- Create an entry with your processed data
- Include proper metadata and documentation

**Option B: Materials Commons Upload**
- Create a test project in Materials Commons
- Upload your dataset
- Add description and metadata

**Note:** If you cannot access upload functionality, prepare the upload package (data + metadata) and document the steps you would take to upload it.

**Deliverables:**
- Code to upload data (or prepare upload package)
- Screenshot or documentation of the upload process
- Entry ID or project link (if successful upload)
- OR detailed documentation of upload preparation steps
- Save documentation as `upload_documentation.txt`

---

### Task 4.6: Compare Data Repositories
Write a comparison of the different data sources you've used in this assignment.

**Compare:**
1. **Materials Project** (Part 1)
2. **Literature/Papers via KnowMat2** (Part 2)
3. **NOMAD** (Part 4)
4. **Materials Commons** (Part 4)

**Discuss:**
- Data availability and coverage
- Ease of access and API usability
- Data format and standardization
- Metadata quality
- Use cases for each repository
- Advantages and limitations

**Deliverables:**
- Write a comparison paragraph for a. **database vs literature data extraction** (MP vs KnowMat2), b. **research data management (RDM) platforms** (NOMAD vs Materials Commons)
- Include this as multi-line comments in your code or as a separate markdown file
- Create a comparison table with at least 5 comparison criteria
- Save as `data_repository_comparison.md` or include in your code comments

---

### Task 4.7: Data Integration Challenge
Combine data from multiple sources to create a comprehensive dataset.

**⚠️ Note:** This is where you will likely encounter the most challenges! Different repositories use different:
- Formula conventions (e.g., "LiFePO4" vs "Li1Fe1P1O4")
- Property units and naming
- Data structures and formats
- Metadata standards

**Requirements:**
- Attempt to take data from Materials Project (Part 1)
- Attempt to take data from NOMAD (Part 4.3)
- Try to identify overlapping materials (same composition in both databases)
- Attempt to compare properties between the two sources (e.g., band gap, formation energy)
- Try to create a unified dataset that includes data from both sources

**Deliverables:**
- Code showing your attempted merge process (even if it doesn't fully work)
- Documentation of challenges encountered:
  - What data format issues did you face?
  - How did you try to resolve them?
  - What property mismatches did you find?
- Print whatever statistics you were able to obtain:
  - Number of materials in Materials Project data
  - Number of materials in NOMAD data
  - Number of overlapping materials (if any)
  - Examples of property comparisons (if successful)
- If successful: Create a visualization comparing properties from both sources
- If partially successful: Save whatever merged data you obtained as `integrated_dataset.csv`
- **If unsuccessful: Write a reflection on what you learned about data integration challenges**

---

## Common Mistakes and Tips

### Part 1: Materials Project API
- **Mistake:** Not checking if the API key is properly set
  - **Fix:** Print a small test query first to verify your API connection
- **Mistake:** Forgetting to handle duplicate formulas before dropping duplicate rows
  - **Fix:** Follow Tasks 1.10-1.12 in order - average first, then drop
- **Mistake:** Using `drop_duplicates()` without specifying which columns
  - **Fix:** Be explicit: `df.drop_duplicates(subset=['formula', 'band_gap'])`

### Part 2: KnowMat2
- **Mistake:** Running Python scripts outside the activated virtual environment
  - **Fix:** Always check for `(.venv)` in your terminal prompt before running scripts
- **Mistake:** Placing the `.env` file in the wrong directory
  - **Fix:** The `.env` file MUST be in `HW1/KnowMat2/`, not in `HW1/` or elsewhere
- **Mistake:** Trying to use OpenAI when Ollama is configured (or vice versa)
  - **Fix:** Check `USE_OLLAMA` setting in your `.env` file

### Part 3: Featurization
- **Mistake:** Not renaming the target column to 'target' before using CBFV
  - **Fix:** CBFV expects a column named 'target', so rename band_gap
- **Mistake:** Forgetting to set `random_state` for reproducibility
  - **Fix:** Always use `random_state=13` as specified in the instructions

### Part 4: Data Integration
- **Mistake:** Expecting all integration tasks to work smoothly
  - **Reality:** This part is SUPPOSED to be challenging - document your struggles!
- **Mistake:** Not documenting errors or challenges encountered
  - **Fix:** Keep notes on what went wrong and why - this is part of the learning objective
- **Mistake:** Giving up when data formats don't match
  - **Fix:** Try to identify WHY they don't match - that's the whole point of this exercise

---

## Grading Rubric

**Part 1 (25 points):**
- Tasks 1.1-1.5: 10 points
- Tasks 1.6-1.12: 15 points

**Part 2 (20 points):**
- Tasks 2.1-2.3: 10 points
- Tasks 2.4-2.5: 10 points

**Part 3 (20 points):**
- Tasks 3.1-3.3: 10 points
- Tasks 3.4-3.5: 10 points

**Part 4 (35 points):**
- Tasks 4.1-4.3 (NOMAD download): 10 points
- Tasks 4.4-4.5 (Data upload): 10 points
- Task 4.6 (Repository comparison): 10 points
- Task 4.7 (Data integration): 5 points

**Total: 100 points**

---

## Submission Guidelines

**Important Note on File Naming:** Throughout this assignment, you are required to save data and plots to specific filenames (e.g., `bandgap_distribution.png`, `integrated_dataset.csv`). It is critical that you use these exact filenames, as your submission may be graded by scripts that look for these files. Failure to use the correct filenames may result in a loss of points.

**IMPORTANT: Use UV for package management throughout this assignment. See UV_QUICK_REFERENCE.md for commands.**

1. **Code:** Submit a Python file (or multiple files) named `hw1_yourname.py` (or `hw1_part1.py`, `hw1_part2.py`, `hw1_part3.py`)
2. **Output:** Include print statements showing your results for each task
3. **Comments:** Add comments explaining your approach
4. **Figures:** Save any plots as PNG files and include them in your submission
5. **Discussion:** Include your written responses to discussion questions as multi-line comments
6. **Environment:** Your code should work with both Ollama (for testing) and OpenAI (for grading)

**File naming:**
- `hw1_yourname.py` - Your main Python code (or separate files for each part)
- `bandgap_distribution.png` - Band gap histogram
- `nomad_downloaded_data.csv` (or .json) - Data downloaded from NOMAD
- `upload_ready_data.csv` - Formatted data ready for upload
- `upload_documentation.txt` - Upload process documentation
- `data_repository_comparison.md` - Repository comparison write-up
- `integrated_dataset.csv` - Merged data from multiple sources
- `repository_comparison_plot.png` - Visualization comparing repositories
- Any additional plots you create

**Deadline:** January 29, 2026 at 23:59