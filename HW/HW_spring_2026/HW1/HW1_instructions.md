# HW1: Materials Data Extraction & Featurization (Spring 2026)

## Prerequisites: Environment Setup

Before you begin, please set up your Python environment. All required packages for this assignment are listed in the `pyproject.toml` file.

**Setup Steps:**
1.  Navigate to the `HW1` directory in your terminal.
2.  Create a virtual environment: `uv venv`
3.  Activate the environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).
4.  Install all dependencies:
    ```bash
    uv pip install -e .[nomad]
    ```
This single command will install `pymatgen`, `cbfv`, `nomad-lab`, and all other necessary packages for this assignment.

---

## Overview
**Jupyter Notebook Provided:** A starter Jupyter Notebook, `HW1_spring2026.ipynb`, is provided in the assignment folder. You are encouraged to use it to develop and test your code for each task.

The goal of this assignment is to extract crystalline materials data from multiple sources, clean the data, identify specific information within it, and train simple ML models.

**Assignment Structure:**
- **Part 1:** Extract data via Materials Project API (~1 hour)
- **Part 2:** Use featurizers and run simple machine learning models (~1 hour)
- **Part 3:** Extract data from NOMAD repository (~1 hour)

**Total Estimated Time:** ~3 hours

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

### Task 1.2: Query Be-Containing Materials
Let's explore materials containing Be (Beryllium) or Mg (Magnesium). Create a search that finds all non-unary beryllium- or magnesium-containing materials with:
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

## Part 2: Featurization and Machine Learning

### Task 2.1: Import CBFV
Import a featurizer that has been developed by our group: CBFV (Composition-Based Feature Vector).

There are a few different featurizers that can be used:
- oliynyk (default)
- onehot
- mat2vec

**Deliverables:**
- Import CBFV library
- Verify it's working correctly

---

### Task 2.2: Prepare Datasets
Let's make 3 copies of our dataset and use the oliynyk, onehot, and mat2vec featurizers to create features.

**Requirements:**
- Call each dataset df_oli, df_onehot, df_mat2vec
- Convert the band_gap column name to 'target' for each dataset

**Deliverables:**
- Create three copies of the dataset
- Rename the target column appropriately

---

### Task 2.3: Create Features
Use CBFV to create features for each of the 3 datasets.

**Deliverables:**
- Apply oliynyk featurizer to df_oli
- Apply onehot featurizer to df_onehot
- Apply mat2vec featurizer to df_mat2vec
- Print the shape of each featurized dataset

---

### Task 2.4: Train and Evaluate Models
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

### Task 2.5: Model Comparison
Which model performed the best? In a few sentences, explain why you think this model performed the best.

**Deliverables:**
- Identify which featurizer produced the best model
- Write 3-5 sentences explaining why you think this model performed the best
- Include this as comments in your code or in a separate text section

---

## Part 3: Data Repository Integration - NOMAD

In this part, you will learn how to interact with the NOMAD open materials science data repository. You'll download data from NOMAD and compare it with data from Materials Project.

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

---

### Task 3.1: Setup NOMAD Access
Set up access to the NOMAD repository.

**Requirements:**
- Install the NOMAD Python client (should be in pyproject.toml)
- Import the necessary libraries
- Familiarize yourself with NOMAD's API structure

**Deliverables:**
- Code to import NOMAD libraries
- Print NOMAD client version or verify connection

---

### Task 3.2: Browse NOMAD Database
Explore the NOMAD repository to understand what data is available.

**Requirements:**
- Search for datasets related to beryllium-containing materials (similar to Part 1)
- Browse available entries and their metadata
- Identify at least 3 different datasets of interest

**Deliverables:**
- Code to query NOMAD for beryllium materials
- Print the number of available entries
- Display metadata for 3 selected entries (entry ID, formula, properties available)

---

### Task 3.3: Download Data from NOMAD
Download crystallographic or computational data from NOMAD for a specific material or set of materials.

**Requirements:**
- Choose one of the following approaches:
  1. Download data for a specific composition (e.g., BeO)
  2. Download a dataset of similar materials (e.g., Be-containing oxides)
- Extract relevant properties (structure, energy, band gap, etc.)
- Store the data in a structured format (DataFrame or dictionary)

**Deliverables:**
- Code to download data from NOMAD
- Save the downloaded data as a CSV or JSON file
- Print summary statistics of the downloaded data (number of entries, property ranges)
- Save as `nomad_downloaded_data.csv` or `nomad_downloaded_data.json`

---

### Task 3.4: Compare Data Repositories
Write a comparison of the different data sources you've used in this assignment.

**Compare:**
1. **Materials Project** (Part 1)
2. **NOMAD** (Part 3)

**Discuss:**
- Data availability and coverage
- Ease of access and API usability
- Data format and standardization
- Metadata quality
- Use cases for each repository
- Advantages and limitations

**Deliverables:**
- Write a comparison paragraph comparing Materials Project and NOMAD
- Include this as multi-line comments in your code or as a separate markdown file
- Create a comparison table with at least 5 comparison criteria
- Save as `data_repository_comparison.md` or include in your code comments

---

### Task 3.5: Data Integration Challenge
Combine data from multiple sources to create a comprehensive dataset.

**⚠️ Note:** This is where you will likely encounter the most challenges! Different repositories use different:
- Formula conventions (e.g., "BeO" vs "Be1O1")
- Property units and naming
- Data structures and formats
- Metadata standards

**Requirements:**
- Attempt to take data from Materials Project (Part 1)
- Attempt to take data from NOMAD (Task 3.3)
- Attempt to take data from JARVIS-DFT from JARVIS-DB (https://github.com/usnistgov/jarvis)
- Try to identify overlapping materials (same composition in all databases)
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

### Part 2: Featurization
- **Mistake:** Not renaming the target column to 'target' before using CBFV
  - **Fix:** CBFV expects a column named 'target', so rename band_gap
- **Mistake:** Forgetting to set `random_state` for reproducibility
  - **Fix:** Always use `random_state=13` as specified in the instructions

### Part 3: Data Integration
- **Mistake:** Expecting all integration tasks to work smoothly
  - **Reality:** This part is SUPPOSED to be challenging - document your struggles!
- **Mistake:** Not documenting errors or challenges encountered
  - **Fix:** Keep notes on what went wrong and why - this is part of the learning objective
- **Mistake:** Giving up when data formats don't match
  - **Fix:** Try to identify WHY they don't match - that's the whole point of this exercise

---

## Grading Rubric

**Part 1 (40 points):**
- Tasks 1.1-1.5: 15 points
- Tasks 1.6-1.12: 25 points

**Part 2 (25 points):**
- Tasks 2.1-2.3: 15 points
- Tasks 2.4-2.5: 10 points

**Part 3 (35 points):**
- Tasks 3.1-3.3 (NOMAD download): 15 points
- Task 3.4 (Repository comparison): 10 points
- Task 3.5 (Data integration): 10 points

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

**File naming:**
- `hw1_yourname.py` - Your main Python code (or separate files for each part)
- `bandgap_distribution.png` - Band gap histogram
- `nomad_downloaded_data.csv` (or .json) - Data downloaded from NOMAD
- `data_repository_comparison.md` - Repository comparison write-up
- `integrated_dataset.csv` - Merged data from multiple sources
- Any additional plots you create

**Deadline:** January 29, 2026 at 23:59