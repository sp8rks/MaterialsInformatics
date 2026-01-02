# HW2: Molecular Featurization, SMILES in ML, and Data Format Structures (Spring 2026)

## Prerequisites: Environment Setup

Before you begin, please set up your Python environment. All required packages for this assignment (like `rdkit`, `umap-learn`, `py3dmol`, etc.) are listed in the `pyproject.toml` file.

**Setup Steps:**
1.  Navigate to the `HW2` directory in your terminal.
2.  Create a virtual environment: `uv venv`
3.  Activate the environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).
4.  Install all dependencies from the `pyproject.toml` file:
    ```bash
    uv pip install -e .
    ```

---

## Overview
**Jupyter Notebook Provided:** A starter Jupyter Notebook, `HW2_spring2026_final.ipynb`, is provided in the assignment folder. You are encouraged to use it to develop and test your code for each task.

The goal of this assignment is to work with molecular data using RDKit and SMILES strings, perform feature extraction and machine learning, and explore different data format structures.

**Assignment Structure:**
- **Part 1:** Molecular Featurization and Machine Learning (~5-6 hours)
- **Part 2:** Data Format Structures (2D graphs, 3D models, file formats, graph data types) (~2-3 hours)

**Total Estimated Time:** 7-9 hours

**Submission:** Submit your Python code file(s) that complete all tasks below.

**Dataset:** `smiles_tg.csv` - SMILES strings with glass transition temperature (Tg) values

---

## Part 1: Molecular Featurization and Machine Learning

### Task 1.1: Load and Explore Data
Import the `smiles_tg.csv` file into a dataframe and identify how many unique SMILES strings are in the dataset.

**Deliverables:**
- Load the CSV file
- Count and print the number of unique SMILES strings
- Print the first 10 rows of the dataframe

---

### Task 1.2: Clean Data
Drop the duplicate SMILES strings in the dataset, keep the first entry and reset the index.

**Deliverables:**
- Remove duplicate SMILES strings
- Reset the index
- Print the shape of the dataframe before and after

---

### Task 1.3: Create Basic Descriptors Function
We are going to use the RDKit library to convert the SMILES strings into molecular objects that we will use as features for our models. Let's first generate basic descriptors for the molecules.

Write a function called `get_basic_descriptors` that takes a SMILES string as input and returns a dictionary of descriptors.

**Required descriptors:**
- `'MW'`: molecular weight
- `'HBD'`: number of hydrogen bond donors
- `'HBA'`: number of hydrogen bond acceptors
- `'TPSA'`: topological polar surface area
- `'Rotatable_Bonds'`: number of rotatable bonds

**Deliverables:**
- Define the `get_basic_descriptors(smiles)` function
- Test it on a sample SMILES string
- Print the output to verify it returns a dictionary

---

### Task 1.4: Create Morgan Fingerprint Function
Create a function called `get_morgan_fingerprint` that takes in SMILES strings and generates Morgan fingerprints with a radius of 2 and a length nBits=1024. Return the fingerprint as a list.

**Function parameters:**
- `smiles`: SMILES string
- `radius`: fingerprint radius (default=2)
- `nBits`: number of bits (default=1024)

**Hint:** This can be done in a very few lines of code (don't complicate it).

**Deliverables:**
- Define the `get_morgan_fingerprint(smiles, radius, nBits)` function
- Test it on a sample SMILES string
- Print the length of the fingerprint to verify

---

### Task 1.5: Create Topological Fingerprint Function
Create a function called `get_topological_fingerprint` that generates topological fingerprints from SMILES strings. These fingerprints capture the 2D structural features of molecules.

**Function parameters:**
- `smiles`: SMILES string
- `nBits`: number of bits (default=25)

**Deliverables:**
- Define the `get_topological_fingerprint(smiles, nBits)` function
- Test it on a sample SMILES string
- Return the fingerprint as a list

---

### Task 1.6: Generate Basic Descriptors for Dataset
We are now going to use these functions to generate features for our models. Use the `get_basic_descriptors` function to convert the SMILES strings in the dataset to features.

Create a list that contains the descriptors for each SMILES string in the dataset. Remember, your function returns a dictionary of descriptors, so you will have a single list where each SMILES string is represented by a dictionary of descriptors.

**Expected output format:**
```python
[{'MW': 167.188, 'HBD': 0, 'HBA': 5, 'TPSA': 75.99, 'Rotatable_Bonds': 0}, {...}, ...]
```

**Deliverables:**
- Apply `get_basic_descriptors` to all SMILES strings
- Print the first entry to verify format

---

### Task 1.7: Clustering and PCA Visualization
Convert the list of dictionaries into a dataframe called `df_descriptors`.

**Steps:**
1. Scale the features using StandardScaler from sklearn.preprocessing
2. Fit and transform the dataframe (call this scaled dataframe X)
3. Use KMeans clustering with n_clusters=5, random_state=0
4. Extract the labels using kmeans.labels_ and add as a column 'cluster' in df_descriptors
5. Use PCA to reduce the dimensionality of X to 2 dimensions
6. Add pca1 and pca2 columns to the df_descriptors dataframe
7. Use seaborn to create a scatter plot with data=df_descriptors, x=pca1, y=pca2, hue=cluster

**Deliverables:**
- Code for all steps above
- Save the PCA clustering plot as `pca_clustering.png`

---

### Task 1.8: UMAP Visualization
UMAP is a dimensionality reduction technique used for visualization of high-dimensional data. Take the list of dictionaries we created earlier and create another dataframe from it.

**Requirements:**
- Create a UMAP plot with 2 dimensions
- Set UMAP parameters: n_neighbors=15, min_dist=0.1, n_components=2
- When making the UMAP plot, set a color bar as the Tg values from the original dataframe

**Deliverables:**
- Code to create UMAP embedding
- Save the UMAP plot as `umap_visualization.png`

---

### Task 1.9: Train Initial Models
Create a train-test split of the data using the featurized dataset and the Tg values.

**Requirements:**
- Set test_size=0.3 and random_state=42
- Train a linear model (Ridge regression) on the training data
- Train a non-linear model (Random Forest) on the training data
- Test both models on the test set
- Print the R² and RMSE score for each model
- Create 2 parity plots, one for each model

**Questions to answer:**
- In either of these parity plots, can you spot a clear outlier in the model?
- If we were to remove this outlier, would our model improve?

**Deliverables:**
- Code for train-test split and model training
- Print R² and RMSE for both models
- Save parity plots as `ridge_parity.png` and `rf_parity.png`
- Answer the questions and save them in `hw2_discussion.md`

---

### Task 1.10: GridSearchCV Hyperparameter Tuning

Let's see if we can improve the Ridge and Random Forest models by tuning their hyperparameters using `GridSearchCV`.

**What is GridSearchCV?**
GridSearchCV is a tool from scikit-learn that automatically tests different combinations of hyperparameters (model settings) to find the best combination for your data. Instead of manually trying different values for parameters like `alpha` in Ridge regression or `n_estimators` in Random Forest, GridSearchCV tests all combinations and uses cross-validation to evaluate which settings produce the best model performance. This helps you optimize your models without guessing.

**Estimated Time:** ~15-20 minutes

---

#### **Part A: Ridge Model Tuning**

**Requirements:**
- Instantiate `GridSearchCV` with your Ridge model.
- Use the following parameter grid:
    ```python
    param_grid_ridge = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
    ```
- Use the following settings for the search: `cv=5`, `scoring='r2'`, `n_jobs=-1`.
- Fit the `GridSearchCV` object on your **training data**.

**Deliverables:**
1.  Print the best alpha found (`best_params_`).
2.  Print the best R² score from the cross-validation (`best_score_`).
3.  Use the best estimator (`best_estimator_`) to make predictions on the **test data**.
4.  Calculate and print the R² and RMSE for the tuned model on the test data.

---

#### **Part B: Random Forest Model Tuning**

**Requirements:**
- Instantiate `GridSearchCV` with your Random Forest model.
- Use the following parameter grid:
    ```python
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20]
    }
    ```
- Use the same settings for the search: `cv=5`, `scoring='r2'`, `n_jobs=-1`.
- Fit the `GridSearchCV` object on your **training data**.

**Deliverables:**
1.  Print the best parameters found (`best_params_`).
2.  Print the best R² score from the cross-validation (`best_score_`).
3.  Use the best estimator (`best_estimator_`) to make predictions on the **test data**.
4.  Calculate and print the R² and RMSE for the tuned model on the test data.

---

#### **Part C: Comparison**

**Deliverables:**
- Answer the following question: Did you see a significant improvement in the performance of the models on the test set after tuning? Compare the before and after R² and RMSE scores. Save them in `hw2_discussion.md`

---

### Task 1.11: Morgan Fingerprint Models
Generate features from the `get_morgan_fingerprint` function.

**Requirements:**
1. Create a list called `morgan_fingerprints` of the Morgan fingerprints for each SMILES string
2. Create a new dataframe from the morgan_fingerprints list
3. Split the dataset into test and train set (test_size=0.3, random_state=42)
4. Standardize the data using StandardScaler
5. Train a linear Ridge model (alpha=0.1)
6. Train a non-linear Random Forest model (n_estimators=100, random_state=42)
7. Test models on the test set
8. Print R² and RMSE score for each model
9. Plot a parity plot for each model

**Deliverables:**
- Code for all steps above
- Print R² and RMSE for both models
- Save parity plots as `morgan_ridge_parity.png` and `morgan_rf_parity.png`

---

### Task 1.12: Topological Fingerprint and Ensemble Models
Use the `get_topological_fingerprint` function to generate 2D structural features.

**Requirements:**
1. Create a list called `topological_fps` of the topological fingerprints for each SMILES string
2. Convert to a dataframe
3. Create a train-test split (test_size=0.3, random_state=42)
4. Train a Support Vector Regression (SVR) model
5. Train a Random Forest model
6. Create a 3rd ensemble model using VotingRegressor that combines the SVR and Random Forest models
7. Test each model
8. Print R² and RMSE score for all 3 models
9. Create a parity plot for each model

**Questions to answer:**
- Do we see any improvement in the performance when we combine into an ensemble of models?

**Deliverables:**
- Code for all steps above
- Print R² and RMSE for all three models
- Save parity plots as `topo_svr_parity.png`, `topo_rf_parity.png`, `topo_ensemble_parity.png`
- Answer the question as a comment in your code

---

### Task 1.13: Molecular Similarity Analysis
For the final exercise, we are going to identify the most similar molecules in our dataset relative to the first molecule in the dataset.

**Requirements:**
1. Generate Morgan fingerprints (radius=2, nBits=1024) for the first molecule
2. Generate Morgan fingerprints for all other molecules in the dataset
3. Calculate the Tanimoto similarity between the first molecule and the rest
4. Print the top 10 most similar molecules to the first molecule

**Expected output format:**
```
Most similar molecules:
                             SMILES                                     Similarity
0    *C1COC2C1OCC2Oc1ccc(cc1)CNC(=O)CCCCCCC(=O)NCc1...      1.000000
222                 *N=Occ1                                   0.444444
...
```

5. Use `MolsToGridImage` to display the first molecule and the 4 most similar molecules

**Deliverables:**
- Code to calculate Tanimoto similarity
- Print top 10 most similar molecules
- Save the grid image as `similar_molecules.png`

---

## Part 2: Data Format Structures

In this section, we will explore different ways to represent molecular data. We'll convert SMILES strings into various data formats and discuss their advantages and limitations.

---

### Task 2.1: Convert SMILES to 2D Molecular Structure Graph
Select 3 different SMILES strings from your dataset. Use RDKit to convert each SMILES string to a molecular graph representation and visualize the 2D molecular structure for each molecule.

**Hint:** Use RDKit's Draw.MolToImage() or similar functions.

**Deliverables:**
- Code to convert 3 SMILES to 2D structures
- Save visualizations as `molecule1_2d.png`, `molecule2_2d.png`, `molecule3_2d.png`

---

### Task 2.2: Convert SMILES to 3D Model
Take the same 3 SMILES strings and generate 3D conformers.

**Requirements:**
- Use RDKit's embedding functions (e.g., AllChem.EmbedMolecule, AllChem.UFFOptimizeMolecule)
- Visualize the 3D structures
- **Hint:** You can use py3Dmol or RDKit's built-in 3D visualization tools

**Deliverables:**
- Code to generate 3D conformers
- Save 3D visualizations or print coordinates

---

### Task 2.3: Save Molecules to Standard File Formats
Export the 3 molecules to common chemical file formats.

**Required formats:**
- SDF (Structure Data File) format
- MOL format
- PDB format

**Hint:** Use RDKit's Chem.MolToMolFile(), Chem.MolToPDBFile(), etc.

**Deliverables:**
- Code to save molecules in all three formats
- Files: `molecule1.sdf`, `molecule1.mol`, `molecule1.pdb` (and similarly for molecules 2 and 3)

---

### Task 2.4: Convert SMILES to Graph (Abstract Data Type)
Convert one of your SMILES strings to a graph data structure (nodes and edges).

**Requirements:**
- Nodes should represent atoms with properties (element, charge, etc.)
- Edges should represent bonds with properties (bond type, order)
- You can use NetworkX or create a custom dictionary/class representation
- Print or visualize the graph structure showing:
  - Number of nodes (atoms)
  - Number of edges (bonds)
  - Node attributes (atom types)
  - Edge attributes (bond types)

**Deliverables:**
- Code to create graph representation
- Print graph statistics (nodes, edges, attributes)
- Optionally: visualize the graph and save as `molecule_graph.png`

---

### Task 2.5: Discussion - Data Format Pros and Cons
For each of the 4 data formats you created above, discuss:

**1. 2D Molecular Structure Graph**
- Pros:
- Cons:
- Best use cases:

**2. 3D Model**
- Pros:
- Cons:
- Best use cases:

**3. Molecule Files (SDF/MOL/PDB)**
- Pros:
- Cons:
- Best use cases:

**4. Graph (Abstract Data Type)**
- Pros:
- Cons:
- Best use cases:

**Finally, answer:** Which representation would be most suitable for:
- Machine learning with graph neural networks?
- Visualizing molecules for publication?
- Sharing data with other researchers?
- Molecular dynamics simulations?

**Deliverables:**
- Write your discussion as multi-line comments in your code or in a separate markdown/text file
- Provide at least 2-3 points for each pro/con section
- Minimum 2-3 sentences for each "best use case" and final question

---

## Common Mistakes and Tips

### Part 1: Molecular Featurization
- **Mistake:** Forgetting to convert RDKit fingerprints to lists/arrays
  - **Fix:** Use `list(fingerprint)` or `np.array(fingerprint)` after generating fingerprints
- **Mistake:** Not scaling features before applying PCA or UMAP
  - **Fix:** Always use `StandardScaler` before dimensionality reduction
- **Mistake:** Using different random_state values across train-test splits
  - **Fix:** Consistently use `random_state=42` for all splits in this assignment
- **Mistake:** Applying GridSearchCV to the entire dataset instead of just training data
  - **Fix:** Only fit GridSearchCV on `X_train, y_train`, then evaluate on test set

### RDKit Specific
- **Mistake:** Not checking if SMILES string is valid before processing
  - **Fix:** Add `if mol is not None:` check after `Chem.MolFromSmiles()`
  - Example:
    ```python
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    ```
- **Mistake:** Confusing Morgan fingerprints with topological fingerprints
  - **Fix:** Morgan = `AllChem.GetMorganFingerprintAsBitVect()`, Topological = `Chem.RDKFingerprint()`

### Part 2: Data Formats
- **Mistake:** Not generating 3D coordinates before saving to PDB format
  - **Fix:** Must use `AllChem.EmbedMolecule()` and `AllChem.UFFOptimizeMolecule()` first
- **Mistake:** Saving files with incorrect extensions
  - **Fix:** Double-check file naming requirements - they must match exactly for grading
- **Mistake:** Forgetting to convert bond types when creating graph representations
  - **Fix:** RDKit bonds have a `.GetBondType()` method - use it for edge attributes

### Visualization
- **Mistake:** Not saving plots with the exact filename specified
  - **Fix:** Copy-paste the required filenames from instructions to avoid typos
- **Mistake:** Creating plots without axis labels or titles
  - **Fix:** Always add `plt.xlabel()`, `plt.ylabel()`, and `plt.title()`

---

## Grading Rubric

**Part 1: Molecular Featurization and Machine Learning (70 points)**
- Tasks 1.1-1.6 (Feature engineering): 20 points
- Tasks 1.7-1.8 (Visualization): 10 points
- Tasks 1.9-1.10 (Initial models and tuning): 15 points
- Tasks 1.11-1.12 (Advanced models): 15 points
- Task 1.13 (Similarity analysis): 10 points

**Part 2: Data Format Structures (30 points)**
- Tasks 2.1-2.4 (Format conversions): 20 points
- Task 2.5 (Discussion): 10 points

**Total: 100 points**

---

## Submission Guidelines

**Important Note on File Naming:** Throughout this assignment, you are required to save data and plots to specific filenames. It is critical that you use these exact filenames, as your submission may be graded by scripts that look for these files. Failure to use the correct filenames may result in a loss of points.

1. **Code:** Submit Python file(s) named `hw2_yourname.py` (or split into `hw2_part1.py`, `hw2_part2.py`)
2. **Figures:** Include all generated plots (.png files)
3. **Molecule Files:** Include the .sdf, .mol, .pdb files you created
4. **Discussion:** Include your written responses to discussion questions
5. **Comments:** Add clear comments explaining your approach

**Required files:**
- `hw2_yourname.py` - Your main Python code
- All PNG plots generated throughout the assignment
- Molecule files (SDF, MOL, PDB)
- `hw2_discussion.md` for your written responses

**Deadline:** February 19, 2026 at 23:59