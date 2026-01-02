# Homework Assignments Summary - Spring 2026

## HW1: Materials Data Extraction & Featurization

**Focus**: Working with extracting crystalline materials data from multiple sources

### Key Tasks:
1. Extract data via Materials Project API
- Extract quaternary Li-containing materials with specific constraints:
  - Band gap: 0.5-3.0 eV
  - Density: 2.0-4.0 g/cm³
- Clean data by handling duplicates and averaging values

2. Extract data from literature via agent (KnowMat2)
- Set up Ollama for local LLM testing (free alternative to OpenAI API)
- Extract data from papers that contain LiFeP2O7 using KnowMat2 (Note: Not all papers are useful)
- Code must support both Ollama (for testing) and OpenAI API (for grading)
- Compare atomic coordinates results with Materials Project API
- Discuss Ollama performance vs OpenAI expectations

3. Use featurizers and run simple machine learning model on it
- Use CBFV (Composition-Based Feature Vector) with three different featurizers:
  - Oliynyk
  - Onehot
  - Mat2Vec
- Train Random Forest regressors on each featurized dataset to predict band gaps
- Compare model performance using mean squared error

4. Download and upload data using NOMAD or Materials Commons
- Browse and download data from NOMAD repository
- Prepare processed data for upload to a repository
- Upload data to NOMAD staging server or Materials Commons
- Compare data repositories (Materials Project, NOMAD, Materials Commons, Literature)
- Integrate data from multiple sources

### Learning Objectives:
- Materials Project API usage
- Agentic AI usage for literature extraction
- Working with local LLMs (Ollama) and commercial APIs (OpenAI)
- Writing model-agnostic code that supports multiple LLM backends
- Data cleaning and preprocessing
- Composition-based featurization techniques
- Model comparison and evaluation
- Working with open materials science data repositories
- Data integration from multiple sources
- Understanding repository ecosystems

---

## HW2: Molecular Featurization, SMILES in ML; data format structure

**Focus**: Working with molecular data using RDKit and SMILES strings

### Key Tasks:

#### Feature Generation
Implement three types of molecular featurization:

1. **Basic Descriptors**:
   - Molecular weight
   - H-bond donors/acceptors
   - Topological polar surface area (TPSA)
   - Rotatable bonds

2. **Morgan Fingerprints**:
   - Radius = 2
   - nBits = 1024

3. **Topological Fingerprints**:
   - nBits = 25

#### Machine Learning Tasks
- Perform clustering (KMeans, n_clusters=5)
- Dimensionality reduction (PCA, UMAP)
- Train models:
  - Linear: Ridge regression
  - Non-linear: Random Forest, SVR
- Hyperparameter optimization with GridSearchCV
- Create ensemble models using VotingRegressor
- Calculate molecular similarity using Tanimoto similarity

#### Data Formatting Tasks
- Convert the SMILES text into:
1.  Molecular structure graph
2.  3D model
3.  Molecule Files
3.  Graph (abstract data type)
- What are the pros and cons for each data type?

### Target Property:
Glass transition temperature (Tg)

### Learning Objectives:
- SMILES string processing
- Molecular featurization techniques
- Clustering and dimensionality reduction
- Ensemble learning
- Molecular similarity analysis

---

## HW3: Deep Learning - VAE & CrabNet

**Focus**: Generative models and transformer-based architectures

### Part 1: Variational Autoencoder (VAE)

#### Architecture:
**Encoder**:
- Input: 32×32 grayscale microstructure images
- Conv layer 1: 16 filters, kernel=3, stride=2, padding=1
- Conv layer 2: 32 filters, kernel=3, stride=2, padding=1
- ReLU activations
- Latent dimension: 8

**Decoder**:
- Fully connected layer from latent space
- Transposed conv 1: 16 filters, kernel=3, stride=2, padding=1
- Transposed conv 2: 1 filter, kernel=3, stride=2, padding=1
- Sigmoid activation

#### Tasks:
- Implement VAE loss (reconstruction + KL divergence with β=0.1)
- Train on microstructure images
- Generate similar microstructure images by sampling latent space
- Generate 5 variations of an input image

### Part 2: CrabNet (Transformer Model)

#### Tasks:
- Train CrabNet on shear modulus dataset
- Use CBFV for featurization
- Evaluate on 10% test set
- Make predictions on 10 new material compositions:
  - Fe₂O₃, TiO₂, Al₂O₃, SiO₂, ZnO
  - CaTiO₃, Li₄Ti₅O₁₂, BaTiO₃
  - LiFePO₄, MgAl₂O₄

### Learning Objectives:
- Convolutional neural networks for images
- Variational autoencoders and generative modeling
- Latent space manipulation
- Transformer architectures for materials science
- Transfer learning with pre-trained models

---
