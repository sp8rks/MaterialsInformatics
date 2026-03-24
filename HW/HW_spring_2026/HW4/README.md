# HW4: Bayesian Inference, Gaussian Processes, and Bayesian Optimization

## Setup

### First-Time Setup

1. **Install UV** (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Navigate to the HW4 folder**:
```bash
cd HW/HW_spring_2026/HW4
```

3. **Sync the environment** (creates `.venv` and installs all dependencies):
```bash
uv sync
```

4. **Activate the virtual environment**:
```bash
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

5. **Read the assignment instructions** in `HW4_instructions.md`.

### Quick Start (After Initial Setup)

```bash
cd HW/HW_spring_2026/HW4
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python hw4_yourname.py
```

Or open `HW4_spring2026.ipynb` in Jupyter:
```bash
jupyter notebook HW4_spring2026.ipynb
```

---

## Assignment Structure

See `HW4_instructions.md` for complete instructions, task descriptions, and rubric.

**Total Estimated Time:** 15–20 hours | **Due:** April 7, 2026

### Part 1: Bayesian Inference and Information Theory (~5–7 hrs)

Build a complete Bayesian campaign modeled on a YSZ thermal barrier coating screening study:
- **Task 1.1** — Bayesian updating with a Beta-Bernoulli conjugate prior; posterior tracking across 10 adhesion tests
- **Task 1.2** — Prior sensitivity analysis: three researchers, three priors, same data
- **Task 1.3** — Bayesian linear regression for κ–φ (thermal conductivity vs. porosity); prior and posterior predictive distributions
- **Task 1.4** — Shannon entropy, differential entropy, joint/conditional entropy; entropy reduction across observations
- **Task 1.5** — KL divergence, reverse KL, f-divergences (χ², Hellinger, TV), Jensen-Shannon divergence
- **Task 1.6** — Cross-entropy, negative log-likelihood, Brier score, proper scoring rules, calibration
- **Task 1.7** — Mutual information, conditional MI, Expected Information Gain (EIG)
- **Task 1.8** — Differential entropy of Gaussians, GP posterior entropy, ELBO derivation
- **Task 1.9** *(Bonus)* — Fisher information, Cramér-Rao bound, information geometry

### Part 2: Gaussian Process Fundamentals (~3 hrs)

Map elastic modulus across the Al-Cu-Mg ternary alloy system using Gaussian Processes:
- **Task 2.1** — GP priors: RBF, Matérn, Periodic, Rational Quadratic; kernel heatmaps
- **Task 2.2** — Kernel engineering: sum/product kernels, ARD, composite kernels; LML comparison
- **Task 2.3** — Aleatoric vs. epistemic uncertainty; hyperparameter sensitivity
- **Task 2.4** — Log marginal likelihood (LML) landscape: 1D sweep and 2D surface

### Part 3: Bayesian Optimization by Hand (~5–7 hrs)

Run a BO campaign for StructureLab's crossed-barrel lattice crash absorbers (EV battery protection):
- **Tasks 3.1–3.2** — Load dataset, compute top-5% threshold, initialize 5-sample campaign
- **Task 3.3** — GP surrogate with `Matern(nu=2.5)` + feature normalization
- **Task 3.4** — UCB acquisition function + κ sensitivity analysis
- **Task 3.5** — Expected Improvement (EI) acquisition function
- **Task 3.6** — Probability of Improvement (PI) + three-way acquisition comparison
- **Task 3.7** — Simple regret, cumulative regret, theoretical O(1/√T) bounds, adaptive κ_t *(Bonus)*
- **Task 3.8** — Full campaigns: UCB, EI, and random baseline (150-iteration budget)
- **Task 3.9** — Failure modes: kernel misspecification, non-Gaussian noise, batch queries
- **Task 3.10** — Final report: plots and written summary for StructureLab

### Part 4: Bayesian Hyperparameter Tuning with Optuna (~2–3 hrs)

Predict heat capacity (Cp) of high-entropy oxide ceramics for concentrated solar power:
- **Task 4.1** — Explore `data/cp_data_cleaned.csv`; understand data leakage risk
- **Task 4.2** — CBFV features + group-based train/test split (`GroupShuffleSplit` by formula)
- **Task 4.3** — Optuna objective function with inline comments on each hyperparameter
- **Task 4.4** — Run 50-trial TPE study; compare to grid search cost
- **Task 4.5** — Evaluate best model; parity plot and optimization history
- **Task 4.6** — Compare RF to HW3 CrabNet; connect TPE back to the GP-UCB framework from Part 3

---

## Data Files

| File | Used in | Description |
|---|---|---|
| `data/crossed_barrel_dataset.csv` | Parts 2, 3 | Geometric parameters (n, θ, r, t) → toughness for ~1800 simulated lattice designs |
| `data/cp_data_cleaned.csv` | Part 4 | Formula + temperature T → heat capacity Cp for ~300 HEO compounds |

---

## Submission

| File | Description |
|---|---|
| `hw4_yourname.py` or `HW4_spring2026.ipynb` | Code |
| `hw4_yourname_answers.md` (or notebook markdown cells) | All written answers |
| All `.png` files listed in `HW4_instructions.md` | Figures (27 files + 2 bonus) |

**Key rules:**
- Use `random_state=88` everywhere
- Do not use `plt.show()` — save all figures to files
- Written answers go in markdown, not code comments or print statements
- All functions must have docstrings (graded in Part 3)
