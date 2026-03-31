# Bayesian Optimization for Materials Informatics

This folder contains teaching notebooks and a local `uv` environment for a Bayesian optimization unit in a materials informatics course.

## Environment

Create or refresh the environment with:

```powershell
uv sync
```

Activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Launch JupyterLab with:

```powershell
uv run jupyter lab
```

## Notebooks

- `bayesian_optimization_deep_dive.ipynb`: Core lecture notebook connecting Gaussian processes to Bayesian optimization with materials-oriented examples.
- `honegumi_materials_workflow.ipynb`: Honegumi template generation plus the Honegumi RAG assistant workflow for natural-language problem descriptions.
