# microstructure-gen

Offline, laptop-friendly generative modeling demo for synthetic microstructure images.

## Quickstart (uv)

```bash
uv venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

uv pip install -e ".[dev]"
```

## Project layout

- `microgen/`: package code
- `tests/`: test suite
- `data/`: generated datasets
- `checkpoints/`: saved model checkpoints
- `outputs/`: generated image grids/figures
- `notebooks/`: demo notebook(s)
