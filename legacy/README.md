# Legacy and experimental scripts

This folder holds **superseded or exploratory** one-off training and optimization
scripts kept for reference. They are not part of the supported workflow.

**Supported entry points** (at repository root):

- `train.py` — primary GNN training (BRCA/TCGA graph or Kaggle fallback via `load_data`)
- `pretrain.py` — self-supervised pretraining helper
- `run_pipeline.py` — preprocess → graph build → train → evaluate
- `evaluate_models.py` — model comparison
- `build_graph.py` / `preprocess_data.py` — data preparation

Run them from the repo root after `pip install -e .` (or with `PYTHONPATH` set to
the project root; root scripts add the path automatically).

The Python package for shared code lives under `gnn_cancer/`.
