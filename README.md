# gnn_cancer

[![CI](https://github.com/jbInf-08/gnn_cancer/actions/workflows/ci.yml/badge.svg)](https://github.com/jbInf-08/gnn_cancer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Graph neural networks (PyTorch Geometric) for cancer data workflows: GCN, GraphSAGE, and GAT on graph data derived from public resources (e.g. TCGA/GDC orientation), with WandB logging and optional pretraining.

**This repository is the implementation.** Training code lives under the `gnn_cancer/` package; runnable drivers remain at the repository root. Large data, checkpoints, and API keys are **not** included (see `.gitignore`).

## Golden path

```bash
pip install -e ".[dev]"
make train-smoke
python train.py --cancer_type BRCA --model GCN --data_source TCGA
```

## Install

```bash
pip install -e ".[dev]"          # from repo root, editable install
# or:
pip install -e ".[dev,notebooks]"
```

**PyTorch Geometric** may require matching `torch-scatter` / `torch-sparse` wheels for your OS/CUDA. See the [PyG install guide](https://pyg.org/install.html) if `pip install -e .` does not complete.

### GDC client (data transfer, optional)

The GDC **binary** is not stored in the repository. On Linux or macOS, use:

```bash
bash scripts/download_gdc_client.sh
```

Or: `conda install -c bioconda gdc-client`. The script downloads the official Ubuntu distribution, verifies the published MD5, and explains fallbacks if the NCI link changes.

### UUID ↔ TCGA barcode mapping

`uuid_to_barcode.csv` is **not** committed. To regenerate a mapping from the GDC API:

```bash
python scripts/download_uuid_to_barcode.py --project TCGA-BRCA
```

(Outputs under `data/metadata/` by default; path is gitignored when appropriate.)

## Usage

**Training**

- **Public benchmark (no GDC download, no W&B key):** trains GCN, GraphSAGE, and GAT on the UCI Wisconsin breast-cancer set via scikit-learn, with a kNN graph:
  ```bash
  python train.py --cancer_type BRCA --model GCN --data_source BENCHMARK --no-wandb --quiet --export-results results/reproducible_baseline_uwbc.json
  ```
  Metrics are written to the JSON path you pass to `--export-results` (see `docs/UPDATED_PAPER.md`).

- **TCGA / preprocessed graph** (optional; requires `WANDB_API_KEY` or `config/api_keys.json` unless you pass `--no-wandb`):
  ```bash
  python train.py --cancer_type BRCA --model GCN --data_source TCGA
  ```
  For the BRCA+TCGA path, build or obtain `data/processed/BRCA_comprehensive_data.pt` first (see `scripts/`).

`train.py` trains GCN, GraphSAGE, and GAT in one run; `--model` is only used for WandB run naming.

**End-to-end driver:**

```bash
python run_pipeline.py
```

**Programmatic import:**

```python
from gnn_cancer.models.gnn_models import get_model

model = get_model("GraphSAGE", in_channels=256, out_channels=2, hidden_channels=128, num_layers=3)
```

## Layout

- `gnn_cancer/` — installable package (`models/`, `utils/`, `data_sources/`, …)
- `train.py` — primary training entry
- `experiments/` — exploratory standalone scripts moved out of root for maintainability
- `scripts/` — GDC/data helpers (including `download_gdc_client.sh`, `download_uuid_to_barcode.py`)
- `legacy/` — old experimental training scripts (see `legacy/README.md`)
- `docs/UPDATED_PAPER.md` — research draft (not automatically aligned with the running code)
- `docs/PROJECTS_*.md` — **optional** private notes; these patterns are gitignored so you can keep them only on your machine
- `notebooks/results_summary.ipynb` — minimal template to summarize **your** result artifacts

Regenerated images and checkpoints should stay under `results/` and are ignored from git; see `.gitignore`.

## Verifying quantitative claims

```bash
python scripts/audit_claims.py
```

Writes `results/claims_audit.json` (GDC TCGA-BRCA case count from the public API, sklearn Wisconsin facts, scan for legacy `967189`-style literals, and `train.py` loss wiring). Use it before asserting cohort sizes in writing.

## Tests and CI

```bash
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `main`/`master`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (environment, PyG install, WandB, tests).

## Release discipline

Use annotated tags and update `CHANGELOG.md` for each release cut so consumers can track model/training behavior changes.

## License

MIT — see [LICENSE](LICENSE).
