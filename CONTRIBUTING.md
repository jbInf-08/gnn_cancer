# Contributing

## Environment

1. **Python** 3.9+ (3.11 matches CI).
2. From the repository root, use an **editable install** so imports resolve:

   ```bash
   pip install -e ".[dev]"
   ```

3. **PyTorch Geometric** sometimes needs extra wheels (`torch-scatter`, `torch-sparse`) that must match your PyTorch and CUDA/CPU build. If `pip install -e .` fails on the PyG stack, follow the [official PyG install instructions](https://pyg.org/install.html) for your platform, then re-run `pip install -e .`.

## Data and secrets

- Do not commit **API keys**, raw patient data, or large checkpoints. See `.gitignore`.
- **WandB**: set `WANDB_API_KEY` or use `config/api_keys.json` (gitignored) for training runs.
- **GDC client**: use `scripts/download_gdc_client.sh` (Linux/macOS) or `conda install -c bioconda gdc-client`; do not commit vendor binaries.

## Running the main pipeline

```bash
python train.py --cancer_type BRCA --model GCN --data_source TCGA
```

BRCA + TCGA expects a prepared graph at `data/processed/BRCA_comprehensive_data.pt` when you use that path; see `scripts/` for data prep.

```bash
pytest
```

matches the checks in `.github/workflows/ci.yml` (byte-compile + tests).

## Where to put new work

- **Core models and utilities** → `gnn_cancer/`.
- **New one-off training scripts** → avoid duplicating root drivers; prefer a **branch**, `notebooks/`, or extending `train.py` / the package. Do **not** add new files under `legacy/` (see `legacy/README.md`).

## Pull requests

- Keep changes focused and described in complete sentences.
- Ensure `pytest` passes and that you have not staged generated `results/` artifacts or local-only docs (e.g. `docs/PROJECTS_*.md` are gitignored by design).
