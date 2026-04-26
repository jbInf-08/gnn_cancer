"""Smoke test: package and core modules import (no data, no GPU)."""
import sys
from pathlib import Path

import pytest

# Ensure repository root (parent of gnn_cancer package) is on path for local runs
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_gnn_cancer_package():
    import gnn_cancer

    assert isinstance(gnn_cancer.__version__, str)


def test_get_model_factory():
    pytest.importorskip("torch_geometric")
    import torch
    from gnn_cancer.models.gnn_models import get_model

    m = get_model("GCN", in_channels=8, out_channels=2, hidden_channels=16, num_layers=2, num_heads=2)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    m.eval()
    with torch.no_grad():
        y = m(x, edge_index)
    assert y.shape == (4, 2)
