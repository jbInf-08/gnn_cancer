"""Tests for public benchmark graph builders (no private TCGA files)."""
import torch

from gnn_cancer.benchmark_datasets import load_uci_breast_cancer_graph


def test_wisconsin_breast_cancer_shape_and_split():
    data = load_uci_breast_cancer_graph(k_neighbors=10, seed=42)
    assert data.num_nodes == 569
    assert data.x.shape[1] == 30
    assert data.y.shape[0] == 569
    assert int(data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()) == 569
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] > 0
