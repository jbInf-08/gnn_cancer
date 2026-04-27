"""
Real public benchmark graphs for end-to-end training without proprietary TCGA files.

UCI / sklearn Wisconsin Breast Cancer: a standard tabular dataset bundled with scikit-learn
(Wolberg et al.; cited via sklearn). We use k-NN in standardized feature space as the graph
(undirected), then node-level binary classification. This is not TCGA-BRCA omics, but it is
honest, reproducible, and suitable for verifying the training stack.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def load_uci_breast_cancer_graph(
    k_neighbors: int = 10,
    seed: int = 42,
) -> Data:
    """
    Build a PyG Data object from sklearn's Wisconsin breast cancer dataset.

    Labels follow sklearn: 0 = malignant, 1 = benign.
    """
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1")

    raw = load_breast_cancer()
    X = StandardScaler().fit_transform(raw.data)
    y = raw.target.astype(np.int64)
    n = X.shape[0]

    # Symmetric kNN graph (undirected); exclude self-loops
    knn = kneighbors_graph(
        X, n_neighbors=k_neighbors, mode="connectivity", include_self=False
    )
    # Undirected: union with transpose
    adj = knn.maximum(knn.T)
    adj.setdiag(0)
    rows, cols = adj.nonzero()
    edge_index = np.vstack([rows, cols]).astype(np.int64)

    idx = np.arange(n)
    idx_train, idx_temp, _, _ = train_test_split(
        idx, y, test_size=0.3, stratify=y, random_state=seed
    )
    y_temp = y[idx_temp]
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.as_tensor(idx_train, dtype=torch.long)] = True
    val_mask[torch.as_tensor(idx_val, dtype=torch.long)] = True
    test_mask[torch.as_tensor(idx_test, dtype=torch.long)] = True

    data = Data(
        x=torch.as_tensor(X, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        y=torch.as_tensor(y, dtype=torch.long),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    # Document provenance (not a standard PyG field, but useful for json export)
    data.dataset_name = "sklearn_breast_cancer_wisconsin"
    data.dataset_description = (
        "UCI ML Breast Cancer Wisconsin (Diagnostic) via sklearn.datasets.load_breast_cancer; "
        "kNN graph in standardized feature space."
    )
    data.k_neighbors = k_neighbors
    return data
