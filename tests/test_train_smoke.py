import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Data
from gnn_cancer.models.gnn_models import get_model


def test_tiny_end_to_end_train_step():
    x = torch.randn(12, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]],
        dtype=torch.long,
    )
    y = torch.randint(0, 2, (12,), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    model = get_model("GCN", in_channels=16, out_channels=2, hidden_channels=8, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits, data.y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
