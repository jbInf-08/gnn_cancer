import sys
from pathlib import Path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from typing import Tuple
from gnn_cancer.models.gnn_models import get_model

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        N = z1.size(0)
        sim = torch.mm(z1, z2.t()) / self.temperature
        positives = sim.diag().unsqueeze(1)
        negatives = sim[~torch.eye(N, dtype=torch.bool, device=z1.device)].view(N, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(N, device=z1.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

class GraphAugmentation:
    def __init__(self, edge_dropout: float = 0.2, node_dropout: float = 0.1, feature_dropout: float = 0.1):
        self.edge_dropout = edge_dropout
        self.node_dropout = node_dropout
        self.feature_dropout = feature_dropout
    def augment(self, data: Data) -> Tuple[Data, Data]:
        return self._create_view(data), self._create_view(data)
    def _create_view(self, data: Data) -> Data:
        edge_index, _ = dropout_adj(data.edge_index, p=self.edge_dropout)
        x = data.x.clone()
        node_mask = torch.rand(x.size(0)) > self.node_dropout
        x[~node_mask] = 0
        feature_mask = torch.rand(x.size()) > self.feature_dropout
        x = x * feature_mask
        return Data(x=x, edge_index=edge_index)

class SelfSupervisedPretrainer:
    def __init__(self, model: nn.Module, augmenter: GraphAugmentation, device: torch.device, lr: float = 0.001):
        self.model = model
        self.augmenter = augmenter
        self.device = device
        self.criterion = ContrastiveLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    def pretrain(self, data: Data, epochs: int = 100) -> list:
        self.model.train()
        losses = []
        for epoch in range(epochs):
            v1, v2 = self.augmenter.augment(data)
            v1, v2 = v1.to(self.device), v2.to(self.device)
            z1 = self.model(v1.x, v1.edge_index)
            z2 = self.model(v2.x, v2.edge_index)
            loss = self.criterion(z1, z2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        return losses

def pretrain_model(data: Data, model_type: str = 'GCN', hidden_channels: int = 128, num_layers: int = 4, device: torch.device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_type, in_channels=data.num_node_features, hidden_channels=hidden_channels, num_layers=num_layers).to(device)
    augmenter = GraphAugmentation()
    pretrainer = SelfSupervisedPretrainer(model, augmenter, device)
    losses = pretrainer.pretrain(data)
    return model, losses 