import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleGAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=0.3))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.3))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * 4, output_dim, heads=1, dropout=0.3))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

def main():
    logger.info("Testing small GAT model on real data...")
    
    # Load the real data
    data_path = Path("data/enhanced/real_only_torch_geometric_data.pt")
    if not data_path.exists():
        logger.error("Real data not found!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(data_path, map_location=device, weights_only=False)
    
    logger.info(f"Data loaded: {data.x.shape}, {data.edge_index.shape}")
    
    # Create small model
    model = SimpleGAT(
        input_dim=data.x.size(1),
        hidden_dim=32,  # Very small
        output_dim=2,
        num_layers=2
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Simple train/test split
    num_nodes = data.x.size(0)
    indices = np.arange(num_nodes)
    y_np = data.y.cpu().numpy()
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(indices, y_np))
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index)
            test_pred = test_out[test_mask].argmax(dim=1)
            test_true = data.y[test_mask]
            f1 = f1_score(test_true.cpu().numpy(), test_pred.cpu().numpy(), average='binary', zero_division=0)
            
            logger.info(f"Epoch {epoch+1}: Loss={loss.item():.4f}, F1={f1:.4f}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 