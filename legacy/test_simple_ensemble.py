"""
Simple test for ensemble models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

def test_simple_gat():
    """Test a simple GAT model"""
    print("Testing simple GAT model...")
    
    # Test data
    num_nodes = 10
    num_features = 19
    num_edges = 20
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Simple GAT model
    class SimpleGAT(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleGAT, self).__init__()
            self.conv1 = GATv2Conv(input_dim, hidden_dim // 4, heads=4, concat=True)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
            self.output = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x, edge_index):
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            return self.output(x)
    
    model = SimpleGAT(input_dim=num_features, hidden_dim=128, output_dim=2)
    
    with torch.no_grad():
        output = model(x, edge_index)
        print(f"Simple GAT output shape: {output.shape}")
    
    print("Simple GAT test passed!")

def test_skip_connections():
    """Test skip connections"""
    print("Testing skip connections...")
    
    # Test data
    num_nodes = 10
    num_features = 19
    num_edges = 20
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # GAT with skip connections
    class GATWithSkip(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GATWithSkip, self).__init__()
            self.conv1 = GATv2Conv(input_dim, hidden_dim // 4, heads=4, concat=True)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
            
            # Skip connection projections
            self.skip1 = nn.Linear(input_dim, hidden_dim)
            self.skip2 = nn.Linear(hidden_dim, hidden_dim)
            
            self.output = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x, edge_index):
            # First layer
            x1 = F.elu(self.conv1(x, edge_index))
            skip1 = self.skip1(x)
            if skip1.shape == x1.shape:
                x1 = x1 + skip1
            
            # Second layer
            x2 = F.elu(self.conv2(x1, edge_index))
            skip2 = self.skip2(x1)
            if skip2.shape == x2.shape:
                x2 = x2 + skip2
            
            return self.output(x2)
    
    model = GATWithSkip(input_dim=num_features, hidden_dim=128, output_dim=2)
    
    with torch.no_grad():
        output = model(x, edge_index)
        print(f"GAT with skip connections output shape: {output.shape}")
    
    print("Skip connections test passed!")

if __name__ == "__main__":
    test_simple_gat()
    test_skip_connections()
    print("All tests passed!")
