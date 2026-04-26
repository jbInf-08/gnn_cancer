"""
Debug script to isolate GAT issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

def debug_gat():
    """Debug GAT model step by step"""
    print("Debugging GAT model...")
    
    # Create simple test data
    num_nodes = 10
    num_features = 19
    num_edges = 20
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randint(0, 8, (num_edges, 1)).float()
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  edge_attr: {edge_attr.shape}")
    
    # Test basic GAT layer
    try:
        gat_layer = GATv2Conv(
            in_channels=num_features,
            out_channels=32,
            heads=4,
            edge_dim=1
        )
        
        print(f"GAT layer created successfully")
        print(f"  in_channels: {num_features}")
        print(f"  out_channels: 32")
        print(f"  heads: 4")
        print(f"  edge_dim: 1")
        
        # Test forward pass
        output = gat_layer(x, edge_index, edge_attr)
        print(f"✓ Basic GAT forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({num_nodes}, 32*4) = ({num_nodes}, 128)")
        
    except Exception as e:
        print(f"✗ Basic GAT failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    debug_gat()
