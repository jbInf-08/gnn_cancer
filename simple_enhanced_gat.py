"""
Simplified Enhanced GAT for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool

class SimpleEnhancedGAT(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256, 
                 output_dim: int = 2, 
                 num_layers: int = 3, 
                 num_heads: int = 8, 
                 dropout: float = 0.3,
                 use_edge_attr: bool = True):
        super(SimpleEnhancedGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # Edge projection
        if use_edge_attr:
            self.edge_proj = nn.Linear(1, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(
            input_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            edge_dim=hidden_dim if use_edge_attr else None, 
            concat=True
        ))
        
        # Middle layers
        for i in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                hidden_dim, 
                hidden_dim // num_heads, 
                heads=num_heads, 
                dropout=dropout, 
                add_self_loops=True, 
                edge_dim=hidden_dim if use_edge_attr else None, 
                concat=True
            ))
        
        # Last layer
        self.convs.append(GATv2Conv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            edge_dim=hidden_dim if use_edge_attr else None, 
            concat=True
        ))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, elementwise_affine=True) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Process edge attributes
        if self.use_edge_attr and edge_attr is not None:
            edge_embeddings = self.edge_proj(edge_attr)
        else:
            edge_embeddings = None
        
        # Forward pass through GAT layers
        for i, conv in enumerate(self.convs):
            # Apply GAT convolution
            if edge_embeddings is not None:
                x_new = conv(x, edge_index, edge_embeddings)
            else:
                x_new = conv(x, edge_index)
            
            # Apply layer normalization
            x_new = self.layer_norms[i](x_new)
            
            # Apply activation
            x_new = F.elu(x_new)
            
            # Apply dropout
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            x = x_new
        
        # Graph-level pooling
        if batch is None:
            return self.output_proj(x)
        
        # Simple mean pooling
        x = global_mean_pool(x, batch)
        
        # Output projection
        return self.output_proj(x)

def test_simple_gat():
    """Test the simple enhanced GAT model"""
    print("Testing simple enhanced GAT model...")
    
    try:
        # Create test data
        num_nodes = 50
        num_features = 19
        num_edges = 100
        
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randint(0, 8, (num_edges, 1)).float()
        
        # Create model
        model = SimpleEnhancedGAT(
            input_dim=num_features,
            hidden_dim=128,
            output_dim=2,
            num_layers=3,
            num_heads=4,
            dropout=0.3,
            use_edge_attr=True
        )
        
        # Test forward pass
        with torch.no_grad():
            output = model(x, edge_index, edge_attr)
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected output shape: ({num_nodes}, 2)")
            
            # Test with batch
            batch = torch.zeros(num_nodes, dtype=torch.long)
            batch[num_nodes//2:] = 1
            
            output_batch = model(x, edge_index, edge_attr, batch)
            print(f"✓ Batch forward pass successful")
            print(f"  Batch output shape: {output_batch.shape}")
            print(f"  Expected batch output shape: (2, 2)")
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_gat()
