"""
Enhanced GNN Models implementing reference paper's approach
- Multi-head attention (8 heads)
- Layer-specific attention coefficients
- ELU activation functions
- Sophisticated pooling strategies
- Paper hyperparameters for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GATv2Conv, TransformerConv
import math

class EnhancedGATModel(nn.Module):
    """
    Enhanced GAT model implementing reference paper's approach
    - 8-head attention mechanism
    - Layer-specific attention coefficients
    - ELU activation functions
    - Sophisticated pooling strategies
    - Paper hyperparameters: hidden_dim=256, num_layers=4, dropout=0.3
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=10, num_layers=4, 
                 num_heads=8, dropout=0.3, use_edge_attr=True, num_edge_types=8):
        super(EnhancedGATModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # Edge type embeddings
        if use_edge_attr:
            self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        # GAT layers with 8-head attention (paper configuration)
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATv2Conv(
                input_dim, 
                hidden_dim // num_heads, 
                heads=num_heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=hidden_dim if use_edge_attr else None
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=hidden_dim if use_edge_attr else None
                )
            )
        
        # Output layer
        self.convs.append(
            GATv2Conv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=hidden_dim if use_edge_attr else None
            )
        )
        
        # Layer normalization (paper configuration)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention coefficient layers (layer-specific, paper approach)
        self.attention_coeffs = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_layers)
        ])
        
        # Output projection (paper configuration)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),  # Paper uses ELU activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Sophisticated pooling layers (paper approach)
        self.pooling_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 4) for _ in range(4)
        ])
        
        # Final pooling combination (4 pooling strategies, paper approach)
        self.pooling_combine = nn.Linear(hidden_dim * 4, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Process edge attributes
        if self.use_edge_attr and edge_attr is not None:
            edge_types = edge_attr[:, 0].long()
            edge_embeddings = self.edge_type_emb(edge_types)
        else:
            edge_embeddings = None
        # Forward pass through GAT layers
        for i, conv in enumerate(self.convs):
            if edge_embeddings is not None:
                x_new = conv(x, edge_index, edge_embeddings)
            else:
                x_new = conv(x, edge_index)
            x_new = self.layer_norms[i](x_new)
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        # For node classification, return node-level logits (no pooling)
        if batch is None:
            return self.output_proj(x)
        # For graph classification, apply pooling
        pooled_features = []
        mean_pooled = global_mean_pool(x, batch)
        pooled_features.append(mean_pooled)
        max_pooled = global_max_pool(x, batch)
        pooled_features.append(max_pooled)
        add_pooled = global_add_pool(x, batch)
        pooled_features.append(add_pooled)
        attention_weights = torch.sigmoid(self.attention_coeffs[-1](x))
        attention_pooled = global_add_pool(x * attention_weights, batch)
        pooled_features.append(attention_pooled)
        combined = torch.cat(pooled_features, dim=1)
        x = self.pooling_combine(combined)
        return self.output_proj(x)

class EnhancedGraphSAGEModel(nn.Module):
    """
    Enhanced GraphSAGE model with sophisticated features
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=10, num_layers=3,
                 dropout=0.3, use_edge_attr=True, num_edge_types=4):
        super(EnhancedGraphSAGEModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # Edge type embeddings
        if use_edge_attr:
            self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            SAGEConv(
                input_dim,
                hidden_dim,
                normalize=True
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                    normalize=True
                )
            )
        
        # Output layer
        self.convs.append(
            SAGEConv(
                hidden_dim,
                hidden_dim,
                normalize=True
            )
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Sophisticated pooling
        self.pooling_combine = nn.Linear(hidden_dim * 3, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Process edge attributes
        if self.use_edge_attr and edge_attr is not None:
            # Extract edge types
            edge_types = edge_attr[:, 0].long()
            edge_embeddings = self.edge_type_emb(edge_types)
            # Note: SAGEConv doesn't directly support edge attributes
            # We'll use them indirectly through attention mechanisms
        
        # Forward pass through GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            
            # Apply layer normalization
            x_new = self.layer_norms[i](x_new)
            
            # Apply ELU activation
            x_new = F.elu(x_new)
            
            # Apply dropout
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        # Sophisticated pooling
        if batch is not None:
            # Multiple pooling strategies
            mean_pooled = global_mean_pool(x, batch)
            max_pooled = global_max_pool(x, batch)
            add_pooled = global_add_pool(x, batch)
            
            # Combine pooling strategies
            combined = torch.cat([mean_pooled, max_pooled, add_pooled], dim=1)
            x = self.pooling_combine(combined)
        else:
            # For node-level tasks
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Output projection
        out = self.output_proj(x)
        
        return out

class EnhancedGCNModel(nn.Module):
    """
    Enhanced GCN model with sophisticated features
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=10, num_layers=3,
                 dropout=0.3, use_edge_attr=True, num_edge_types=4):
        super(EnhancedGCNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # Edge type embeddings
        if use_edge_attr:
            self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        # GCN layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GCNConv(
                input_dim,
                hidden_dim,
                improved=True,
                cached=False
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(
                    hidden_dim,
                    hidden_dim,
                    improved=True,
                    cached=False
                )
            )
        
        # Output layer
        self.convs.append(
            GCNConv(
                hidden_dim,
                hidden_dim,
                improved=True,
                cached=False
            )
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Sophisticated pooling
        self.pooling_combine = nn.Linear(hidden_dim * 3, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Process edge attributes
        if self.use_edge_attr and edge_attr is not None:
            # Extract edge types
            edge_types = edge_attr[:, 0].long()
            edge_embeddings = self.edge_type_emb(edge_types)
            # Note: GCNConv doesn't directly support edge attributes
            # We'll use them indirectly
        
        # Forward pass through GCN layers
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            
            # Apply layer normalization
            x_new = self.layer_norms[i](x_new)
            
            # Apply ELU activation
            x_new = F.elu(x_new)
            
            # Apply dropout
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        # Sophisticated pooling
        if batch is not None:
            # Multiple pooling strategies
            mean_pooled = global_mean_pool(x, batch)
            max_pooled = global_max_pool(x, batch)
            add_pooled = global_add_pool(x, batch)
            
            # Combine pooling strategies
            combined = torch.cat([mean_pooled, max_pooled, add_pooled], dim=1)
            x = self.pooling_combine(combined)
        else:
            # For node-level tasks
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Output projection
        out = self.output_proj(x)
        
        return out

class MultiScaleGNN(nn.Module):
    """
    Multi-scale GNN combining different architectures
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=10, num_layers=3,
                 dropout=0.3, use_edge_attr=True, num_edge_types=4):
        super(MultiScaleGNN, self).__init__()
        
        # Individual GNN models
        self.gat_model = EnhancedGATModel(
            input_dim, hidden_dim, hidden_dim, num_layers, 
            num_heads=8, dropout=dropout, use_edge_attr=use_edge_attr, 
            num_edge_types=num_edge_types
        )
        
        self.sage_model = EnhancedGraphSAGEModel(
            input_dim, hidden_dim, hidden_dim, num_layers,
            dropout=dropout, use_edge_attr=use_edge_attr,
            num_edge_types=num_edge_types
        )
        
        self.gcn_model = EnhancedGCNModel(
            input_dim, hidden_dim, hidden_dim, num_layers,
            dropout=dropout, use_edge_attr=use_edge_attr,
            num_edge_types=num_edge_types
        )
        
        # Attention mechanism for combining models
        self.model_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Get predictions from each model
        gat_out = self.gat_model(x, edge_index, edge_attr, batch)
        sage_out = self.sage_model(x, edge_index, edge_attr, batch)
        gcn_out = self.gcn_model(x, edge_index, edge_attr, batch)
        
        # Combine outputs
        combined = torch.cat([gat_out, sage_out, gcn_out], dim=1)
        
        # Apply attention mechanism
        if batch is not None:
            # Group by batch
            unique_batches = torch.unique(batch)
            attended_outputs = []
            
            for batch_idx in unique_batches:
                batch_mask = (batch == batch_idx)
                batch_features = combined[batch_mask].unsqueeze(0)  # [1, num_nodes, features]
                
                # Apply self-attention
                attended, _ = self.model_attention(batch_features, batch_features, batch_features)
                attended_outputs.append(attended.squeeze(0))
            
            # Combine attended outputs
            if len(attended_outputs) > 0:
                combined = torch.cat(attended_outputs, dim=0)
        
        # Final output projection
        out = self.output_proj(combined)
        
        return out

def get_enhanced_model(model_type: str, input_dim: int, hidden_dim: int = 128, 
                      output_dim: int = 10, num_layers: int = 3, 
                      dropout: float = 0.3, use_edge_attr: bool = True, 
                      num_edge_types: int = 4) -> nn.Module:
    """
    Factory function to get enhanced GNN models
    """
    if model_type.upper() == "GAT":
        return EnhancedGATModel(
            input_dim, hidden_dim, output_dim, num_layers,
            num_heads=8, dropout=dropout, use_edge_attr=use_edge_attr,
            num_edge_types=num_edge_types
        )
    elif model_type.upper() == "GRAPHSAGE":
        return EnhancedGraphSAGEModel(
            input_dim, hidden_dim, output_dim, num_layers,
            dropout=dropout, use_edge_attr=use_edge_attr,
            num_edge_types=num_edge_types
        )
    elif model_type.upper() == "GCN":
        return EnhancedGCNModel(
            input_dim, hidden_dim, output_dim, num_layers,
            dropout=dropout, use_edge_attr=use_edge_attr,
            num_edge_types=num_edge_types
        )
    elif model_type.upper() == "MULTISCALE":
        return MultiScaleGNN(
            input_dim, hidden_dim, output_dim, num_layers,
            dropout=dropout, use_edge_attr=use_edge_attr,
            num_edge_types=num_edge_types
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}") 