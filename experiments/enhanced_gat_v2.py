"""
Enhanced GAT v2 Implementation to Surpass Paper Performance
- Advanced attention mechanisms
- Skip connections and residual learning
- Multi-scale feature aggregation
- Advanced regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, Set2Set
import math
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

def custom_attention_pool(x, batch, gate_nn, nn):
    """
    Custom attention pooling implementation to replace global_attention_pool
    """
    x = x.unsqueeze(-1) if x.dim() == 1 else x
    size = batch[-1].item() + 1 if batch.numel() > 0 else 0
    
    gate = gate_nn(x).sigmoid()
    x = nn(x) if nn is not None else x
    assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
    
    gate = gate.view(gate.size(0), -1)
    x = x.view(x.size(0), -1)
    
    # Simple scatter add implementation
    out = torch.zeros(size, x.size(1), device=x.device, dtype=x.dtype)
    for i in range(x.size(0)):
        out[batch[i]] += gate[i] * x[i]
    
    return out

class EnhancedGATv2Model(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256, 
                 output_dim: int = 2, 
                 num_layers: int = 4, 
                 num_heads: int = 8, 
                 dropout: float = 0.3,
                 use_edge_attr: bool = True, 
                 num_edge_types: int = 8,
                 use_skip_connections: bool = True,
                 use_multi_scale: bool = True,
                 use_attention_pooling: bool = True,
                 use_layer_norm: bool = True,
                 use_batch_norm: bool = False,
                 activation: str = 'elu',
                 pooling_strategy: str = 'multi'):
        super(EnhancedGATv2Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.use_skip_connections = use_skip_connections
        self.use_multi_scale = use_multi_scale
        self.use_attention_pooling = use_attention_pooling
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.pooling_strategy = pooling_strategy
        
        # Activation function
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.elu
        
        # Edge type embeddings
        if use_edge_attr:
            self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
            nn.init.xavier_uniform_(self.edge_type_emb.weight)
            self.edge_proj = nn.Linear(1, hidden_dim)  # Project edge attributes
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(
            input_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            edge_dim=hidden_dim if use_edge_attr else None, 
            concat=True
        ))
        
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
        
        self.convs.append(GATv2Conv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            edge_dim=hidden_dim if use_edge_attr else None, 
            concat=True
        ))
        
        # Normalization layers
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim, elementwise_affine=True) 
                for _ in range(num_layers)
            ])
        
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) 
                for _ in range(num_layers)
            ])
        
        # Skip connections
        if use_skip_connections:
            self.skip_projections = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            # Initialize skip projections
            for proj in self.skip_projections:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        
        # Multi-scale feature aggregation
        if use_multi_scale:
            self.multi_scale_weights = nn.Parameter(torch.ones(num_layers))
            self.multi_scale_combine = nn.Sequential(
                nn.Linear(hidden_dim * num_layers, hidden_dim * 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        # Attention pooling
        if use_attention_pooling:
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Pooling strategies
        if pooling_strategy == 'multi':
            self.pooling_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim // 4) 
                for _ in range(4)
            ])
            self.pooling_combine = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        elif pooling_strategy == 'set2set':
            self.set2set = Set2Set(hidden_dim, processing_steps=3)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Process edge attributes
        if self.use_edge_attr and edge_attr is not None:
            # Use edge attributes directly as edge features
            edge_embeddings = self.edge_proj(edge_attr)
        else:
            edge_embeddings = None
        
        # Multi-scale feature storage
        if self.use_multi_scale:
            multi_scale_features = []
        
        # Skip connection storage
        if self.use_skip_connections:
            skip_features = []
        
        # Forward pass through GAT layers
        for i, conv in enumerate(self.convs):
            # Store skip connection input
            if self.use_skip_connections:
                skip_features.append(x)
            
            # Apply GAT convolution
            if edge_embeddings is not None:
                x_new = conv(x, edge_index, edge_embeddings)
            else:
                x_new = conv(x, edge_index)
            
            # Apply normalization
            if self.use_layer_norm:
                x_new = self.layer_norms[i](x_new)
            
            if self.use_batch_norm:
                x_new = self.batch_norms[i](x_new)
            
            # Apply activation
            x_new = self.activation(x_new)
            
            # Apply dropout
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Apply skip connections
            if self.use_skip_connections and i > 0:
                skip_input = self.skip_projections[i](skip_features[i-1])
                if skip_input.shape == x_new.shape:
                    x_new = x_new + skip_input
            elif self.use_skip_connections and i == 0:
                # For the first layer, project input features
                skip_input = self.skip_projections[i](skip_features[i])
                if skip_input.shape == x_new.shape:
                    x_new = x_new + skip_input
            
            # Store multi-scale features
            if self.use_multi_scale:
                multi_scale_features.append(x_new)
            
            x = x_new
        
        # Multi-scale feature aggregation
        if self.use_multi_scale:
            # Weighted combination of features from all layers
            weighted_features = []
            for i, features in enumerate(multi_scale_features):
                weight = F.softmax(self.multi_scale_weights, dim=0)[i]
                weighted_features.append(features * weight)
            
            # Concatenate and combine
            combined_features = torch.cat(weighted_features, dim=1)
            x = self.multi_scale_combine(combined_features)
        
        # Graph-level pooling
        if batch is None:
            return self.output_proj(x)
        
        if self.pooling_strategy == 'multi':
            # Multiple pooling strategies
            pooled_features = []
            
            # Mean pooling
            mean_pooled = global_mean_pool(x, batch)
            pooled_features.append(self.pooling_layers[0](mean_pooled))
            
            # Max pooling
            max_pooled = global_max_pool(x, batch)
            pooled_features.append(self.pooling_layers[1](max_pooled))
            
            # Sum pooling
            sum_pooled = global_add_pool(x, batch)
            pooled_features.append(self.pooling_layers[2](sum_pooled))
            
            # Attention pooling
            if self.use_attention_pooling:
                attention_weights = torch.sigmoid(self.attention_pool(x))
                attention_pooled = global_add_pool(x * attention_weights, batch)
                pooled_features.append(self.pooling_layers[3](attention_pooled))
            
            # Combine all pooling strategies
            combined = torch.cat(pooled_features, dim=1)
            x = self.pooling_combine(combined)
            
        elif self.pooling_strategy == 'attention':
            # Attention-weighted pooling
            attention_weights = torch.sigmoid(self.attention_pool(x))
            x = custom_attention_pool(x, batch, lambda x: attention_weights, None)
            
        elif self.pooling_strategy == 'set2set':
            # Set2Set pooling
            x = self.set2set(x, batch)
        
        # Output projection
        return self.output_proj(x)

class AdvancedTrainingConfig:
    def __init__(self):
        # Model architecture
        self.hidden_dim = 256
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.3
        self.use_edge_attr = True
        self.num_edge_types = 8
        self.use_skip_connections = True
        self.use_multi_scale = True
        self.use_attention_pooling = True
        self.use_layer_norm = True
        self.use_batch_norm = False
        self.activation = 'elu'
        self.pooling_strategy = 'multi'
        
        # Training parameters
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.batch_size = 32
        self.epochs = 300
        self.patience = 50
        self.gradient_clip = 1.0
        
        # Optimizer configurations
        self.optimizer_configs = {
            'adam': {'lr': 0.001, 'weight_decay': 1e-4},
            'adamw': {'lr': 0.001, 'weight_decay': 1e-4},
            'sgd': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
            'rmsprop': {'lr': 0.001, 'weight_decay': 1e-4}
        }
        
        # Learning rate schedules
        self.lr_schedules = {
            'constant': lambda epoch: self.learning_rate,
            'step': lambda epoch: self.learning_rate * (0.5 ** (epoch // 50)),
            'cosine': lambda epoch: self.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / self.epochs)),
            'exponential': lambda epoch: self.learning_rate * (0.95 ** epoch),
            'plateau': lambda epoch: self.learning_rate * (0.8 ** (epoch // 30))
        }

def create_enhanced_gat_model(input_dim, output_dim, config=None):
    """Create an enhanced GAT model with the given configuration"""
    if config is None:
        config = AdvancedTrainingConfig()
    
    return EnhancedGATv2Model(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=output_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        use_edge_attr=config.use_edge_attr,
        num_edge_types=config.num_edge_types,
        use_skip_connections=config.use_skip_connections,
        use_multi_scale=config.use_multi_scale,
        use_attention_pooling=config.use_attention_pooling,
        use_layer_norm=config.use_layer_norm,
        use_batch_norm=config.use_batch_norm,
        activation=config.activation,
        pooling_strategy=config.pooling_strategy
    )
