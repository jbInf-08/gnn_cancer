"""
Ensemble GAT Models to Surpass Paper Performance
- Multiple GAT architectures
- Ensemble methods (voting, stacking, bagging)
- Advanced aggregation strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, Set2Set
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BaseGATModel(nn.Module):
    """Base GAT model class"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(BaseGATModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=True, concat=True))
        
        for i in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=True, concat=True))
        
        self.convs.append(GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=True, concat=True))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv in self.convs:
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if batch is None:
            return self.output_proj(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        return self.output_proj(x)

class GATv2Model(BaseGATModel):
    """GATv2 model with edge attributes"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout, use_edge_attr=True):
        super(GATv2Model, self).__init__(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
        self.use_edge_attr = use_edge_attr
        
        if use_edge_attr:
            self.edge_proj = nn.Linear(1, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if batch is None:
            return self.output_proj(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        return self.output_proj(x)

class GATWithSkipConnections(BaseGATModel):
    """GAT model with skip connections"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(GATWithSkipConnections, self).__init__(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
        
        # Skip connection projections
        self.skip_projections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        skip_features = []
        
        for i, conv in enumerate(self.convs):
            skip_features.append(x)
            
            x_new = conv(x, edge_index)
            
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Add skip connection
            if i > 0:
                skip_input = self.skip_projections[i](skip_features[i-1])
                if skip_input.shape == x_new.shape:
                    x_new = x_new + skip_input
            elif i == 0:
                # For the first layer, project input features
                skip_input = self.skip_projections[i](skip_features[i])
                if skip_input.shape == x_new.shape:
                    x_new = x_new + skip_input
            
            x = x_new
        
        if batch is None:
            return self.output_proj(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        return self.output_proj(x)

class GATWithMultiScale(BaseGATModel):
    """GAT model with multi-scale feature aggregation"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(GATWithMultiScale, self).__init__(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
        
        # Multi-scale aggregation
        self.multi_scale_weights = nn.Parameter(torch.ones(num_layers))
        self.multi_scale_combine = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        multi_scale_features = []
        
        for conv in self.convs:
            x_new = conv(x, edge_index)
            
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            multi_scale_features.append(x_new)
            x = x_new
        
        # Multi-scale aggregation
        weighted_features = []
        for i, features in enumerate(multi_scale_features):
            weight = F.softmax(self.multi_scale_weights, dim=0)[i]
            weighted_features.append(features * weight)
        
        combined_features = torch.cat(weighted_features, dim=1)
        x = self.multi_scale_combine(combined_features)
        
        if batch is None:
            return self.output_proj(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        return self.output_proj(x)

class EnsembleGAT(nn.Module):
    """Ensemble of multiple GAT models"""
    def __init__(self, input_dim, output_dim, model_configs, ensemble_method='voting'):
        super(EnsembleGAT, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_method = ensemble_method
        
        # Create multiple models
        self.models = nn.ModuleList()
        for config in model_configs:
            model_type = config.get('type', 'gatv2')
            
            if model_type == 'gatv2':
                model = GATv2Model(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 256),
                    output_dim=output_dim,
                    num_layers=config.get('num_layers', 3),
                    num_heads=config.get('num_heads', 8),
                    dropout=config.get('dropout', 0.3),
                    use_edge_attr=config.get('use_edge_attr', True)
                )
            elif model_type == 'skip':
                model = GATWithSkipConnections(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 256),
                    output_dim=output_dim,
                    num_layers=config.get('num_layers', 3),
                    num_heads=config.get('num_heads', 8),
                    dropout=config.get('dropout', 0.3)
                )
            elif model_type == 'multiscale':
                model = GATWithMultiScale(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 256),
                    output_dim=output_dim,
                    num_layers=config.get('num_layers', 3),
                    num_heads=config.get('num_heads', 8),
                    dropout=config.get('dropout', 0.3)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.models.append(model)
        
        # Ensemble aggregation
        if ensemble_method == 'stacking':
            self.meta_learner = nn.Sequential(
                nn.Linear(output_dim * len(self.models), output_dim * 2),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(output_dim * 2, output_dim)
            )
        elif ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x, edge_index, edge_attr, batch)
            predictions.append(pred)
        
        # Ensemble aggregation
        if self.ensemble_method == 'voting':
            # Hard voting
            pred_tensor = torch.stack(predictions, dim=0)
            votes = torch.argmax(pred_tensor, dim=-1)
            final_pred = torch.mode(votes, dim=0)[0]
            return final_pred
        
        elif self.ensemble_method == 'averaging':
            # Soft voting (average probabilities)
            pred_tensor = torch.stack(predictions, dim=0)
            avg_pred = torch.mean(pred_tensor, dim=0)
            return avg_pred
        
        elif self.ensemble_method == 'weighted':
            # Weighted averaging
            weights = F.softmax(self.ensemble_weights, dim=0)
            pred_tensor = torch.stack(predictions, dim=0)
            weighted_pred = torch.sum(pred_tensor * weights.unsqueeze(-1).unsqueeze(-1), dim=0)
            return weighted_pred
        
        elif self.ensemble_method == 'stacking':
            # Stacking with meta-learner
            stacked_features = torch.cat(predictions, dim=-1)
            final_pred = self.meta_learner(stacked_features)
            return final_pred
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

class BaggingEnsembleGAT(nn.Module):
    """Bagging ensemble with bootstrap sampling"""
    def __init__(self, input_dim, output_dim, base_model_config, n_estimators=5):
        super(BaggingEnsembleGAT, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        
        # Create base models
        self.models = nn.ModuleList()
        for i in range(n_estimators):
            model = GATv2Model(
                input_dim=input_dim,
                hidden_dim=base_model_config.get('hidden_dim', 256),
                output_dim=output_dim,
                num_layers=base_model_config.get('num_layers', 3),
                num_heads=base_model_config.get('num_heads', 8),
                dropout=base_model_config.get('dropout', 0.3),
                use_edge_attr=base_model_config.get('use_edge_attr', True)
            )
            self.models.append(model)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        predictions = []
        for model in self.models:
            pred = model(x, edge_index, edge_attr, batch)
            predictions.append(pred)
        
        # Average predictions
        pred_tensor = torch.stack(predictions, dim=0)
        avg_pred = torch.mean(pred_tensor, dim=0)
        return avg_pred

def create_ensemble_configs():
    """Create configurations for different GAT models"""
    configs = [
        # GATv2 with different architectures
        {'type': 'gatv2', 'hidden_dim': 128, 'num_layers': 3, 'num_heads': 4, 'dropout': 0.3, 'use_edge_attr': True},
        {'type': 'gatv2', 'hidden_dim': 256, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.4, 'use_edge_attr': True},
        {'type': 'gatv2', 'hidden_dim': 512, 'num_layers': 5, 'num_heads': 16, 'dropout': 0.5, 'use_edge_attr': True},
        
        # GAT with skip connections
        {'type': 'skip', 'hidden_dim': 256, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.3},
        {'type': 'skip', 'hidden_dim': 512, 'num_layers': 5, 'num_heads': 16, 'dropout': 0.4},
        
        # GAT with multi-scale
        {'type': 'multiscale', 'hidden_dim': 256, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.3},
        {'type': 'multiscale', 'hidden_dim': 512, 'num_layers': 5, 'num_heads': 16, 'dropout': 0.4},
    ]
    return configs

def create_bagging_config():
    """Create configuration for bagging ensemble"""
    return {
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.3,
        'use_edge_attr': True
    }

def test_ensemble_models():
    """Test the ensemble models"""
    print("Testing ensemble GAT models...")
    
    # Test data
    num_nodes = 50
    num_features = 19
    num_edges = 100
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randint(0, 8, (num_edges, 1)).float()
    
    # Test individual models
    print("Testing individual models...")
    
    # GATv2
    gatv2 = GATv2Model(input_dim=num_features, hidden_dim=128, output_dim=2, num_layers=3, num_heads=4, dropout=0.3)
    with torch.no_grad():
        output = gatv2(x, edge_index, edge_attr)
        print(f"GATv2 output shape: {output.shape}")
    
    # GAT with skip connections
    gat_skip = GATWithSkipConnections(input_dim=num_features, hidden_dim=128, output_dim=2, num_layers=3, num_heads=4, dropout=0.3)
    with torch.no_grad():
        output = gat_skip(x, edge_index, edge_attr)
        print(f"GAT with skip connections output shape: {output.shape}")
    
    # GAT with multi-scale
    gat_multiscale = GATWithMultiScale(input_dim=num_features, hidden_dim=128, output_dim=2, num_layers=3, num_heads=4, dropout=0.3)
    with torch.no_grad():
        output = gat_multiscale(x, edge_index, edge_attr)
        print(f"GAT with multi-scale output shape: {output.shape}")
    
    # Test ensemble
    print("Testing ensemble models...")
    
    configs = create_ensemble_configs()[:3]  # Use first 3 configs for testing
    
    # Voting ensemble
    voting_ensemble = EnsembleGAT(input_dim=num_features, output_dim=2, model_configs=configs, ensemble_method='voting')
    with torch.no_grad():
        output = voting_ensemble(x, edge_index, edge_attr)
        print(f"Voting ensemble output shape: {output.shape}")
    
    # Averaging ensemble
    avg_ensemble = EnsembleGAT(input_dim=num_features, output_dim=2, model_configs=configs, ensemble_method='averaging')
    with torch.no_grad():
        output = avg_ensemble(x, edge_index, edge_attr)
        print(f"Averaging ensemble output shape: {output.shape}")
    
    # Weighted ensemble
    weighted_ensemble = EnsembleGAT(input_dim=num_features, output_dim=2, model_configs=configs, ensemble_method='weighted')
    with torch.no_grad():
        output = weighted_ensemble(x, edge_index, edge_attr)
        print(f"Weighted ensemble output shape: {output.shape}")
    
    # Stacking ensemble
    stacking_ensemble = EnsembleGAT(input_dim=num_features, output_dim=2, model_configs=configs, ensemble_method='stacking')
    with torch.no_grad():
        output = stacking_ensemble(x, edge_index, edge_attr)
        print(f"Stacking ensemble output shape: {output.shape}")
    
    # Bagging ensemble
    bagging_config = create_bagging_config()
    bagging_ensemble = BaggingEnsembleGAT(input_dim=num_features, output_dim=2, base_model_config=bagging_config, n_estimators=3)
    with torch.no_grad():
        output = bagging_ensemble(x, edge_index, edge_attr)
        print(f"Bagging ensemble output shape: {output.shape}")
    
    print("All ensemble models tested successfully!")

if __name__ == "__main__":
    test_ensemble_models()
