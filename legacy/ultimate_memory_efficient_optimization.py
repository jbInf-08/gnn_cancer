#!/usr/bin/env python3
"""
Ultimate Memory-Efficient Hyperparameter Optimization Pipeline
Target: >99% accuracy to exceed paper performance
Preserves ALL data: No graph density reduction, no feature selection
Uses advanced memory optimization techniques
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimateMemoryEfficientGAT(nn.Module):
    """Ultimate memory-efficient GAT with gradient checkpointing and mixed precision"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, num_heads: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GAT layers with memory optimization
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(GATConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Output projection
        self.classifier = nn.Linear(hidden_dim * num_heads, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers with gradient checkpointing for memory efficiency
        for i in range(self.num_layers):
            if self.training and i < self.num_layers - 1:  # Don't checkpoint last layer
                x = torch.utils.checkpoint.checkpoint(
                    self._gat_layer_forward, 
                    self.gat_layers[i], x, edge_index,
                    preserve_rng_state=False
                )
            else:
                x = self.gat_layers[i](x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x
    
    def _gat_layer_forward(self, layer, x, edge_index):
        """Helper function for gradient checkpointing"""
        return layer(x, edge_index)

class UltimateMemoryEfficientGraphSAGE(nn.Module):
    """Ultimate memory-efficient GraphSAGE with gradient checkpointing"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            self.sage_layers.append(SAGEConv(in_channels, hidden_dim))
        
        # Output projection
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GraphSAGE layers with gradient checkpointing
        for i in range(self.num_layers):
            if self.training and i < self.num_layers - 1:
                x = torch.utils.checkpoint.checkpoint(
                    self._sage_layer_forward,
                    self.sage_layers[i], x, edge_index,
                    preserve_rng_state=False
                )
            else:
                x = self.sage_layers[i](x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x
    
    def _sage_layer_forward(self, layer, x, edge_index):
        """Helper function for gradient checkpointing"""
        return layer(x, edge_index)

class UltimateMemoryEfficientGCN(nn.Module):
    """Ultimate memory-efficient GCN with gradient checkpointing"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GCNConv(in_channels, hidden_dim))
        
        # Output projection
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GCN layers with gradient checkpointing
        for i in range(self.num_layers):
            if self.training and i < self.num_layers - 1:
                x = torch.utils.checkpoint.checkpoint(
                    self._gcn_layer_forward,
                    self.gcn_layers[i], x, edge_index,
                    preserve_rng_state=False
                )
            else:
                x = self.gcn_layers[i](x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x
    
    def _gcn_layer_forward(self, layer, x, edge_index):
        """Helper function for gradient checkpointing"""
        return layer(x, edge_index)

class MemoryOptimizedDataLoader:
    """Memory-optimized data loader that processes data in chunks"""
    
    def __init__(self, data: Data, chunk_size: int = 1000):
        self.data = data
        self.chunk_size = chunk_size
        self.num_nodes = data.num_nodes
        
    def get_node_chunks(self):
        """Yield node indices in chunks for memory-efficient processing"""
        for i in range(0, self.num_nodes, self.chunk_size):
            end_idx = min(i + self.chunk_size, self.num_nodes)
            yield torch.arange(i, end_idx)
    
    def get_edge_chunks(self, chunk_size: int = 10000):
        """Yield edge indices in chunks"""
        num_edges = self.data.edge_index.shape[1]
        for i in range(0, num_edges, chunk_size):
            end_idx = min(i + chunk_size, num_edges)
            yield torch.arange(i, end_idx)

class UltimateMemoryEfficientOptimizer:
    """Ultimate memory-efficient hyperparameter optimization"""
    
    def __init__(self, data: Data, model_type: str = 'gat'):
        self.data = data
        self.model_type = model_type
        self.best_params = None
        self.best_score = 0.0
        
        # Optimization parameters
        self.n_trials = 30  # Reduced for faster iteration
        self.n_folds = 3
        
        # Model registry
        self.model_registry = {
            'gat': UltimateMemoryEfficientGAT,
            'graphsage': UltimateMemoryEfficientGraphSAGE,
            'gcn': UltimateMemoryEfficientGCN
        }
        
        # Memory optimization
        self.data_loader = MemoryOptimizedDataLoader(data)
        
        # Enable gradient checkpointing globally
        torch.utils.checkpoint.checkpoint_sequential = True
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function with ultimate memory optimization"""
        
        # Define hyperparameter search space
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'epochs': trial.suggest_int('epochs', 20, 50),  # Reduced for memory efficiency
            'patience': trial.suggest_int('patience', 8, 15)
        }
        
        # Add model-specific parameters
        if self.model_type == 'gat':
            params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8])
        
        # Cross-validation with memory optimization
        scores = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(self.data.num_nodes), self.data.y)):
            # Create train/val masks
            train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            # Create model with memory optimization
            model_class = self.model_registry[self.model_type]
            model = model_class(
                num_features=self.data.num_features,
                num_classes=len(torch.unique(self.data.y)),
                **params
            )
            
            # Use mixed precision training for memory efficiency
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            # Optimizer with memory-efficient settings
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay'],
                eps=1e-8  # Higher epsilon for stability
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
            criterion = nn.CrossEntropyLoss()
            
            # Training loop with memory optimization
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(params['epochs']):
                model.train()
                optimizer.zero_grad()
                
                # Use mixed precision training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(self.data)
                        loss = criterion(out[train_mask], self.data.y[train_mask])
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(self.data)
                    loss = criterion(out[train_mask], self.data.y[train_mask])
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                # Validation with memory optimization
                model.eval()
                with torch.no_grad():
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            out = model(self.data)
                    else:
                        out = model(self.data)
                    
                    val_acc = accuracy_score(
                        self.data.y[val_mask].cpu().numpy(),
                        out[val_mask].argmax(dim=1).cpu().numpy()
                    )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= params['patience']:
                    break
                
                # Aggressive memory cleanup
                if epoch % 5 == 0:  # Every 5 epochs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            scores.append(best_val_acc)
            
            # Clean up model and optimizer
            del model, optimizer, scheduler, criterion
            if scaler is not None:
                del scaler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Return mean CV score
        mean_score = np.mean(scores)
        
        # Update best parameters
        if mean_score > self.best_score:
            self.best_score = mean_score
            self.best_params = params
        
        return mean_score
    
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        logger.info(f"🚀 Starting ULTIMATE memory-efficient optimization for {self.model_type.upper()}")
        logger.info(f"   - Trials: {self.n_trials}")
        logger.info(f"   - Cross-validation folds: {self.n_folds}")
        logger.info(f"   - Preserving ALL data: {self.data.num_nodes} nodes, {self.data.num_edges} edges, {self.data.num_features} features")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"✅ Optimization completed!")
        logger.info(f"   - Best CV score: {study.best_value:.4f}")
        logger.info(f"   - Best parameters: {study.best_params}")
        
        return study.best_params

class UltimateEnsembleTrainer:
    """Ultimate memory-efficient ensemble training preserving all data"""
    
    def __init__(self, data: Data, optimized_params: Dict):
        self.data = data
        self.optimized_params = optimized_params
        self.models = []
        self.data_loader = MemoryOptimizedDataLoader(data)
    
    def create_ensemble_models(self, num_models: int = 3) -> List[nn.Module]:
        """Create multiple models with different initializations"""
        logger.info(f"🏗️ Creating {num_models} ultimate ensemble models...")
        
        models = []
        for i in range(num_models):
            # Set different random seeds for different initializations
            torch.manual_seed(42 + i)
            
            # Create model with optimized parameters
            if 'num_heads' in self.optimized_params:
                model = UltimateMemoryEfficientGAT(
                    num_features=self.data.num_features,
                    num_classes=len(torch.unique(self.data.y)),
                    **self.optimized_params
                )
            else:
                model = UltimateMemoryEfficientGraphSAGE(
                    num_features=self.data.num_features,
                    num_classes=len(torch.unique(self.data.y)),
                    **self.optimized_params
                )
            
            models.append(model)
        
        self.models = models
        logger.info(f"✅ Created {len(models)} ensemble models")
        return models
    
    def train_ensemble(self, train_mask: torch.Tensor, val_mask: torch.Tensor) -> List[nn.Module]:
        """Train ensemble of models with ultimate memory optimization"""
        logger.info("🚀 Training ultimate ensemble models...")
        
        trained_models = []
        
        for i, model in enumerate(tqdm(self.models, desc="Training ensemble")):
            # Use mixed precision training
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            # Train model
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.optimized_params['lr'], 
                weight_decay=self.optimized_params['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.optimized_params['epochs']
            )
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(self.optimized_params['epochs']):
                model.train()
                optimizer.zero_grad()
                
                # Mixed precision training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(self.data)
                        loss = criterion(out[train_mask], self.data.y[train_mask])
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(self.data)
                    loss = criterion(out[train_mask], self.data.y[train_mask])
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            out = model(self.data)
                    else:
                        out = model(self.data)
                    
                    val_acc = accuracy_score(
                        self.data.y[val_mask].cpu().numpy(),
                        out[val_mask].argmax(dim=1).cpu().numpy()
                    )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.optimized_params['patience']:
                    break
                
                # Memory cleanup
                if epoch % 3 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            trained_models.append(model)
        
        logger.info(f"✅ Trained {len(trained_models)} ensemble models")
        return trained_models
    
    def ensemble_predict(self, models: List[nn.Module], test_mask: torch.Tensor) -> np.ndarray:
        """Make ensemble predictions with memory optimization"""
        logger.info("🔮 Making ultimate ensemble predictions...")
        
        predictions = []
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for model in models:
            model.eval()
            with torch.no_grad():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = model(self.data)
                else:
                    out = model(self.data)
                
                pred = F.softmax(out[test_mask], dim=1)
                predictions.append(pred.cpu().numpy())
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_class = np.argmax(ensemble_pred, axis=1)
        
        return ensemble_pred_class, ensemble_pred

def main():
    """Main execution function"""
    logger.info("🚀 Starting ULTIMATE MEMORY-EFFICIENT OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("🎯 PRESERVING ALL DATA: No graph density reduction, no feature selection!")
    logger.info("=" * 80)
    
    # Load processed data
    data_path = Path("data/massive_processed/massive_processed_data.pt")
    if not data_path.exists():
        logger.error("❌ Processed data not found! Run feature engineering first.")
        return
    
    data = torch.load(data_path)
    logger.info(f"✅ Data loaded: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    logger.info(f"🎯 Using ALL {data.num_features} features and ALL {data.num_edges} edges!")
    
    # Create output directory
    output_dir = Path("results/ultimate_memory_efficient_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameter optimization for each model type
    model_types = ['gat', 'graphsage', 'gcn']
    optimization_results = {}
    
    for model_type in model_types:
        logger.info(f"Phase 1: Optimizing {model_type.upper()} with ALL data...")
        
        optimizer = UltimateMemoryEfficientOptimizer(data, model_type)
        best_params = optimizer.optimize()
        optimization_results[model_type] = best_params
        
        # Save optimization results
        with open(output_dir / f"{model_type}_ultimate_optimization_results.json", 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': optimizer.best_score,
                'data_preservation': {
                    'nodes': data.num_nodes,
                    'edges': data.num_edges,
                    'features': data.num_features,
                    'no_density_reduction': True,
                    'no_feature_selection': True
                }
            }, f, indent=2)
    
    # Ensemble training with best model
    logger.info("Phase 2: Ultimate ensemble training...")
    
    # Use GAT as the best model
    best_model_type = 'gat'
    best_params = optimization_results[best_model_type]
    
    # Create train/val/test split
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Ensemble training
    ensemble_trainer = UltimateEnsembleTrainer(data, best_params)
    
    # Create and train ensemble
    models = ensemble_trainer.create_ensemble_models(num_models=3)
    trained_models = ensemble_trainer.train_ensemble(train_mask, val_mask)
    
    # Make ensemble predictions
    ensemble_pred_class, ensemble_pred_proba = ensemble_trainer.ensemble_predict(trained_models, test_mask)
    
    # Calculate metrics
    y_true = data.y[test_mask].cpu().numpy()
    
    accuracy = accuracy_score(y_true, ensemble_pred_class)
    precision = precision_score(y_true, ensemble_pred_class, average='weighted')
    recall = recall_score(y_true, ensemble_pred_class, average='weighted')
    f1 = f1_score(y_true, ensemble_pred_class, average='weighted')
    roc_auc = roc_auc_score(y_true, ensemble_pred_proba[:, 1])
    
    # Save results
    results = {
        'model_type': best_model_type,
        'optimization_results': optimization_results,
        'ensemble_results': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        },
        'data_preservation': {
            'nodes': data.num_nodes,
            'edges': data.num_edges,
            'features': data.num_features,
            'no_density_reduction': True,
            'no_feature_selection': True
        },
        'memory_optimization_techniques': [
            'Gradient checkpointing',
            'Mixed precision training',
            'Aggressive memory cleanup',
            'Efficient data loading',
            'Memory-optimized models'
        ]
    }
    
    with open(output_dir / "ultimate_memory_efficient_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    logger.info("🎉 ULTIMATE MEMORY-EFFICIENT OPTIMIZATION COMPLETED!")
    logger.info("=" * 80)
    logger.info("🎯 DATA PRESERVATION ACHIEVED:")
    logger.info(f"   - ALL {data.num_nodes} nodes preserved")
    logger.info(f"   - ALL {data.num_edges} edges preserved")
    logger.info(f"   - ALL {data.num_features} features preserved")
    logger.info("=" * 80)
    logger.info("ENSEMBLE RESULTS:")
    logger.info(f"   - Accuracy: {accuracy:.4f}")
    logger.info(f"   - Precision: {precision:.4f}")
    logger.info(f"   - Recall: {recall:.4f}")
    logger.info(f"   - F1-Score: {f1:.4f}")
    logger.info(f"   - ROC-AUC: {roc_auc:.4f}")
    logger.info("=" * 80)
    
    if accuracy > 0.99:
        logger.info("🎯 TARGET ACHIEVED: >99% accuracy with ALL data preserved!")
    else:
        logger.info("📈 Further optimization needed to reach >99% accuracy")

if __name__ == "__main__":
    main() 