#!/usr/bin/env python3
"""
Hyperparameter Optimization Pipeline for Superior GNN Performance
Target: >99% accuracy to exceed paper performance
Uses Bayesian optimization for efficient hyperparameter search
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
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedGAT(nn.Module):
    """Optimized GAT with hyperparameter tuning"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, num_heads: int, dropout: float, lr: float, 
                 weight_decay: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GAT layers
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
        
        # GAT layers
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class OptimizedGraphSAGE(nn.Module):
    """Optimized GraphSAGE with hyperparameter tuning"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, dropout: float, lr: float, weight_decay: float, **kwargs):
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
        
        # GraphSAGE layers
        for i in range(self.num_layers):
            x = self.sage_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class OptimizedGCN(nn.Module):
    """Optimized GCN with hyperparameter tuning"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, dropout: float, lr: float, weight_decay: float, **kwargs):
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
        
        # GCN layers
        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization for GNN models"""
    
    def __init__(self, data: Data, model_type: str = 'gat'):
        self.data = data
        self.model_type = model_type
        self.best_params = None
        self.best_score = 0.0
        
        # Optimization parameters
        self.n_trials = 100
        self.n_folds = 5
        
        # Model registry
        self.model_registry = {
            'gat': OptimizedGAT,
            'graphsage': OptimizedGraphSAGE,
            'gcn': OptimizedGCN
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        
        # Define hyperparameter search space
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'epochs': trial.suggest_int('epochs', 50, 200),
            'patience': trial.suggest_int('patience', 10, 30)
        }
        
        # Add model-specific parameters
        if self.model_type == 'gat':
            params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 16, 32])
        
        # Cross-validation
        scores = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(self.data.num_nodes), self.data.y)):
            # Create train/val masks
            train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            # Create model
            model_class = self.model_registry[self.model_type]
            model = model_class(
                num_features=self.data.num_features,
                num_classes=len(torch.unique(self.data.y)),
                **params
            )
            
            # Optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(params['epochs']):
                model.train()
                optimizer.zero_grad()
                
                out = model(self.data)
                loss = criterion(out[train_mask], self.data.y[train_mask])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
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
            
            scores.append(best_val_acc)
        
        # Return mean CV score
        mean_score = np.mean(scores)
        
        # Update best parameters
        if mean_score > self.best_score:
            self.best_score = mean_score
            self.best_params = params
        
        return mean_score
    
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        logger.info(f"🚀 Starting hyperparameter optimization for {self.model_type.upper()}")
        logger.info(f"   - Trials: {self.n_trials}")
        logger.info(f"   - Cross-validation folds: {self.n_folds}")
        
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

class AdvancedDataAugmentation:
    """Advanced data augmentation techniques for GNN training"""
    
    def __init__(self, data: Data):
        self.data = data
        self.augmented_data = []
    
    def feature_noise_augmentation(self, noise_std: float = 0.1) -> Data:
        """Add Gaussian noise to node features"""
        augmented_x = self.data.x + torch.randn_like(self.data.x) * noise_std
        return Data(
            x=augmented_x,
            edge_index=self.data.edge_index,
            edge_attr=self.data.edge_attr,
            y=self.data.y
        )
    
    def edge_dropout_augmentation(self, dropout_rate: float = 0.1) -> Data:
        """Randomly drop edges"""
        num_edges = self.data.edge_index.shape[1]
        keep_mask = torch.rand(num_edges) > dropout_rate
        
        new_edge_index = self.data.edge_index[:, keep_mask]
        new_edge_attr = self.data.edge_attr[keep_mask] if self.data.edge_attr is not None else None
        
        return Data(
            x=self.data.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=self.data.y
        )
    
    def feature_masking_augmentation(self, mask_rate: float = 0.1) -> Data:
        """Randomly mask node features"""
        mask = torch.rand_like(self.data.x) > mask_rate
        augmented_x = self.data.x * mask
        return Data(
            x=augmented_x,
            edge_index=self.data.edge_index,
            edge_attr=self.data.edge_attr,
            y=self.data.y
        )
    
    def generate_augmented_datasets(self, num_augmentations: int = 5) -> List[Data]:
        """Generate multiple augmented datasets"""
        logger.info(f"🔄 Generating {num_augmentations} augmented datasets...")
        
        augmented_datasets = []
        
        for i in range(num_augmentations):
            # Random augmentation parameters
            noise_std = np.random.uniform(0.05, 0.15)
            dropout_rate = np.random.uniform(0.05, 0.15)
            mask_rate = np.random.uniform(0.05, 0.15)
            
            # Apply augmentations
            aug_data = self.feature_noise_augmentation(noise_std)
            aug_data = self.edge_dropout_augmentation(dropout_rate)
            aug_data = self.feature_masking_augmentation(mask_rate)
            
            augmented_datasets.append(aug_data)
        
        logger.info(f"✅ Generated {len(augmented_datasets)} augmented datasets")
        return augmented_datasets

class FeatureSelector:
    """Feature selection to focus on most predictive features"""
    
    def __init__(self, data: Data):
        self.data = data
        self.selected_features = None
        self.feature_importance = None
    
    def calculate_feature_importance(self) -> np.ndarray:
        """Calculate feature importance using correlation with target"""
        logger.info("🔍 Calculating feature importance...")
        
        X = self.data.x.numpy()
        y = self.data.y.numpy()
        
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        self.feature_importance = np.array(correlations)
        return self.feature_importance
    
    def select_top_features(self, top_k: int = 500) -> Data:
        """Select top-k most important features"""
        if self.feature_importance is None:
            self.calculate_feature_importance()
        
        # Get top-k feature indices
        top_indices = np.argsort(self.feature_importance)[-top_k:]
        self.selected_features = top_indices
        
        # Create new data with selected features
        selected_x = self.data.x[:, top_indices]
        
        logger.info(f"✅ Selected top {top_k} features")
        logger.info(f"   - Original features: {self.data.num_features}")
        logger.info(f"   - Selected features: {selected_x.shape[1]}")
        
        return Data(
            x=selected_x,
            edge_index=self.data.edge_index,
            edge_attr=self.data.edge_attr,
            y=self.data.y
        )

class EnsembleTrainer:
    """Ensemble training combining multiple models and augmentations"""
    
    def __init__(self, data: Data, optimized_params: Dict):
        self.data = data
        self.optimized_params = optimized_params
        self.models = []
        self.augmented_datasets = []
        
    def create_ensemble_models(self, num_models: int = 5) -> List[nn.Module]:
        """Create multiple models with different initializations"""
        logger.info(f"🏗️ Creating {num_models} ensemble models...")
        
        models = []
        for i in range(num_models):
            # Set different random seeds for different initializations
            torch.manual_seed(42 + i)
            
            # Create model with optimized parameters
            if 'num_heads' in self.optimized_params:
                model = OptimizedGAT(
                    num_features=self.data.num_features,
                    num_classes=len(torch.unique(self.data.y)),
                    **self.optimized_params
                )
            else:
                model = OptimizedGraphSAGE(
                    num_features=self.data.num_features,
                    num_classes=len(torch.unique(self.data.y)),
                    **self.optimized_params
                )
            
            models.append(model)
        
        self.models = models
        logger.info(f"✅ Created {len(models)} ensemble models")
        return models
    
    def train_ensemble(self, train_mask: torch.Tensor, val_mask: torch.Tensor) -> List[nn.Module]:
        """Train ensemble of models"""
        logger.info("🚀 Training ensemble models...")
        
        trained_models = []
        
        for i, model in enumerate(tqdm(self.models, desc="Training ensemble")):
            # Create augmented dataset for this model
            if i < len(self.augmented_datasets):
                train_data = self.augmented_datasets[i]
            else:
                train_data = self.data
            
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
                
                out = model(train_data)
                loss = criterion(out[train_mask], train_data.y[train_mask])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    out = model(self.data)  # Use original data for validation
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
            
            trained_models.append(model)
        
        logger.info(f"✅ Trained {len(trained_models)} ensemble models")
        return trained_models
    
    def ensemble_predict(self, models: List[nn.Module], test_mask: torch.Tensor) -> np.ndarray:
        """Make ensemble predictions"""
        logger.info("🔮 Making ensemble predictions...")
        
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                out = model(self.data)
                pred = F.softmax(out[test_mask], dim=1)
                predictions.append(pred.cpu().numpy())
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_class = np.argmax(ensemble_pred, axis=1)
        
        return ensemble_pred_class, ensemble_pred

def main():
    """Main execution function"""
    logger.info("🚀 Starting HYPERPARAMETER OPTIMIZATION PIPELINE")
    logger.info("=" * 80)
    
    # Load processed data
    data_path = Path("data/massive_processed/massive_processed_data.pt")
    if not data_path.exists():
        logger.error("❌ Processed data not found! Run feature engineering first.")
        return
    
    data = torch.load(data_path)
    logger.info(f"✅ Data loaded: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    
    # Create output directory
    output_dir = Path("results/hyperparameter_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    feature_selector = FeatureSelector(data)
    data_augmenter = AdvancedDataAugmentation(data)
    
    # Feature selection
    logger.info("Phase 1: Feature selection...")
    selected_data = feature_selector.select_top_features(top_k=500)
    
    # Data augmentation
    logger.info("Phase 2: Data augmentation...")
    augmented_datasets = data_augmenter.generate_augmented_datasets(num_augmentations=5)
    
    # Hyperparameter optimization for each model type
    model_types = ['gat', 'graphsage', 'gcn']
    optimization_results = {}
    
    for model_type in model_types:
        logger.info(f"Phase 3: Optimizing {model_type.upper()}...")
        
        optimizer = HyperparameterOptimizer(selected_data, model_type)
        best_params = optimizer.optimize()
        optimization_results[model_type] = best_params
        
        # Save optimization results
        with open(output_dir / f"{model_type}_optimization_results.json", 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': optimizer.best_score
            }, f, indent=2)
    
    # Ensemble training with best model
    logger.info("Phase 4: Ensemble training...")
    
    # Use GAT as the best model (typically performs best)
    best_model_type = 'gat'
    best_params = optimization_results[best_model_type]
    
    # Create train/val/test split
    num_nodes = selected_data.num_nodes
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
    ensemble_trainer = EnsembleTrainer(selected_data, best_params)
    ensemble_trainer.augmented_datasets = augmented_datasets
    
    # Create and train ensemble
    models = ensemble_trainer.create_ensemble_models(num_models=5)
    trained_models = ensemble_trainer.train_ensemble(train_mask, val_mask)
    
    # Make ensemble predictions
    ensemble_pred_class, ensemble_pred_proba = ensemble_trainer.ensemble_predict(trained_models, test_mask)
    
    # Calculate metrics
    y_true = selected_data.y[test_mask].cpu().numpy()
    
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
        'training_config': {
            'num_models': 5,
            'num_augmentations': 5,
            'feature_selection': 'top_500',
            'cross_validation_folds': 5
        }
    }
    
    with open(output_dir / "ensemble_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    logger.info("🎉 HYPERPARAMETER OPTIMIZATION COMPLETED!")
    logger.info("=" * 80)
    logger.info("ENSEMBLE RESULTS:")
    logger.info(f"   - Accuracy: {accuracy:.4f}")
    logger.info(f"   - Precision: {precision:.4f}")
    logger.info(f"   - Recall: {recall:.4f}")
    logger.info(f"   - F1-Score: {f1:.4f}")
    logger.info(f"   - ROC-AUC: {roc_auc:.4f}")
    logger.info("=" * 80)
    
    if accuracy > 0.99:
        logger.info("🎯 TARGET ACHIEVED: >99% accuracy!")
    else:
        logger.info("📈 Further optimization needed to reach >99% accuracy")

if __name__ == "__main__":
    main() 