#!/usr/bin/env python3
"""
Hybrid Intelligent Optimization Pipeline
Target: >99% accuracy to exceed paper performance
Combines intelligent data sampling for training with full data preservation for evaluation
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
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridDataManager:
    """Hybrid data management: intelligent sampling for training, full data for evaluation"""
    
    def __init__(self, data: Data):
        self.full_data = data
        self.training_data = None
        self.selected_features = None
        
    def create_intelligent_training_data(self, target_edges: int = 100000, target_features: int = 500) -> Data:
        """Create intelligently sampled data for training while preserving full data"""
        logger.info("🧠 Creating intelligent training data...")
        
        # Step 1: Intelligent feature selection
        logger.info("Step 1: Intelligent feature selection...")
        X = self.full_data.x.numpy()
        y = self.full_data.y.numpy()
        
        # Use F-test for feature selection
        selector = SelectKBest(score_func=f_classif, k=target_features)
        X_selected = selector.fit_transform(X, y)
        self.selected_features = selector.get_support(indices=True)
        
        logger.info(f"✅ Selected {len(self.selected_features)} most informative features")
        
        # Step 2: Intelligent edge sampling
        logger.info("Step 2: Intelligent edge sampling...")
        edge_index = self.full_data.edge_index
        edge_attr = self.full_data.edge_attr
        
        # Calculate edge importance scores
        edge_weights = edge_attr[:, 0] if edge_attr is not None else torch.ones(edge_index.shape[1])
        
        # Calculate node degrees
        node_degrees = torch.zeros(self.full_data.num_nodes)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_degrees[src] += 1
            node_degrees[dst] += 1
        
        # Calculate edge importance based on weight and centrality
        edge_importance = edge_weights.clone()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            centrality_score = (node_degrees[src] + node_degrees[dst]) / (2 * node_degrees.max())
            edge_importance[i] *= (1 + centrality_score)
        
        # Keep top edges by importance
        _, top_indices = torch.topk(edge_importance, target_edges)
        
        new_edge_index = edge_index[:, top_indices]
        new_edge_attr = edge_attr[top_indices] if edge_attr is not None else None
        
        logger.info(f"✅ Sampled {new_edge_index.shape[1]:,} most important edges")
        logger.info(f"   - Original: {edge_index.shape[1]:,} edges")
        logger.info(f"   - Sampled: {new_edge_index.shape[1]:,} edges")
        logger.info(f"   - Reduction: {edge_index.shape[1] / new_edge_index.shape[1]:.1f}x")
        
        # Create training data
        selected_x = torch.tensor(X_selected, dtype=torch.float)
        self.training_data = Data(
            x=selected_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=self.full_data.y
        )
        
        return self.training_data
    
    def get_full_data_with_selected_features(self) -> Data:
        """Get full data with only selected features for final evaluation"""
        if self.selected_features is None:
            return self.full_data
        
        # Apply feature selection to full data
        X = self.full_data.x.numpy()
        X_selected = X[:, self.selected_features]
        
        return Data(
            x=torch.tensor(X_selected, dtype=torch.float),
            edge_index=self.full_data.edge_index,
            edge_attr=self.full_data.edge_attr,
            y=self.full_data.y
        )

class HybridGAT(nn.Module):
    """Hybrid GAT with advanced architecture"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, num_heads: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(GATConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout))
            self.layer_norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        # Output projection with advanced architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers with residual connections
        for i in range(self.num_layers):
            residual = x
            x = self.gat_layers[i](x, edge_index)
            x = self.layer_norms[i](x)
            
            # Residual connection (if dimensions match)
            if i > 0 and residual.shape[1] == x.shape[1]:
                x = x + residual
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class HybridGraphSAGE(nn.Module):
    """Hybrid GraphSAGE with advanced architecture"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GraphSAGE layers with residual connections
        self.sage_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            self.sage_layers.append(SAGEConv(in_channels, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection with advanced architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GraphSAGE layers with residual connections
        for i in range(self.num_layers):
            residual = x
            x = self.sage_layers[i](x, edge_index)
            x = self.layer_norms[i](x)
            
            # Residual connection
            if i > 0:
                x = x + residual
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class HybridGCN(nn.Module):
    """Hybrid GCN with advanced architecture"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GCN layers with residual connections
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GCNConv(in_channels, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection with advanced architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GCN layers with residual connections
        for i in range(self.num_layers):
            residual = x
            x = self.gcn_layers[i](x, edge_index)
            x = self.layer_norms[i](x)
            
            # Residual connection
            if i > 0:
                x = x + residual
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class HybridOptimizer:
    """Hybrid hyperparameter optimization with intelligent data management"""
    
    def __init__(self, data_manager: HybridDataManager, model_type: str = 'gat'):
        self.data_manager = data_manager
        self.model_type = model_type
        self.best_params = None
        self.best_score = 0.0
        
        # Optimization parameters
        self.n_trials = 50
        self.n_folds = 3
        
        # Model registry
        self.model_registry = {
            'gat': HybridGAT,
            'graphsage': HybridGraphSAGE,
            'gcn': HybridGCN
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function with hybrid optimization"""
        
        # Define hyperparameter search space
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 3, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'epochs': trial.suggest_int('epochs', 50, 150),
            'patience': trial.suggest_int('patience', 15, 30)
        }
        
        # Add model-specific parameters
        if self.model_type == 'gat':
            params['num_heads'] = trial.suggest_categorical('num_heads', [8, 16])
        
        # Use training data for optimization
        training_data = self.data_manager.training_data
        
        # Cross-validation
        scores = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(training_data.num_nodes), training_data.y)):
            # Create train/val masks
            train_mask = torch.zeros(training_data.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(training_data.num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            # Create model
            model_class = self.model_registry[self.model_type]
            model = model_class(
                num_features=training_data.num_features,
                num_classes=len(torch.unique(training_data.y)),
                **params
            )
            
            # Optimizer with advanced settings
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay'],
                eps=1e-8
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(params['epochs']):
                model.train()
                optimizer.zero_grad()
                
                out = model(training_data)
                loss = criterion(out[train_mask], training_data.y[train_mask])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    out = model(training_data)
                    val_acc = accuracy_score(
                        training_data.y[val_mask].cpu().numpy(),
                        out[val_mask].argmax(dim=1).cpu().numpy()
                    )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= params['patience']:
                    break
                
                # Memory cleanup
                if epoch % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            scores.append(best_val_acc)
            
            # Clean up
            del model, optimizer, scheduler, criterion
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
        logger.info(f"🚀 Starting hybrid optimization for {self.model_type.upper()}")
        logger.info(f"   - Trials: {self.n_trials}")
        logger.info(f"   - Cross-validation folds: {self.n_folds}")
        logger.info(f"   - Training data: {self.data_manager.training_data.num_nodes} nodes, {self.data_manager.training_data.num_edges} edges, {self.data_manager.training_data.num_features} features")
        
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

class HybridEnsembleTrainer:
    """Hybrid ensemble training with full data evaluation"""
    
    def __init__(self, data_manager: HybridDataManager, optimized_params: Dict):
        self.data_manager = data_manager
        self.optimized_params = optimized_params
        self.models = []
    
    def create_ensemble_models(self, num_models: int = 5) -> List[nn.Module]:
        """Create multiple models with different initializations"""
        logger.info(f"🏗️ Creating {num_models} hybrid ensemble models...")
        
        models = []
        for i in range(num_models):
            # Set different random seeds for different initializations
            torch.manual_seed(42 + i)
            
            # Create model with optimized parameters
            if 'num_heads' in self.optimized_params:
                model = HybridGAT(
                    num_features=self.data_manager.training_data.num_features,
                    num_classes=len(torch.unique(self.data_manager.training_data.y)),
                    **self.optimized_params
                )
            else:
                model = HybridGraphSAGE(
                    num_features=self.data_manager.training_data.num_features,
                    num_classes=len(torch.unique(self.data_manager.training_data.y)),
                    **self.optimized_params
                )
            
            models.append(model)
        
        self.models = models
        logger.info(f"✅ Created {len(models)} ensemble models")
        return models
    
    def train_ensemble(self, train_mask: torch.Tensor, val_mask: torch.Tensor) -> List[nn.Module]:
        """Train ensemble of models on training data"""
        logger.info("🚀 Training hybrid ensemble models...")
        
        training_data = self.data_manager.training_data
        trained_models = []
        
        for i, model in enumerate(tqdm(self.models, desc="Training ensemble")):
            # Train model
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.optimized_params['lr'], 
                weight_decay=self.optimized_params['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(self.optimized_params['epochs']):
                model.train()
                optimizer.zero_grad()
                
                out = model(training_data)
                loss = criterion(out[train_mask], training_data.y[train_mask])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    out = model(training_data)
                    val_acc = accuracy_score(
                        training_data.y[val_mask].cpu().numpy(),
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
                if epoch % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            trained_models.append(model)
        
        logger.info(f"✅ Trained {len(trained_models)} ensemble models")
        return trained_models
    
    def ensemble_predict_full_data(self, models: List[nn.Module], test_mask: torch.Tensor) -> np.ndarray:
        """Make ensemble predictions on full data for final evaluation"""
        logger.info("🔮 Making hybrid ensemble predictions on FULL data...")
        
        # Get full data with selected features
        full_data = self.data_manager.get_full_data_with_selected_features()
        
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                out = model(full_data)
                pred = F.softmax(out[test_mask], dim=1)
                predictions.append(pred.cpu().numpy())
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_class = np.argmax(ensemble_pred, axis=1)
        
        return ensemble_pred_class, ensemble_pred

def main():
    """Main execution function"""
    logger.info("🚀 Starting HYBRID INTELLIGENT OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("🧠 Hybrid approach: Intelligent sampling for training, full data for evaluation!")
    logger.info("=" * 80)
    
    # Load processed data
    data_path = Path("data/massive_processed/massive_processed_data.pt")
    if not data_path.exists():
        logger.error("❌ Processed data not found! Run feature engineering first.")
        return
    
    full_data = torch.load(data_path)
    logger.info(f"✅ Full data loaded: {full_data.num_nodes} nodes, {full_data.num_edges} edges, {full_data.num_features} features")
    
    # Create hybrid data manager
    data_manager = HybridDataManager(full_data)
    training_data = data_manager.create_intelligent_training_data(target_edges=100000, target_features=500)
    
    logger.info(f"🎯 Training data: {training_data.num_nodes} nodes, {training_data.num_edges} edges, {training_data.num_features} features")
    logger.info(f"📊 Data retention: {training_data.num_features/full_data.num_features*100:.1f}% features, {training_data.num_edges/full_data.num_edges*100:.1f}% edges")
    
    # Create output directory
    output_dir = Path("results/hybrid_intelligent_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameter optimization for each model type
    model_types = ['gat', 'graphsage', 'gcn']
    optimization_results = {}
    
    for model_type in model_types:
        logger.info(f"Phase 1: Optimizing {model_type.upper()} with hybrid approach...")
        
        optimizer = HybridOptimizer(data_manager, model_type)
        best_params = optimizer.optimize()
        optimization_results[model_type] = best_params
        
        # Save optimization results
        with open(output_dir / f"{model_type}_hybrid_optimization_results.json", 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': optimizer.best_score,
                'data_management': {
                    'full_nodes': full_data.num_nodes,
                    'full_edges': full_data.num_edges,
                    'full_features': full_data.num_features,
                    'training_nodes': training_data.num_nodes,
                    'training_edges': training_data.num_edges,
                    'training_features': training_data.num_features,
                    'hybrid_approach': True
                }
            }, f, indent=2)
    
    # Ensemble training with best model
    logger.info("Phase 2: Hybrid ensemble training...")
    
    # Use GAT as the best model
    best_model_type = 'gat'
    best_params = optimization_results[best_model_type]
    
    # Create train/val/test split on training data
    num_nodes = training_data.num_nodes
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
    ensemble_trainer = HybridEnsembleTrainer(data_manager, best_params)
    
    # Create and train ensemble
    models = ensemble_trainer.create_ensemble_models(num_models=5)
    trained_models = ensemble_trainer.train_ensemble(train_mask, val_mask)
    
    # Make ensemble predictions on FULL data
    ensemble_pred_class, ensemble_pred_proba = ensemble_trainer.ensemble_predict_full_data(trained_models, test_mask)
    
    # Calculate metrics
    y_true = training_data.y[test_mask].cpu().numpy()
    
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
        'data_management': {
            'full_nodes': full_data.num_nodes,
            'full_edges': full_data.num_edges,
            'full_features': full_data.num_features,
            'training_nodes': training_data.num_nodes,
            'training_edges': training_data.num_edges,
            'training_features': training_data.num_features,
            'hybrid_approach': True
        },
        'hybrid_techniques': [
            'Intelligent feature selection (F-test)',
            'Edge importance scoring',
            'Node centrality preservation',
            'Training on sampled data',
            'Evaluation on full data',
            'Advanced model architectures',
            'Residual connections',
            'Layer normalization'
        ]
    }
    
    with open(output_dir / "hybrid_intelligent_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    logger.info("🎉 HYBRID INTELLIGENT OPTIMIZATION COMPLETED!")
    logger.info("=" * 80)
    logger.info("🧠 HYBRID DATA MANAGEMENT ACHIEVED:")
    logger.info(f"   - Full data: {full_data.num_nodes} nodes, {full_data.num_edges:,} edges, {full_data.num_features} features")
    logger.info(f"   - Training data: {training_data.num_nodes} nodes, {training_data.num_edges:,} edges, {training_data.num_features} features")
    logger.info(f"   - Data retention: {training_data.num_features/full_data.num_features*100:.1f}% features, {training_data.num_edges/full_data.num_edges*100:.1f}% edges")
    logger.info("=" * 80)
    logger.info("ENSEMBLE RESULTS (on full data):")
    logger.info(f"   - Accuracy: {accuracy:.4f}")
    logger.info(f"   - Precision: {precision:.4f}")
    logger.info(f"   - Recall: {recall:.4f}")
    logger.info(f"   - F1-Score: {f1:.4f}")
    logger.info(f"   - ROC-AUC: {roc_auc:.4f}")
    logger.info("=" * 80)
    
    if accuracy > 0.99:
        logger.info("🎯 TARGET ACHIEVED: >99% accuracy with hybrid intelligent optimization!")
    else:
        logger.info("📈 Further optimization needed to reach >99% accuracy")

if __name__ == "__main__":
    main() 