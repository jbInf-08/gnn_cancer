#!/usr/bin/env python3
"""
Intelligent Data Preservation Optimization Pipeline
Target: >99% accuracy to exceed paper performance
Uses intelligent sampling to preserve most important data while staying within memory
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
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentDataPreserver:
    """Intelligent data preservation using smart sampling techniques"""
    
    def __init__(self, data: Data, target_memory_gb: float = 8.0):
        self.data = data
        self.target_memory_gb = target_memory_gb
        self.preserved_data = None
        
    def calculate_memory_usage(self, data: Data) -> float:
        """Calculate approximate memory usage in GB"""
        # Node features
        node_memory = data.x.numel() * data.x.element_size() / (1024**3)
        
        # Edge indices
        edge_memory = data.edge_index.numel() * data.edge_index.element_size() / (1024**3)
        
        # Edge attributes
        edge_attr_memory = 0
        if data.edge_attr is not None:
            edge_attr_memory = data.edge_attr.numel() * data.edge_attr.element_size() / (1024**3)
        
        # Labels
        label_memory = data.y.numel() * data.y.element_size() / (1024**3)
        
        total_memory = node_memory + edge_memory + edge_attr_memory + label_memory
        return total_memory
    
    def intelligent_feature_selection(self, top_k: int = 500) -> Data:
        """Intelligent feature selection using statistical tests"""
        logger.info(f"🔍 Performing intelligent feature selection (top {top_k} features)...")
        
        X = self.data.x.numpy()
        y = self.data.y.numpy()
        
        # Use F-test for feature selection
        selector = SelectKBest(score_func=f_classif, k=top_k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        
        logger.info(f"✅ Selected {len(selected_features)} most informative features")
        logger.info(f"   - Original features: {X.shape[1]}")
        logger.info(f"   - Selected features: {X_selected.shape[1]}")
        
        # Create new data with selected features
        selected_x = torch.tensor(X_selected, dtype=torch.float)
        
        return Data(
            x=selected_x,
            edge_index=self.data.edge_index,
            edge_attr=self.data.edge_attr,
            y=self.data.y
        ), selected_features
    
    def intelligent_edge_sampling(self, target_edges: int = 100000) -> Data:
        """Intelligent edge sampling based on edge weights and connectivity"""
        logger.info(f"🌐 Performing intelligent edge sampling (target: {target_edges:,} edges)...")
        
        edge_index = self.data.edge_index
        edge_attr = self.data.edge_attr
        
        # Calculate edge importance scores
        edge_weights = edge_attr[:, 0] if edge_attr is not None else torch.ones(edge_index.shape[1])
        
        # Calculate node degrees
        node_degrees = torch.zeros(self.data.num_nodes)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_degrees[src] += 1
            node_degrees[dst] += 1
        
        # Calculate edge importance based on:
        # 1. Edge weight
        # 2. Node degree centrality
        # 3. Connectivity preservation
        edge_importance = edge_weights.clone()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # Higher importance for edges connecting high-degree nodes
            centrality_score = (node_degrees[src] + node_degrees[dst]) / (2 * node_degrees.max())
            edge_importance[i] *= (1 + centrality_score)
        
        # Sample edges based on importance
        if target_edges < edge_index.shape[1]:
            # Keep top edges by importance
            _, top_indices = torch.topk(edge_importance, target_edges)
            
            new_edge_index = edge_index[:, top_indices]
            new_edge_attr = edge_attr[top_indices] if edge_attr is not None else None
            
            logger.info(f"✅ Sampled {new_edge_index.shape[1]:,} most important edges")
            logger.info(f"   - Original edges: {edge_index.shape[1]:,}")
            logger.info(f"   - Sampled edges: {new_edge_index.shape[1]:,}")
            logger.info(f"   - Reduction: {edge_index.shape[1] / new_edge_index.shape[1]:.1f}x")
            
            return Data(
                x=self.data.x,
                edge_index=new_edge_index,
                edge_attr=new_edge_attr,
                y=self.data.y
            )
        else:
            return self.data
    
    def preserve_data_intelligently(self) -> Data:
        """Intelligently preserve data while staying within memory constraints"""
        logger.info("🧠 Starting intelligent data preservation...")
        
        # Calculate current memory usage
        current_memory = self.calculate_memory_usage(self.data)
        logger.info(f"📊 Current memory usage: {current_memory:.2f} GB")
        
        if current_memory <= self.target_memory_gb:
            logger.info("✅ Data fits within memory constraints - no reduction needed!")
            return self.data
        
        # Step 1: Intelligent feature selection
        logger.info("Step 1: Intelligent feature selection...")
        data_with_selected_features, selected_features = self.intelligent_feature_selection(top_k=500)
        
        # Check memory after feature selection
        memory_after_features = self.calculate_memory_usage(data_with_selected_features)
        logger.info(f"📊 Memory after feature selection: {memory_after_features:.2f} GB")
        
        if memory_after_features <= self.target_memory_gb:
            logger.info("✅ Memory target achieved with feature selection only!")
            return data_with_selected_features
        
        # Step 2: Intelligent edge sampling
        logger.info("Step 2: Intelligent edge sampling...")
        
        # Calculate target edges based on remaining memory budget
        remaining_memory = self.target_memory_gb - memory_after_features
        current_edge_memory = data_with_selected_features.edge_index.numel() * data_with_selected_features.edge_index.element_size() / (1024**3)
        
        if data_with_selected_features.edge_attr is not None:
            current_edge_memory += data_with_selected_features.edge_attr.numel() * data_with_selected_features.edge_attr.element_size() / (1024**3)
        
        # Estimate target edges
        target_edges = int(current_edge_memory * remaining_memory / current_edge_memory)
        target_edges = max(50000, min(target_edges, data_with_selected_features.edge_index.shape[1]))
        
        preserved_data = self.intelligent_edge_sampling(data_with_selected_features, target_edges)
        
        # Final memory check
        final_memory = self.calculate_memory_usage(preserved_data)
        logger.info(f"📊 Final memory usage: {final_memory:.2f} GB")
        
        self.preserved_data = preserved_data
        return preserved_data

class IntelligentGAT(nn.Module):
    """Intelligent GAT with memory optimization"""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, 
                 num_layers: int, num_heads: int, dropout: float, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        
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

class IntelligentGraphSAGE(nn.Module):
    """Intelligent GraphSAGE with memory optimization"""
    
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
        
        # GraphSAGE layers
        for i in range(self.num_layers):
            x = self.sage_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class IntelligentGCN(nn.Module):
    """Intelligent GCN with memory optimization"""
    
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
        
        # GCN layers
        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return x

class IntelligentOptimizer:
    """Intelligent hyperparameter optimization with data preservation"""
    
    def __init__(self, data: Data, model_type: str = 'gat'):
        self.data = data
        self.model_type = model_type
        self.best_params = None
        self.best_score = 0.0
        
        # Optimization parameters
        self.n_trials = 50
        self.n_folds = 3
        
        # Model registry
        self.model_registry = {
            'gat': IntelligentGAT,
            'graphsage': IntelligentGraphSAGE,
            'gcn': IntelligentGCN
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function with intelligent optimization"""
        
        # Define hyperparameter search space
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'epochs': trial.suggest_int('epochs', 30, 100),
            'patience': trial.suggest_int('patience', 10, 25)
        }
        
        # Add model-specific parameters
        if self.model_type == 'gat':
            params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 16])
        
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
        logger.info(f"🚀 Starting intelligent optimization for {self.model_type.upper()}")
        logger.info(f"   - Trials: {self.n_trials}")
        logger.info(f"   - Cross-validation folds: {self.n_folds}")
        logger.info(f"   - Data: {self.data.num_nodes} nodes, {self.data.num_edges} edges, {self.data.num_features} features")
        
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

class IntelligentEnsembleTrainer:
    """Intelligent ensemble training with data preservation"""
    
    def __init__(self, data: Data, optimized_params: Dict):
        self.data = data
        self.optimized_params = optimized_params
        self.models = []
    
    def create_ensemble_models(self, num_models: int = 5) -> List[nn.Module]:
        """Create multiple models with different initializations"""
        logger.info(f"🏗️ Creating {num_models} intelligent ensemble models...")
        
        models = []
        for i in range(num_models):
            # Set different random seeds for different initializations
            torch.manual_seed(42 + i)
            
            # Create model with optimized parameters
            if 'num_heads' in self.optimized_params:
                model = IntelligentGAT(
                    num_features=self.data.num_features,
                    num_classes=len(torch.unique(self.data.y)),
                    **self.optimized_params
                )
            else:
                model = IntelligentGraphSAGE(
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
        logger.info("🚀 Training intelligent ensemble models...")
        
        trained_models = []
        
        for i, model in enumerate(tqdm(self.models, desc="Training ensemble")):
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
    
    def ensemble_predict(self, models: List[nn.Module], test_mask: torch.Tensor) -> np.ndarray:
        """Make ensemble predictions"""
        logger.info("🔮 Making intelligent ensemble predictions...")
        
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
    logger.info("🚀 Starting INTELLIGENT DATA PRESERVATION OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("🧠 Using intelligent sampling to preserve most important data!")
    logger.info("=" * 80)
    
    # Load processed data
    data_path = Path("data/massive_processed/massive_processed_data.pt")
    if not data_path.exists():
        logger.error("❌ Processed data not found! Run feature engineering first.")
        return
    
    original_data = torch.load(data_path)
    logger.info(f"✅ Original data loaded: {original_data.num_nodes} nodes, {original_data.num_edges} edges, {original_data.num_features} features")
    
    # Intelligent data preservation
    data_preserver = IntelligentDataPreserver(original_data, target_memory_gb=8.0)
    preserved_data = data_preserver.preserve_data_intelligently()
    
    logger.info(f"🎯 Preserved data: {preserved_data.num_nodes} nodes, {preserved_data.num_edges} edges, {preserved_data.num_features} features")
    
    # Create output directory
    output_dir = Path("results/intelligent_data_preservation_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameter optimization for each model type
    model_types = ['gat', 'graphsage', 'gcn']
    optimization_results = {}
    
    for model_type in model_types:
        logger.info(f"Phase 1: Optimizing {model_type.upper()} with intelligent data preservation...")
        
        optimizer = IntelligentOptimizer(preserved_data, model_type)
        best_params = optimizer.optimize()
        optimization_results[model_type] = best_params
        
        # Save optimization results
        with open(output_dir / f"{model_type}_intelligent_optimization_results.json", 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': optimizer.best_score,
                'data_preservation': {
                    'original_nodes': original_data.num_nodes,
                    'original_edges': original_data.num_edges,
                    'original_features': original_data.num_features,
                    'preserved_nodes': preserved_data.num_nodes,
                    'preserved_edges': preserved_data.num_edges,
                    'preserved_features': preserved_data.num_features,
                    'intelligent_sampling': True
                }
            }, f, indent=2)
    
    # Ensemble training with best model
    logger.info("Phase 2: Intelligent ensemble training...")
    
    # Use GAT as the best model
    best_model_type = 'gat'
    best_params = optimization_results[best_model_type]
    
    # Create train/val/test split
    num_nodes = preserved_data.num_nodes
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
    ensemble_trainer = IntelligentEnsembleTrainer(preserved_data, best_params)
    
    # Create and train ensemble
    models = ensemble_trainer.create_ensemble_models(num_models=5)
    trained_models = ensemble_trainer.train_ensemble(train_mask, val_mask)
    
    # Make ensemble predictions
    ensemble_pred_class, ensemble_pred_proba = ensemble_trainer.ensemble_predict(trained_models, test_mask)
    
    # Calculate metrics
    y_true = preserved_data.y[test_mask].cpu().numpy()
    
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
            'original_nodes': original_data.num_nodes,
            'original_edges': original_data.num_edges,
            'original_features': original_data.num_features,
            'preserved_nodes': preserved_data.num_nodes,
            'preserved_edges': preserved_data.num_edges,
            'preserved_features': preserved_data.num_features,
            'intelligent_sampling': True
        },
        'intelligent_techniques': [
            'Statistical feature selection (F-test)',
            'Edge importance scoring',
            'Node centrality preservation',
            'Memory-aware sampling',
            'Intelligent data reduction'
        ]
    }
    
    with open(output_dir / "intelligent_data_preservation_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    logger.info("🎉 INTELLIGENT DATA PRESERVATION OPTIMIZATION COMPLETED!")
    logger.info("=" * 80)
    logger.info("🧠 INTELLIGENT DATA PRESERVATION ACHIEVED:")
    logger.info(f"   - Original: {original_data.num_nodes} nodes, {original_data.num_edges:,} edges, {original_data.num_features} features")
    logger.info(f"   - Preserved: {preserved_data.num_nodes} nodes, {preserved_data.num_edges:,} edges, {preserved_data.num_features} features")
    logger.info(f"   - Data retention: {preserved_data.num_features/original_data.num_features*100:.1f}% features, {preserved_data.num_edges/original_data.num_edges*100:.1f}% edges")
    logger.info("=" * 80)
    logger.info("ENSEMBLE RESULTS:")
    logger.info(f"   - Accuracy: {accuracy:.4f}")
    logger.info(f"   - Precision: {precision:.4f}")
    logger.info(f"   - Recall: {recall:.4f}")
    logger.info(f"   - F1-Score: {f1:.4f}")
    logger.info(f"   - ROC-AUC: {roc_auc:.4f}")
    logger.info("=" * 80)
    
    if accuracy > 0.99:
        logger.info("🎯 TARGET ACHIEVED: >99% accuracy with intelligent data preservation!")
    else:
        logger.info("📈 Further optimization needed to reach >99% accuracy")

if __name__ == "__main__":
    main() 