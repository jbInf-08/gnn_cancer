"""
Comprehensive GAT Optimization to Fully Surpass the Paper
- Advanced hyperparameter optimization
- Multiple training strategies
- Comprehensive evaluation metrics
- Data quality improvements
- Model architecture enhancements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import optuna
from optuna.samplers import TPESampler
import wandb
from tqdm import tqdm
import time
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from optimized_gat_implementation import OptimizedGATModel, AdvancedTrainingConfig

class ComprehensiveGATOptimizer:
    """
    Comprehensive GAT optimizer implementing all paper-surpassing strategies
    """
    
    def __init__(self, data_path: str, device: str = 'auto'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.config = AdvancedTrainingConfig()
        self.best_model = None
        self.best_score = 0.0
        self.optimization_history = []
        
        # Load data
        self.load_data()
        
        logger.info(f"Initialized ComprehensiveGATOptimizer on device: {self.device}")
        logger.info(f"Data shape: {self.data.x.shape}, Edges: {self.data.edge_index.shape[1]}")
    
    def load_data(self):
        """Load and validate data quality"""
        try:
            # Load the enhanced data
            data_file = Path(self.data_path) / "enhanced" / "real_only_torch_geometric_data.pt"
            if data_file.exists():
                self.data = torch.load(data_file, map_location=self.device)
                logger.info(f"Loaded data from {data_file}")
            else:
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            # Validate data quality
            self.validate_data_quality()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data_quality(self):
        """Validate data quality and graph construction"""
        logger.info("Validating data quality...")
        
        # Check for NaN values
        if torch.isnan(self.data.x).any():
            logger.warning("Found NaN values in node features, cleaning...")
            self.data.x = torch.nan_to_num(self.data.x, nan=0.0)
        
        if torch.isnan(self.data.edge_attr).any():
            logger.warning("Found NaN values in edge attributes, cleaning...")
            self.data.edge_attr = torch.nan_to_num(self.data.edge_attr, nan=0.0)
        
        # Check graph connectivity
        num_nodes = self.data.x.shape[0]
        num_edges = self.data.edge_index.shape[1]
        density = num_edges / (num_nodes * (num_nodes - 1))
        
        logger.info(f"Graph statistics:")
        logger.info(f"  Nodes: {num_nodes}")
        logger.info(f"  Edges: {num_edges}")
        logger.info(f"  Density: {density:.6f}")
        logger.info(f"  Node features: {self.data.x.shape[1]}")
        logger.info(f"  Edge features: {self.data.edge_attr.shape[1] if self.data.edge_attr is not None else 0}")
        
        # Check for isolated nodes
        edge_set = set()
        for i in range(self.data.edge_index.shape[1]):
            edge_set.add((self.data.edge_index[0, i].item(), self.data.edge_index[1, i].item()))
        
        isolated_nodes = 0
        for i in range(num_nodes):
            connected = False
            for j in range(num_nodes):
                if i != j and ((i, j) in edge_set or (j, i) in edge_set):
                    connected = True
                    break
            if not connected:
                isolated_nodes += 1
        
        if isolated_nodes > 0:
            logger.warning(f"Found {isolated_nodes} isolated nodes")
        
        # Validate labels
        if hasattr(self.data, 'y') and self.data.y is not None:
            unique_labels = torch.unique(self.data.y)
            logger.info(f"Labels: {unique_labels.tolist()}")
            for label in unique_labels:
                count = (self.data.y == label).sum().item()
                logger.info(f"  Label {label}: {count} samples")
    
    def create_train_val_test_split(self, test_size=0.2, val_size=0.2, random_state=42):
        """Create stratified train/validation/test split"""
        logger.info("Creating train/validation/test split...")
        
        # Get node indices
        num_nodes = self.data.x.shape[0]
        node_indices = np.arange(num_nodes)
        
        # Get labels for stratification
        if hasattr(self.data, 'y') and self.data.y is not None:
            labels = self.data.y.cpu().numpy()
        else:
            # If no labels, use random split
            labels = np.zeros(num_nodes)
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            node_indices, test_size=test_size, random_state=random_state, 
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=labels[train_val_idx] if len(np.unique(labels)) > 1 else None
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        
        logger.info(f"Split sizes - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        return train_mask, val_mask, test_mask
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        
        # Define hyperparameter search space
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.6),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'pooling_strategy': trial.suggest_categorical('pooling_strategy', ['multi', 'attention', 'set2set']),
            'use_skip_connections': trial.suggest_categorical('use_skip_connections', [True, False]),
            'use_graph_attention': trial.suggest_categorical('use_graph_attention', [True, False]),
        }
        
        # Create model with trial parameters
        model = OptimizedGATModel(
            input_dim=self.data.x.shape[1],
            hidden_dim=params['hidden_dim'],
            output_dim=2,  # Binary classification
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            dropout=params['dropout'],
            use_edge_attr=True,
            num_edge_types=8,
            use_skip_connections=params['use_skip_connections'],
            use_graph_attention=params['use_graph_attention'],
            pooling_strategy=params['pooling_strategy']
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Loss function with class weights
        if hasattr(self.data, 'y') and self.data.y is not None:
            labels = self.data.y[self.train_mask]
            class_counts = torch.bincount(labels)
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum()
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_score = 0.0
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(100):  # Reduced epochs for optimization
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(self.data.x, self.data.edge_index, self.data.edge_attr)
            loss = criterion(out[self.train_mask], self.data.y[self.train_mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(self.data.x, self.data.edge_index, self.data.edge_attr)
                val_pred = val_out[self.val_mask].argmax(dim=1)
                val_score = f1_score(self.data.y[self.val_mask].cpu(), val_pred.cpu(), average='weighted')
            
            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                break
        
        return best_val_score
    
    def optimize_hyperparameters(self, n_trials=100):
        """Run comprehensive hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation score: {best_score:.4f}")
        
        # Save optimization results
        optimization_results = {
            'best_params': best_params,
            'best_score': best_score,
            'study_history': study.trials_dataframe().to_dict('records')
        }
        
        with open('results/hyperparameter_optimization_results.json', 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        return best_params, best_score
    
    def train_optimized_model(self, params):
        """Train model with optimized parameters"""
        logger.info("Training optimized model...")
        
        # Create model with best parameters
        model = OptimizedGATModel(
            input_dim=self.data.x.shape[1],
            hidden_dim=params['hidden_dim'],
            output_dim=2,
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            dropout=params['dropout'],
            use_edge_attr=True,
            num_edge_types=8,
            use_skip_connections=params['use_skip_connections'],
            use_graph_attention=params['use_graph_attention'],
            pooling_strategy=params['pooling_strategy']
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Loss function
        if hasattr(self.data, 'y') and self.data.y is not None:
            labels = self.data.y[self.train_mask]
            class_counts = torch.bincount(labels)
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum()
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_score = 0.0
        patience_counter = 0
        max_patience = 50
        training_history = []
        
        for epoch in range(300):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(self.data.x, self.data.edge_index, self.data.edge_attr)
            loss = criterion(out[self.train_mask], self.data.y[self.train_mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(self.data.x, self.data.edge_index, self.data.edge_attr)
                val_pred = val_out[self.val_mask].argmax(dim=1)
                val_score = f1_score(self.data.y[self.val_mask].cpu(), val_pred.cpu(), average='weighted')
                
                # Additional metrics
                val_acc = accuracy_score(self.data.y[self.val_mask].cpu(), val_pred.cpu())
                val_precision = precision_score(self.data.y[self.val_mask].cpu(), val_pred.cpu(), average='weighted')
                val_recall = recall_score(self.data.y[self.val_mask].cpu(), val_pred.cpu(), average='weighted')
            
            # Update scheduler
            scheduler.step(val_score)
            
            # Record history
            training_history.append({
                'epoch': epoch,
                'train_loss': loss.item(),
                'val_score': val_score,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                self.best_model = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}, Val Score={val_score:.4f}, Val Acc={val_acc:.4f}")
        
        # Save training history
        with open('results/optimized_training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Load best model
        model.load_state_dict(self.best_model)
        
        return model, training_history
    
    def evaluate_model(self, model):
        """Comprehensive model evaluation"""
        logger.info("Evaluating optimized model...")
        
        model.eval()
        with torch.no_grad():
            # Test predictions
            test_out = model(self.data.x, self.data.edge_index, self.data.edge_attr)
            test_pred = test_out[self.test_mask].argmax(dim=1)
            test_proba = F.softmax(test_out[self.test_mask], dim=1)
            
            # Calculate metrics
            test_acc = accuracy_score(self.data.y[self.test_mask].cpu(), test_pred.cpu())
            test_precision = precision_score(self.data.y[self.test_mask].cpu(), test_pred.cpu(), average='weighted')
            test_recall = recall_score(self.data.y[self.test_mask].cpu(), test_pred.cpu(), average='weighted')
            test_f1 = f1_score(self.data.y[self.test_mask].cpu(), test_pred.cpu(), average='weighted')
            
            # ROC AUC
            test_auc = roc_auc_score(self.data.y[self.test_mask].cpu(), test_proba[:, 1].cpu())
            
            # Confusion matrix
            cm = confusion_matrix(self.data.y[self.test_mask].cpu(), test_pred.cpu())
            
            # Classification report
            report = classification_report(self.data.y[self.test_mask].cpu(), test_pred.cpu(), output_dict=True)
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        with open('results/optimized_model_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_results(results, cm)
        
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {test_acc:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  F1-Score: {test_f1:.4f}")
        logger.info(f"  AUC: {test_auc:.4f}")
        
        return results
    
    def plot_results(self, results, cm):
        """Plot evaluation results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Metrics bar plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [results['test_accuracy'], results['test_precision'], 
                 results['test_recall'], results['test_f1'], results['test_auc']]
        
        axes[1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        axes[1].set_title('Test Metrics')
        axes[1].set_ylabel('Score')
        axes[1].set_ylim(0, 1)
        
        for i, v in enumerate(values):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/optimized_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_optimization(self):
        """Run the complete optimization pipeline"""
        logger.info("Starting comprehensive GAT optimization...")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Create train/val/test split
        self.create_train_val_test_split()
        
        # Optimize hyperparameters
        best_params, best_score = self.optimize_hyperparameters(n_trials=50)
        
        # Train optimized model
        model, training_history = self.train_optimized_model(best_params)
        
        # Evaluate model
        results = self.evaluate_model(model)
        
        # Save final model
        torch.save(model.state_dict(), 'models/optimized_gat_model.pt')
        
        logger.info("Comprehensive optimization completed!")
        return model, results, training_history

def main():
    """Main execution function"""
    # Initialize optimizer
    optimizer = ComprehensiveGATOptimizer('data')
    
    # Run comprehensive optimization
    model, results, history = optimizer.run_comprehensive_optimization()
    
    print("\n" + "="*50)
    print("COMPREHENSIVE GAT OPTIMIZATION COMPLETED")
    print("="*50)
    print(f"Best Test F1-Score: {results['test_f1']:.4f}")
    print(f"Best Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Best Test AUC: {results['test_auc']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
