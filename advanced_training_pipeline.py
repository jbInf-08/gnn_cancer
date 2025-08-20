"""
Advanced Training Pipeline to Surpass Paper Performance
- Comprehensive hyperparameter optimization
- Advanced training strategies
- Ensemble methods
- Cross-validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from enhanced_gat_v2 import EnhancedGATv2Model, AdvancedTrainingConfig, create_enhanced_gat_model

class AdvancedTrainingPipeline:
    def __init__(self, data_path: str, device: str = 'auto'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.config = AdvancedTrainingConfig()
        self.best_model = None
        self.best_score = 0.0
        self.optimization_history = []
        self.results = {}
        
        # Load data
        self.load_data()
        logger.info(f"Initialized AdvancedTrainingPipeline on device: {self.device}")
        logger.info(f"Data shape: {self.data.x.shape}, Edges: {self.data.edge_index.shape[1]}")
    
    def load_data(self):
        """Load and validate data"""
        try:
            data_file = Path(self.data_path) / "enhanced" / "real_only_torch_geometric_data.pt"
            if data_file.exists():
                self.data = torch.load(data_file, map_location=self.device)
                logger.info(f"Loaded data from {data_file}")
            else:
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            self.validate_data_quality()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data_quality(self):
        """Validate and clean data quality"""
        logger.info("Validating data quality...")
        
        # Clean NaN values
        if torch.isnan(self.data.x).any():
            logger.warning("Found NaN values in node features, cleaning...")
            self.data.x = torch.nan_to_num(self.data.x, nan=0.0)
        
        if torch.isnan(self.data.edge_attr).any():
            logger.warning("Found NaN values in edge attributes, cleaning...")
            self.data.edge_attr = torch.nan_to_num(self.data.edge_attr, nan=0.0)
        
        # Log statistics
        num_nodes = self.data.x.shape[0]
        num_edges = self.data.edge_index.shape[1]
        density = num_edges / (num_nodes * (num_nodes - 1))
        
        logger.info(f"Graph statistics:")
        logger.info(f"  Nodes: {num_nodes}")
        logger.info(f"  Edges: {num_edges}")
        logger.info(f"  Density: {density:.6f}")
        logger.info(f"  Node features: {self.data.x.shape[1]}")
        logger.info(f"  Edge features: {self.data.edge_attr.shape[1] if self.data.edge_attr is not None else 0}")
        
        # Check labels
        if hasattr(self.data, 'y') and self.data.y is not None:
            unique_labels = torch.unique(self.data.y)
            logger.info(f"Labels: {unique_labels.tolist()}")
            for label in unique_labels:
                count = (self.data.y == label).sum().item()
                logger.info(f"  Label {label}: {count} samples")
    
    def create_train_val_test_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Create stratified train/validation/test splits"""
        logger.info("Creating train/validation/test splits...")
        
        # Get node indices
        num_nodes = self.data.x.shape[0]
        node_indices = torch.arange(num_nodes)
        
        # Create splits
        train_val_indices, test_indices = train_test_split(
            node_indices, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.data.y if hasattr(self.data, 'y') else None
        )
        
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size,
            random_state=random_state,
            stratify=self.data.y[train_val_indices] if hasattr(self.data, 'y') else None
        )
        
        # Create masks
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[train_indices] = True
        self.val_mask[val_indices] = True
        self.test_mask[test_indices] = True
        
        logger.info(f"Split sizes:")
        logger.info(f"  Train: {self.train_mask.sum().item()}")
        logger.info(f"  Validation: {self.val_mask.sum().item()}")
        logger.info(f"  Test: {self.test_mask.sum().item()}")
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        # Sample hyperparameters
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.6),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'pooling_strategy': trial.suggest_categorical('pooling_strategy', ['multi', 'attention', 'set2set']),
            'use_skip_connections': trial.suggest_categorical('use_skip_connections', [True, False]),
            'use_multi_scale': trial.suggest_categorical('use_multi_scale', [True, False]),
            'use_attention_pooling': trial.suggest_categorical('use_attention_pooling', [True, False]),
            'activation': trial.suggest_categorical('activation', ['elu', 'relu', 'leaky_relu']),
        }
        
        # Create model
        model = create_enhanced_gat_model(
            input_dim=self.data.x.shape[1],
            output_dim=2,
            config=self._create_config_from_params(params)
        ).to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'], 
            weight_decay=params['weight_decay']
        )
        
        # Create loss function with class weights
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
        
        for epoch in range(100):
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
                val_score = f1_score(
                    self.data.y[self.val_mask].cpu(), 
                    val_pred.cpu(), 
                    average='weighted'
                )
            
            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                break
        
        return best_val_score
    
    def _create_config_from_params(self, params):
        """Create config from hyperparameters"""
        config = AdvancedTrainingConfig()
        config.hidden_dim = params['hidden_dim']
        config.num_layers = params['num_layers']
        config.num_heads = params['num_heads']
        config.dropout = params['dropout']
        config.learning_rate = params['learning_rate']
        config.weight_decay = params['weight_decay']
        config.pooling_strategy = params['pooling_strategy']
        config.use_skip_connections = params['use_skip_connections']
        config.use_multi_scale = params['use_multi_scale']
        config.use_attention_pooling = params['use_attention_pooling']
        config.activation = params['activation']
        return config
    
    def optimize_hyperparameters(self, n_trials=100):
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best validation score: {self.best_score:.4f}")
        
        return self.best_params
    
    def train_final_model(self, params=None):
        """Train final model with best parameters"""
        if params is None:
            params = self.best_params
        
        logger.info("Training final model...")
        
        # Create model
        config = self._create_config_from_params(params)
        model = create_enhanced_gat_model(
            input_dim=self.data.x.shape[1],
            output_dim=2,
            config=config
        ).to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'], 
            weight_decay=params['weight_decay']
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Create loss function
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
                val_score = f1_score(
                    self.data.y[self.val_mask].cpu(), 
                    val_pred.cpu(), 
                    average='weighted'
                )
            
            # Update scheduler
            scheduler.step(val_score)
            
            # Store history
            training_history.append({
                'epoch': epoch,
                'train_loss': loss.item(),
                'val_score': val_score
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
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Score: {val_score:.4f}")
        
        # Load best model
        model.load_state_dict(self.best_model)
        self.final_model = model
        
        return training_history
    
    def evaluate_model(self, model=None):
        """Evaluate model on test set"""
        if model is None:
            model = self.final_model
        
        logger.info("Evaluating model...")
        
        model.eval()
        with torch.no_grad():
            test_out = model(self.data.x, self.data.edge_index, self.data.edge_attr)
            test_pred = test_out[self.test_mask].argmax(dim=1)
            test_proba = F.softmax(test_out[self.test_mask], dim=1)
            
            # Calculate metrics
            accuracy = accuracy_score(self.data.y[self.test_mask].cpu(), test_pred.cpu())
            precision = precision_score(self.data.y[self.test_mask].cpu(), test_pred.cpu(), average='weighted')
            recall = recall_score(self.data.y[self.test_mask].cpu(), test_pred.cpu(), average='weighted')
            f1 = f1_score(self.data.y[self.test_mask].cpu(), test_pred.cpu(), average='weighted')
            roc_auc = roc_auc_score(self.data.y[self.test_mask].cpu(), test_proba[:, 1].cpu())
            
            # Confusion matrix
            cm = confusion_matrix(self.data.y[self.test_mask].cpu(), test_pred.cpu())
            
            # Store results
            self.results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm.tolist(),
                'predictions': test_pred.cpu().numpy(),
                'probabilities': test_proba.cpu().numpy()
            }
            
            logger.info(f"Test Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  ROC AUC: {roc_auc:.4f}")
            
            return self.results
    
    def save_results(self, output_dir="results"):
        """Save results and model"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save model
        if self.final_model is not None:
            torch.save(self.final_model.state_dict(), output_path / "enhanced_gat_model.pt")
        
        # Save results
        with open(output_path / "enhanced_gat_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save hyperparameters
        with open(output_path / "enhanced_gat_params.json", 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def run_full_pipeline(self, n_trials=100):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        # Create data splits
        self.create_train_val_test_splits()
        
        # Optimize hyperparameters
        self.optimize_hyperparameters(n_trials=n_trials)
        
        # Train final model
        training_history = self.train_final_model()
        
        # Evaluate model
        results = self.evaluate_model()
        
        # Save results
        self.save_results()
        
        logger.info("Full pipeline completed!")
        return results

def main():
    """Main function to run the advanced training pipeline"""
    # Initialize pipeline
    pipeline = AdvancedTrainingPipeline("data")
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(n_trials=50)
    
    print("Final Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
