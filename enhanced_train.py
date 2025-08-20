"""
Enhanced Training Pipeline implementing reference paper's approach
- Proper train/validation/test splits (70/15/15)
- Stratified sampling by mutation class
- Comprehensive hyperparameter tuning
- K-fold cross-validation
- Enhanced evaluation framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import wandb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import enhanced models
from models.enhanced_gnn_models import get_enhanced_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """
    Enhanced trainer implementing reference paper's approach
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project=config.get('project_name', 'enhanced-gnn-cancer'),
                config=config,
                name=f"{config['model']}_{config['cancer_type']}_enhanced"
            )
        
        # Results storage
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': {},
            'best_model_path': None
        }
    
    def load_and_preprocess_data(self, cancer_type: str) -> List[torch.Tensor]:
        """
        Load and preprocess data with proper splits (graph-level)
        """
        logger.info(f"Loading sample graphs for {cancer_type}")
        
        # Load the sample graph dataset
        data_path = Path(f"data/processed/{cancer_type}_sample_dataset.pt")
        if not data_path.exists():
            raise FileNotFoundError(f"Sample dataset not found: {data_path}")
        
        sample_graphs = torch.load(data_path, weights_only=False)
        logger.info(f"Loaded {len(sample_graphs)} sample graphs")
        
        # Extract graph-level labels
        graph_labels = [g.y.item() for g in sample_graphs]
        
        # Encode labels
        label_encoder = LabelEncoder()
        graph_labels_encoded = label_encoder.fit_transform(graph_labels)
        for i, g in enumerate(sample_graphs):
            g.y = torch.tensor([graph_labels_encoded[i]], dtype=torch.long)
        
        logger.info(f"Labels: {np.unique(graph_labels_encoded, return_counts=True)}")
        
        return sample_graphs
    
    def create_stratified_splits(self, sample_graphs: List[torch.Tensor], n_splits: int = 5) -> List[Tuple]:
        """
        Create stratified k-fold splits for graph-level data
        """
        logger.info(f"Creating {n_splits}-fold stratified splits for sample graphs")
        
        labels_np = np.array([g.y.item() for g in sample_graphs])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []
        for train_val_idx, test_idx in skf.split(np.arange(len(labels_np)), labels_np):
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.15/(0.7+0.15),
                stratify=labels_np[train_val_idx],
                random_state=42
            )
            splits.append((train_idx, val_idx, test_idx))
        logger.info(f"Created {len(splits)} splits")
        return splits
    
    def create_data_loaders(self, sample_graphs: List[torch.Tensor], train_idx, val_idx, test_idx) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for train/validation/test sets (graph-level)
        """
        train_graphs = [sample_graphs[i] for i in train_idx]
        val_graphs = [sample_graphs[i] for i in val_idx]
        test_graphs = [sample_graphs[i] for i in test_idx]
        train_loader = DataLoader(train_graphs, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=self.config['batch_size'], shuffle=False)
        logger.info(f"Created loaders: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
        return train_loader, val_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, fold: int) -> Dict:
        """
        Train model with enhanced training loop
        """
        logger.info(f"Training model for fold {fold}")
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            total_train_loss = 0
            train_predictions = []
            train_labels = []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Collect predictions
                pred = out.argmax(dim=1)
                train_predictions.extend(pred.cpu().numpy())
                train_labels.extend(batch.y.cpu().numpy())
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = F.cross_entropy(out, batch.y)
                    total_val_loss += loss.item()
                    
                    pred = out.argmax(dim=1)
                    val_predictions.extend(pred.cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            train_metric = self.calculate_metrics(train_predictions, train_labels)
            val_metric = self.calculate_metrics(val_predictions, val_labels)
            
            # Store results
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                model_path = f"models/checkpoints/best_model_fold_{fold}.pt"
                torch.save(model.state_dict(), model_path)
                self.results['best_model_path'] = model_path
            else:
                patience_counter += 1
            
            # Logging
            if self.config.get('use_wandb', True):
                wandb.log({
                    f'fold_{fold}/train_loss': avg_train_loss,
                    f'fold_{fold}/val_loss': avg_val_loss,
                    f'fold_{fold}/train_acc': train_metric['accuracy'],
                    f'fold_{fold}/val_acc': val_metric['accuracy'],
                    f'fold_{fold}/train_f1': train_metric['f1'],
                    f'fold_{fold}/val_f1': val_metric['f1'],
                    f'fold_{fold}/epoch': epoch
                })
            
            logger.info(f"Fold {fold}, Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Train Acc: {train_metric['accuracy']:.4f}, "
                       f"Val Acc: {val_metric['accuracy']:.4f}")
            
            # Early stopping
            if patience_counter >= self.config.get('patience', 10):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss
        }
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, fold: int) -> Dict:
        """
        Evaluate model on test set
        """
        logger.info(f"Evaluating model for fold {fold}")
        
        model.eval()
        test_predictions = []
        test_labels = []
        test_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                probs = F.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                
                test_predictions.extend(pred.cpu().numpy())
                test_labels.extend(batch.y.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(test_predictions, test_labels, test_probs)
        
        logger.info(f"Fold {fold} Test Results: {metrics}")
        
        return metrics
    
    def calculate_metrics(self, predictions: List, labels: List, 
                         probabilities: Optional[List] = None) -> Dict:
        """
        Calculate comprehensive metrics
        """
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
        }
        
        if probabilities is not None:
            try:
                # Calculate AUC for each class
                probabilities = np.array(probabilities)
                labels_array = np.array(labels)
                
                if probabilities.shape[1] == 2:
                    # Binary classification
                    metrics['auc'] = roc_auc_score(labels_array, probabilities[:, 1])
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(labels_array, probabilities, multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
                metrics['auc'] = 0.0
        
        return metrics
    
    def run_cross_validation(self, cancer_type: str) -> Dict:
        """
        Run k-fold cross-validation (graph-level)
        """
        logger.info(f"Running cross-validation for {cancer_type}")
        sample_graphs = self.load_and_preprocess_data(cancer_type)
        splits = self.create_stratified_splits(sample_graphs, n_splits=self.config.get('n_folds', 5))
        fold_results = []
        for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold + 1}/{len(splits)}")
            train_loader, val_loader, test_loader = self.create_data_loaders(sample_graphs, train_idx, val_idx, test_idx)
            input_dim = sample_graphs[0].x.shape[1]
            num_classes = len(np.unique([g.y.item() for g in sample_graphs]))
            num_edge_types = sample_graphs[0].edge_attr.shape[1] if sample_graphs[0].edge_attr is not None else 1
            model = get_enhanced_model(
                model_type=self.config['model'],
                input_dim=input_dim,
                hidden_dim=self.config.get('hidden_dim', 128),
                output_dim=num_classes,
                num_layers=self.config.get('num_layers', 3),
                dropout=self.config.get('dropout', 0.3),
                use_edge_attr=sample_graphs[0].edge_attr is not None,
                num_edge_types=num_edge_types
            ).to(self.device)
            train_results = self.train_model(model, train_loader, val_loader, fold)
            if self.results['best_model_path']:
                model.load_state_dict(torch.load(self.results['best_model_path']))
            test_results = self.evaluate_model(model, test_loader, fold)
            fold_results.append({
                'fold': fold,
                'train_results': train_results,
                'test_results': test_results
            })
        aggregated_results = self.aggregate_cv_results(fold_results)
        self.save_results(aggregated_results, cancer_type)
        return aggregated_results
    
    def aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate cross-validation results
        """
        test_metrics = [fold['test_results'] for fold in fold_results]
        
        aggregated = {
            'mean_accuracy': np.mean([m['accuracy'] for m in test_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in test_metrics]),
            'mean_f1': np.mean([m['f1'] for m in test_metrics]),
            'std_f1': np.std([m['f1'] for m in test_metrics]),
            'mean_precision': np.mean([m['precision'] for m in test_metrics]),
            'std_precision': np.std([m['precision'] for m in test_metrics]),
            'mean_recall': np.mean([m['recall'] for m in test_metrics]),
            'std_recall': np.std([m['recall'] for m in test_metrics]),
            'fold_results': fold_results
        }
        
        if 'auc' in test_metrics[0]:
            aggregated['mean_auc'] = np.mean([m['auc'] for m in test_metrics])
            aggregated['std_auc'] = np.std([m['auc'] for m in test_metrics])
        
        logger.info(f"Aggregated Results: {aggregated}")
        
        return aggregated
    
    def save_results(self, results: Dict, cancer_type: str):
        """
        Save results to file
        """
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"{cancer_type}_enhanced_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / f"{cancer_type}_enhanced_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Enhanced GNN Results for {cancer_type}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.config['model']}\n")
            f.write(f"Cross-validation folds: {self.config.get('n_folds', 5)}\n\n")
            f.write("Aggregated Metrics:\n")
            f.write(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}\n")
            f.write(f"F1-Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}\n")
            f.write(f"Precision: {results['mean_precision']:.4f} ± {results['std_precision']:.4f}\n")
            f.write(f"Recall: {results['mean_recall']:.4f} ± {results['std_recall']:.4f}\n")
            if 'mean_auc' in results:
                f.write(f"AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}\n")
        
        logger.info(f"Results saved to {results_file} and {summary_file}")

def main():
    """
    Main function to run enhanced training
    """
    # Configuration
    config = {
        'cancer_type': 'BRCA',
        'model': 'GAT',
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'weight_decay': 1e-4,
        'patience': 10,
        'n_folds': 5,
        'use_wandb': True,
        'project_name': 'enhanced-gnn-cancer'
    }
    
    # Initialize trainer
    trainer = EnhancedTrainer(config)
    
    # Run cross-validation
    results = trainer.run_cross_validation(config['cancer_type'])
    
    print(f"Enhanced training completed!")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Mean F1-Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")

if __name__ == "__main__":
    main() 