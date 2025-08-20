import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import os
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedGATModel(nn.Module):
    """Enhanced GAT model with exact paper specifications"""
    
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 3, 
                 num_heads: int = 8, num_classes: int = 2, dropout: float = 0.5):
        super(EnhancedGATModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # GAT layers with exact paper specifications
        self.convs = nn.ModuleList()
        
        # First layer: input -> hidden
        self.convs.append(GATv2Conv(num_features, hidden_dim // num_heads, 
                                   heads=num_heads, dropout=dropout, concat=True))
        
        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // num_heads, 
                                       heads=num_heads, dropout=dropout, concat=True))
        
        # Final layer: hidden -> output
        self.convs.append(GATv2Conv(hidden_dim, num_classes, heads=1, 
                                   dropout=dropout, concat=False))
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        # Apply GAT layers with ELU activation and dropout
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.layer_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class EnhancedGCNModel(nn.Module):
    """Enhanced GCN model with exact paper specifications"""
    
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 3, 
                 num_classes: int = 2, dropout: float = 0.5):
        super(EnhancedGCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Final layer
        self.convs.append(GCNConv(hidden_dim, num_classes))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        # Apply GCN layers with ReLU activation and dropout
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.layer_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class EnhancedGraphSAGEModel(nn.Module):
    """Enhanced GraphSAGE model with exact paper specifications"""
    
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 3, 
                 num_classes: int = 2, dropout: float = 0.5):
        super(EnhancedGraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr='mean'))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
        
        # Final layer
        self.convs.append(SAGEConv(hidden_dim, num_classes, aggr='mean'))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(num_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        # Apply GraphSAGE layers with skip connections
        for i, conv in enumerate(self.convs[:-1]):
            # Skip connection
            skip = self.skip_connections[i](x)
            
            # GraphSAGE convolution
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.layer_norms[i](x)
            
            # Add skip connection
            x = x + skip
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class BalancedTrainer:
    """
    Enhanced trainer with balanced datasets and focal loss
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        
    def load_balanced_data(self, method='smote') -> Data:
        """Load balanced data from various methods"""
        logger.info(f"Loading {method} balanced data...")
        
        enhanced_dir = Path("data/enhanced")
        
        # Load balanced dataset
        with open(enhanced_dir / f"{method}_balanced_data.pkl", 'rb') as f:
            balanced_data = pickle.load(f)
        
        # Load original graph
        with open(enhanced_dir / "comprehensive_graph.pkl", 'rb') as f:
            graph = pickle.load(f)
        
        # Create PyTorch Geometric Data
        X = balanced_data['X']
        y = balanced_data['y']
        
        # Convert to tensors
        x = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Create edge index from original graph
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        edge_list = []
        for edge in graph.edges():
            edge_list.append([node_to_idx[edge[0]], node_to_idx[edge[1]]])
            edge_list.append([node_to_idx[edge[1]], node_to_idx[edge[0]]])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y_tensor)
        
        logger.info(f"Loaded {method} balanced data: {data.x.shape}, {data.edge_index.shape}, {data.y.shape}")
        
        # Analyze class distribution
        class_counts = Counter(y)
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        return data
    
    def load_class_weights(self) -> torch.Tensor:
        """Load pre-computed class weights"""
        try:
            with open('data/enhanced/class_weights.pkl', 'rb') as f:
                class_weights = pickle.load(f)
            logger.info(f"Loaded class weights: {class_weights}")
            return class_weights.to(self.device)
        except FileNotFoundError:
            logger.warning("Class weights not found, using default")
            return None
    
    def create_stratified_splits(self, data: Data, n_splits: int = 5) -> List[Tuple]:
        """Create stratified train/val/test splits (70/15/15)"""
        logger.info("Creating stratified splits...")
        
        # Get node indices
        node_indices = np.arange(data.x.size(0))
        labels = data.y.numpy()
        
        # Create stratified splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        splits = []
        for train_val_idx, test_idx in skf.split(node_indices, labels):
            # Split train_val into train and validation (70/15/15)
            train_val_labels = labels[train_val_idx]
            train_val_indices = train_val_idx
            
            # Create another split for train/val
            train_idx, val_idx = next(StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                                    .split(train_val_indices, train_val_labels))
            
            # Convert back to original indices
            train_idx = train_val_indices[train_idx]
            val_idx = train_val_indices[val_idx]
            
            splits.append((train_idx, val_idx, test_idx))
        
        logger.info(f"Created {len(splits)} stratified splits")
        return splits
    
    def train_model(self, model, data: Data, train_idx, val_idx, 
                   learning_rate: float = 0.001, weight_decay: float = 5e-4,
                   epochs: int = 100, patience: int = 10, 
                   loss_type: str = 'focal', class_weights: torch.Tensor = None) -> Dict:
        """Train model with focal loss or weighted cross-entropy"""
        
        model = model.to(self.device)
        data = data.to(self.device)
        
        # Optimizer: Adam with learning rate 0.001 and weight decay 5e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss function selection
        if loss_type == 'focal':
            criterion = FocalLoss(alpha=1, gamma=2)
            logger.info("Using Focal Loss")
        elif loss_type == 'weighted' and class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info("Using Weighted Cross-Entropy Loss")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using Standard Cross-Entropy Loss")
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                              factor=0.5, patience=5, verbose=False)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_balanced_accs = []
        val_balanced_accs = []
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[train_idx], data.y[train_idx])
            
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_idx], data.y[val_idx])
                
                # Calculate metrics
                train_pred = out[train_idx].argmax(dim=1)
                val_pred = val_out[val_idx].argmax(dim=1)
                
                train_acc = accuracy_score(data.y[train_idx].cpu(), train_pred.cpu())
                val_acc = accuracy_score(data.y[val_idx].cpu(), val_pred.cpu())
                
                # Balanced accuracy
                train_balanced_acc = balanced_accuracy_score(data.y[train_idx].cpu(), train_pred.cpu())
                val_balanced_acc = balanced_accuracy_score(data.y[val_idx].cpu(), val_pred.cpu())
            
            # Record history
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_balanced_accs.append(train_balanced_acc)
            val_balanced_accs.append(val_balanced_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                           f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                           f"Train Bal Acc: {train_balanced_acc:.4f}, Val Bal Acc: {val_balanced_acc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'train_balanced_accs': train_balanced_accs,
            'val_balanced_accs': val_balanced_accs,
            'best_epoch': len(train_losses) - patience_counter - 1
        }
    
    def evaluate_model(self, model, data: Data, test_idx) -> Dict:
        """Evaluate model performance with balanced metrics"""
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            test_out = out[test_idx]
            test_labels = data.y[test_idx]
            
            # Calculate probabilities
            probs = F.softmax(test_out, dim=1)
            
            # Calculate metrics
            pred = test_out.argmax(dim=1)
            
            accuracy = accuracy_score(test_labels.cpu(), pred.cpu())
            balanced_accuracy = balanced_accuracy_score(test_labels.cpu(), pred.cpu())
            
            # Handle cases where all predictions are the same class
            unique_preds = torch.unique(pred).cpu().numpy()
            unique_labels = torch.unique(test_labels).cpu().numpy()
            
            if len(unique_preds) == 1 and len(unique_labels) == 1:
                # All predictions and labels are the same class
                if unique_preds[0] == unique_labels[0]:
                    precision = 1.0
                    recall = 1.0
                    f1 = 1.0
                else:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
            elif len(unique_preds) == 1:
                # All predictions are the same class but labels are different
                precision = precision_score(test_labels.cpu(), pred.cpu(), average='binary', zero_division=0)
                recall = recall_score(test_labels.cpu(), pred.cpu(), average='binary', zero_division=0)
                f1 = f1_score(test_labels.cpu(), pred.cpu(), average='binary', zero_division=0)
            else:
                # Normal case with mixed predictions
                precision = precision_score(test_labels.cpu(), pred.cpu(), average='binary', zero_division=0)
                recall = recall_score(test_labels.cpu(), pred.cpu(), average='binary', zero_division=0)
                f1 = f1_score(test_labels.cpu(), pred.cpu(), average='binary', zero_division=0)
            
            # ROC-AUC and PR-AUC
            try:
                roc_auc = roc_auc_score(test_labels.cpu(), probs[:, 1].cpu())
            except:
                roc_auc = 0.5
            
            try:
                pr_auc = average_precision_score(test_labels.cpu(), probs[:, 1].cpu())
            except:
                pr_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'predictions': pred.cpu().numpy(),
            'probabilities': probs.cpu().numpy()
        }
    
    def run_comprehensive_evaluation(self, model_class, model_name: str, data: Data, 
                                   loss_type: str = 'focal', class_weights: torch.Tensor = None) -> Dict:
        """Run comprehensive evaluation with balanced data and focal loss"""
        logger.info(f"Running comprehensive evaluation for {model_name} with {loss_type} loss...")
        
        # Create splits
        splits = self.create_stratified_splits(data)
        
        # Use paper's hyperparameters
        best_params = {
            'hidden_dim': 64,
            'dropout': 0.5,
            'learning_rate': 0.001
        }
        
        # Separate model parameters from training parameters
        model_params = {k: v for k, v in best_params.items() if k != 'learning_rate'}
        learning_rate = best_params['learning_rate']
        
        # Full evaluation with best parameters
        fold_results = []
        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {i+1}/{len(splits)}")
            
            # Create model with best parameters
            model = model_class(num_features=data.x.size(1), **model_params)
            
            # Train model with focal loss or weighted loss
            train_result = self.train_model(model, data, train_idx, val_idx, 
                                          learning_rate=learning_rate, 
                                          loss_type=loss_type, 
                                          class_weights=class_weights)
            
            # Evaluate on test set
            test_result = self.evaluate_model(train_result['model'], data, test_idx)
            
            # Store results
            fold_result = {
                'fold': i,
                'best_params': best_params,
                'train_history': train_result,
                'test_metrics': test_result
            }
            fold_results.append(fold_result)
            
            logger.info(f"Fold {i+1} Results: F1={test_result['f1']:.4f}, "
                       f"Balanced Acc={test_result['balanced_accuracy']:.4f}, "
                       f"ROC-AUC={test_result['roc_auc']:.4f}, PR-AUC={test_result['pr_auc']:.4f}")
        
        # Aggregate results
        avg_metrics = {}
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            values = [fold['test_metrics'][metric] for fold in fold_results]
            avg_metrics[metric] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
        
        results = {
            'model_name': model_name,
            'best_params': best_params,
            'fold_results': fold_results,
            'average_metrics': avg_metrics,
            'loss_type': loss_type
        }
        
        logger.info(f"{model_name} Average Results: F1={avg_metrics['f1']:.4f}, "
                   f"Balanced Acc={avg_metrics['balanced_accuracy']:.4f}, "
                   f"ROC-AUC={avg_metrics['roc_auc']:.4f}, PR-AUC={avg_metrics['pr_auc']:.4f}")
        
        return results
    
    def save_results(self, results: Dict, method: str):
        """Save results to files"""
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Prepare JSON-serializable results
        json_results = {
            'model_name': results['model_name'],
            'best_params': results['best_params'],
            'average_metrics': {},
            'fold_results': [],
            'loss_type': results['loss_type'],
            'balancing_method': method
        }
        
        # Convert average_metrics to JSON-serializable format
        for key, value in results['average_metrics'].items():
            if isinstance(value, np.ndarray):
                json_results['average_metrics'][key] = value.tolist()
            else:
                json_results['average_metrics'][key] = value
        
        # Process fold results, removing non-serializable objects
        for fold_result in results['fold_results']:
            json_fold = {
                'fold': fold_result['fold'],
                'best_params': fold_result['best_params'],
                'test_metrics': {}
            }
            
            # Convert test_metrics to JSON-serializable format
            for key, value in fold_result['test_metrics'].items():
                if isinstance(value, np.ndarray):
                    json_fold['test_metrics'][key] = value.tolist()
                else:
                    json_fold['test_metrics'][key] = value
            
            # Add train history without the model object
            if 'train_history' in fold_result:
                train_history = fold_result['train_history']
                json_fold['train_history'] = {
                    'train_losses': train_history.get('train_losses', []),
                    'val_losses': train_history.get('val_losses', []),
                    'train_accuracies': train_history.get('train_accs', []),
                    'val_accuracies': train_history.get('val_accs', []),
                    'train_balanced_accuracies': train_history.get('train_balanced_accs', []),
                    'val_balanced_accuracies': train_history.get('val_balanced_accs', []),
                    'best_epoch': train_history.get('best_epoch', 0)
                }
            
            json_results['fold_results'].append(json_fold)
        
        # Save JSON results
        filename = f"{results['model_name']}_{method}_{results['loss_type']}_results.json"
        with open(f'results/{filename}', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save detailed metrics
        metrics_file = f'results/{results["model_name"]}_{method}_{results["loss_type"]}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results['average_metrics'], f, indent=2)
        
        logger.info(f"Results saved to results/{filename}")
        logger.info(f"Metrics saved to {metrics_file}")

def main():
    """Main function to run enhanced training with balanced datasets"""
    logger.info("Starting enhanced training with balanced datasets and focal loss...")
    
    # Initialize trainer
    trainer = BalancedTrainer(device='cpu')
    
    # Load class weights
    class_weights = trainer.load_class_weights()
    
    # Define models
    models = {
        'GAT': EnhancedGATModel,
        'GCN': EnhancedGCNModel,
        'GraphSAGE': EnhancedGraphSAGEModel
    }
    
    # Define balancing methods to test
    balancing_methods = ['smote', 'adasyn', 'undersampled', 'hybrid']
    loss_types = ['focal', 'weighted']
    
    # Run comprehensive evaluation for each combination
    all_results = {}
    
    for method in balancing_methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {method.upper()} balanced dataset")
        logger.info(f"{'='*60}")
        
        try:
            # Load balanced data
            data = trainer.load_balanced_data(method)
            
            for loss_type in loss_types:
                logger.info(f"\n{'='*40}")
                logger.info(f"Testing {loss_type.upper()} loss")
                logger.info(f"{'='*40}")
                
                for model_name, model_class in models.items():
                    logger.info(f"\nTraining {model_name} with {method} data and {loss_type} loss...")
                    
                    # Use class weights only for weighted loss
                    weights = class_weights if loss_type == 'weighted' else None
                    
                    results = trainer.run_comprehensive_evaluation(
                        model_class, model_name, data, 
                        loss_type=loss_type, 
                        class_weights=weights
                    )
                    
                    # Store results
                    key = f"{model_name}_{method}_{loss_type}"
                    all_results[key] = results
                    
                    # Save results
                    trainer.save_results(results, method)
        
        except FileNotFoundError as e:
            logger.warning(f"Balanced dataset {method} not found: {e}")
            continue
    
    # Print final comparison
    logger.info(f"\n{'='*100}")
    logger.info("FINAL RESULTS COMPARISON (BALANCED DATASETS)")
    logger.info(f"{'='*100}")
    
    for key, results in all_results.items():
        metrics = results['average_metrics']
        logger.info(f"{key:30} | F1: {metrics['f1']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | "
                   f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    
    logger.info("Enhanced training with balanced datasets complete!")

if __name__ == "__main__":
    main() 