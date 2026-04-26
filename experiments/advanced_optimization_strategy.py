#!/usr/bin/env python3
"""
Advanced Optimization Strategy to Achieve All 12 Metrics Exceeding Paper Performance
Comprehensive approach to improve GAT and GCN models to match/exceed paper performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import warnings
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedGAT(nn.Module):
    """Advanced GAT model with enhanced architecture for better performance"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num_heads=8, dropout=0.2):
        super(AdvancedGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Enhanced GAT layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.conv_layers.append(GATv2Conv(
            input_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            concat=True
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATv2Conv(
                hidden_dim, 
                hidden_dim // num_heads, 
                heads=num_heads, 
                dropout=dropout, 
                add_self_loops=True, 
                concat=True
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output layer
        self.conv_layers.append(GATv2Conv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            concat=True
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Enhanced output projection with multiple layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Skip connection for input
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Input projection for skip connection
        x_input = self.input_projection(x)
        
        # GAT layers with residual connections
        for i, (conv, bn, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            if i == 0:
                # First layer
                x = conv(x, edge_index)
                x = bn(x)
                x = F.elu(x)
                x = dropout(x)
            else:
                # Subsequent layers with residual connections
                residual = x
                x = conv(x, edge_index)
                x = bn(x)
                x = F.elu(x + residual)  # Residual connection
                x = dropout(x)
        
        # Final output projection
        return self.output(x)

class AdvancedGCN(nn.Module):
    """Advanced GCN model with enhanced architecture for better performance"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(AdvancedGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Enhanced GCN layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output layer
        self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Enhanced output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Skip connection for input
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Input projection for skip connection
        x_input = self.input_projection(x)
        
        # GCN layers with residual connections
        for i, (conv, bn, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            if i == 0:
                # First layer
                x = conv(x, edge_index)
                x = bn(x)
                x = F.elu(x)
                x = dropout(x)
            else:
                # Subsequent layers with residual connections
                residual = x
                x = conv(x, edge_index)
                x = bn(x)
                x = F.elu(x + residual)  # Residual connection
                x = dropout(x)
        
        # Final output projection
        return self.output(x)

class AdvancedFocalLoss(nn.Module):
    """Advanced Focal Loss with adaptive alpha and gamma"""
    
    def __init__(self, alpha=1.0, gamma=2.0, adaptive=True):
        super(AdvancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.adaptive:
            # Adaptive alpha based on class distribution
            batch_size = targets.size(0)
            pos_count = (targets == 1).sum().float()
            neg_count = (targets == 0).sum().float()
            
            if pos_count > 0 and neg_count > 0:
                pos_ratio = pos_count / batch_size
                adaptive_alpha = 1.0 / pos_ratio  # Higher weight for minority class
            else:
                adaptive_alpha = self.alpha
        else:
            adaptive_alpha = self.alpha
        
        focal_loss = adaptive_alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class AdvancedOptimizationTrainer:
    """Advanced trainer with comprehensive optimization strategies"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_data(self):
        """Load the real cancer genomics data"""
        logger.info("Loading real cancer genomics data...")
        
        try:
            data_file = Path(self.data_path) / "enhanced" / "real_only_torch_geometric_data.pt"
            if data_file.exists():
                self.data = torch.load(data_file, weights_only=False)
                logger.info(f"Loaded data: {self.data.x.shape[0]} nodes, {self.data.edge_index.shape[1]} edges")
                self.data = self.data.to(self.device)
            else:
                raise FileNotFoundError(f"Data file not found: {data_file}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_advanced_splits(self, max_samples=2000):
        """Create advanced data splits with better balance"""
        logger.info("Creating advanced data splits...")
        
        y_np = self.data.y.cpu().numpy()
        positive_indices = np.where(y_np == 1)[0]
        negative_indices = np.where(y_np == 0)[0]
        
        # Ensure we have enough positive samples
        if len(positive_indices) < 10:
            logger.warning(f"Only {len(positive_indices)} positive samples available")
            return None
        
        # Limit negative samples while maintaining balance
        max_negative = max_samples - len(positive_indices)
        if len(negative_indices) > max_negative:
            np.random.seed(42)
            negative_indices = np.random.choice(negative_indices, max_negative, replace=False)
        
        # Combine and shuffle
        all_indices = np.concatenate([positive_indices, negative_indices])
        np.random.seed(42)
        np.random.shuffle(all_indices)
        
        # Stratified split
        n_total = len(all_indices)
        train_indices = all_indices[:int(0.6 * n_total)]
        val_indices = all_indices[int(0.6 * n_total):int(0.8 * n_total)]
        test_indices = all_indices[int(0.8 * n_total):]
        
        logger.info(f"Advanced splits created: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        return train_indices, val_indices, test_indices
    
    def train_advanced_model(self, model_class, model_name, train_indices, val_indices, test_indices):
        """Train advanced model with comprehensive optimization"""
        logger.info(f"Training advanced {model_name}...")
        
        # Create model
        input_dim = self.data.x.shape[1]
        if model_class == AdvancedGAT:
            model = AdvancedGAT(
                input_dim=input_dim,
                hidden_dim=256,  # Larger hidden dimension
                output_dim=2,
                num_layers=4,    # More layers
                num_heads=8,     # More attention heads
                dropout=0.2      # Lower dropout
            ).to(self.device)
        else:  # AdvancedGCN
            model = AdvancedGCN(
                input_dim=input_dim,
                hidden_dim=256,  # Larger hidden dimension
                output_dim=2,
                num_layers=4,    # More layers
                dropout=0.2      # Lower dropout
            ).to(self.device)
        
        # Advanced loss function
        loss_fn = AdvancedFocalLoss(alpha=1.0, gamma=2.0, adaptive=True)
        
        # Advanced optimizer with different learning rates
        optimizer = optim.AdamW([
            {'params': model.conv_layers.parameters(), 'lr': 0.001},
            {'params': model.output.parameters(), 'lr': 0.0005},
            {'params': model.batch_norms.parameters(), 'lr': 0.001}
        ], weight_decay=0.01)
        
        # Advanced scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training loop with advanced techniques
        best_metrics = None
        best_model_state = None
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(100):
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(self.data.x, self.data.edge_index, self.data.edge_attr)
            
            # Compute loss on training samples
            train_outputs = outputs[train_indices]
            train_labels = self.data.y[train_indices]
            train_loss = loss_fn(train_outputs, train_labels)
            
            # Backward pass
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = outputs[val_indices]
                val_labels = self.data.y[val_indices]
                val_loss = loss_fn(val_outputs, val_labels)
                
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_metrics = self.compute_metrics(val_labels, val_predictions)
                
                # Use balanced accuracy for early stopping
                current_score = val_metrics['balanced_accuracy']
            
            # Early stopping with patience
            if best_metrics is None or current_score > best_metrics['balanced_accuracy']:
                best_metrics = val_metrics.copy()
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {current_score:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = outputs[test_indices]
            test_labels = self.data.y[test_indices]
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_metrics = self.compute_metrics(test_labels, test_predictions)
        
        logger.info(f"Final {model_name} test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def compute_metrics(self, y_true, y_pred):
        """Compute comprehensive metrics"""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        metrics = {}
        metrics['accuracy'] = (y_true_np == y_pred_np).mean()
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_np, y_pred_np)
        metrics['f1_score'] = f1_score(y_true_np, y_pred_np, average='macro')
        metrics['precision'] = precision_score(y_true_np, y_pred_np, average='macro')
        metrics['recall'] = recall_score(y_true_np, y_pred_np, average='macro')
        
        # ROC-AUC and PR-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_np)
        except:
            metrics['roc_auc'] = 0.5
        
        try:
            metrics['pr_auc'] = average_precision_score(y_true_np, y_pred_np)
        except:
            metrics['pr_auc'] = 0.0
        
        return metrics
    
    def run_advanced_optimization(self):
        """Run advanced optimization for all models"""
        logger.info("Starting advanced optimization...")
        
        # Load data
        self.load_data()
        
        # Create splits
        splits = self.create_advanced_splits()
        if splits is None:
            logger.error("Failed to create data splits")
            return None
        
        train_indices, val_indices, test_indices = splits
        
        # Train advanced models
        results = {}
        
        # Advanced GAT
        gat_metrics = self.train_advanced_model(
            AdvancedGAT, "Advanced GAT", train_indices, val_indices, test_indices
        )
        results['Advanced_GAT'] = gat_metrics
        
        # Advanced GCN
        gcn_metrics = self.train_advanced_model(
            AdvancedGCN, "Advanced GCN", train_indices, val_indices, test_indices
        )
        results['Advanced_GCN'] = gcn_metrics
        
        # Save results
        with open('results/advanced_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Advanced optimization completed!")
        return results

def main():
    """Main function to run advanced optimization"""
    trainer = AdvancedOptimizationTrainer("data")
    results = trainer.run_advanced_optimization()
    
    if results:
        print("\n🎯 ADVANCED OPTIMIZATION RESULTS:")
        print("=" * 50)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        
        print(f"\n📁 Results saved to: results/advanced_optimization_results.json")

if __name__ == "__main__":
    main()
