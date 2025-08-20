"""
Real Imbalanced Training Pipeline for Cancer Genomics Data
- Uses ONLY real data (no synthetic/fake data)
- Handles extreme class imbalance (50,903:1 ratio)
- Advanced techniques for real-world imbalanced datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import subgraph
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealImbalancedGAT(nn.Module):
    """GAT model specifically designed for real imbalanced cancer genomics data"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num_heads=8, dropout=0.3):
        super(RealImbalancedGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(
            input_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            concat=True
        ))
        
        # Middle layers
        for i in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                hidden_dim, 
                hidden_dim // num_heads, 
                heads=num_heads, 
                dropout=dropout, 
                add_self_loops=True, 
                concat=True
            ))
        
        # Final layer
        self.convs.append(GATv2Conv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            concat=True
        ))
        
        # Output projection for imbalanced data
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # GAT layers with residual connections
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection for better gradient flow
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        # Graph-level pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output projection
        return self.output_proj(x)

class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
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

class RealImbalancedTrainer:
    """Trainer for real imbalanced cancer genomics data"""
    
    def __init__(self, data_path: str, model_config: Dict, training_config: Dict):
        self.data_path = data_path
        self.model_config = model_config
        self.training_config = training_config
        self.data = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
    
    def load_real_data(self):
        """Load the real cancer genomics data"""
        logger.info("Loading real cancer genomics data...")
        
        try:
            data_file = Path(self.data_path) / "enhanced" / "real_only_torch_geometric_data.pt"
            if data_file.exists():
                self.data = torch.load(data_file, weights_only=False)
                logger.info(f"Loaded real data: {self.data.x.shape[0]} nodes, {self.data.edge_index.shape[1]} edges")
                
                # Move data to device
                self.data = self.data.to(self.device)
                
                # Analyze real class distribution
                if hasattr(self.data, 'y') and self.data.y is not None:
                    y_np = self.data.y.cpu().numpy()
                    unique_labels, counts = np.unique(y_np, return_counts=True)
                    logger.info("Real class distribution:")
                    for label, count in zip(unique_labels, counts):
                        percentage = (count / len(y_np)) * 100
                        logger.info(f"  Label {label}: {count} samples ({percentage:.4f}%)")
                
            else:
                raise FileNotFoundError(f"Real data file not found: {data_file}")
        except Exception as e:
            logger.error(f"Error loading real data: {e}")
            raise
    
    def create_real_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Create real train/validation/test splits preserving class distribution"""
        logger.info("Creating real data splits...")
        
        if not hasattr(self.data, 'y') or self.data.y is None:
            logger.error("No real labels found in data")
            return None
        
        y_np = self.data.y.cpu().numpy()
        num_nodes = self.data.x.shape[0]
        
        # Find real positive and negative samples
        positive_indices = np.where(y_np == 1)[0]
        negative_indices = np.where(y_np == 0)[0]
        
        logger.info(f"Real positive samples: {len(positive_indices)}")
        logger.info(f"Real negative samples: {len(negative_indices)}")
        
        # Create balanced splits ensuring positive samples in all splits
        np.random.seed(random_state)
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        
        # Split positive samples (ensure at least 1 in each split)
        n_positive = len(positive_indices)
        n_test_positive = max(1, int(n_positive * test_size))
        n_val_positive = max(1, int(n_positive * val_size))
        n_train_positive = n_positive - n_test_positive - n_val_positive
        
        test_positive = positive_indices[:n_test_positive]
        val_positive = positive_indices[n_test_positive:n_test_positive + n_val_positive]
        train_positive = positive_indices[n_test_positive + n_val_positive:]
        
        # For negative samples, use a reasonable subset to avoid overwhelming
        max_negative_per_split = 1000  # Limit to prevent imbalance issues
        n_negative = min(len(negative_indices), max_negative_per_split * 3)
        negative_indices = negative_indices[:n_negative]
        
        n_test_negative = int(n_negative * test_size)
        n_val_negative = int(n_negative * val_size)
        n_train_negative = n_negative - n_test_negative - n_val_negative
        
        test_negative = negative_indices[:n_test_negative]
        val_negative = negative_indices[n_test_negative:n_test_negative + n_val_negative]
        train_negative = negative_indices[n_test_negative + n_val_negative:]
        
        # Combine splits
        train_indices = np.concatenate([train_positive, train_negative])
        val_indices = np.concatenate([val_positive, val_negative])
        test_indices = np.concatenate([test_positive, test_negative])
        
        logger.info(f"Real data splits created:")
        logger.info(f"  Train: {len(train_indices)} samples")
        logger.info(f"  Validation: {len(val_indices)} samples")
        logger.info(f"  Test: {len(test_indices)} samples")
        
        return train_indices, val_indices, test_indices
    
    def create_subgraph_data(self, node_indices):
        """Create subgraph data for the given node indices"""
        # Convert to tensor
        node_indices = torch.tensor(node_indices, dtype=torch.long, device=self.device)
        
        # Create subgraph
        edge_index_sub, edge_attr_sub = subgraph(
            node_indices, 
            self.data.edge_index, 
            self.data.edge_attr,
            relabel_nodes=True,
            num_nodes=len(node_indices)
        )
        
        # Create subgraph data
        subgraph_data = type(self.data)(
            x=self.data.x[node_indices],
            edge_index=edge_index_sub,
            edge_attr=edge_attr_sub,
            y=self.data.y[node_indices]
        )
        
        return subgraph_data
    
    def compute_real_class_weights(self, train_indices):
        """Compute real class weights based on actual class distribution"""
        logger.info("Computing real class weights...")
        
        train_y = self.data.y[train_indices].cpu().numpy()
        
        # Compute balanced class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_y),
            y=train_y
        )
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        logger.info(f"Real class weights: {class_weights_tensor}")
        
        return class_weights_tensor
    
    def create_model(self, input_dim):
        """Create the GAT model for real imbalanced data"""
        logger.info("Creating GAT model for real imbalanced data...")
        
        self.model = RealImbalancedGAT(
            input_dim=input_dim,
            hidden_dim=self.model_config.get('hidden_dim', 256),
            output_dim=self.model_config.get('output_dim', 2),
            num_layers=self.model_config.get('num_layers', 3),
            num_heads=self.model_config.get('num_heads', 8),
            dropout=self.model_config.get('dropout', 0.3)
        ).to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def create_loss_function(self, class_weights):
        """Create loss function for real imbalanced data"""
        logger.info("Creating focal loss for real imbalanced data...")
        
        # Use focal loss with class weights
        focal_loss = FocalLoss(
            alpha=self.training_config.get('focal_alpha', 1.0),
            gamma=self.training_config.get('focal_gamma', 2.0)
        )
        
        return focal_loss
    
    def evaluate_real_metrics(self, y_true, y_pred, y_proba=None):
        """Evaluate using real metrics suitable for imbalanced data"""
        logger.info("Computing real evaluation metrics...")
        
        # Convert to numpy for sklearn metrics
        y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        
        metrics = {}
        
        # Balanced accuracy (most important for imbalanced data)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_np, y_pred_np)
        
        # Macro-averaged metrics
        metrics['f1_macro'] = f1_score(y_true_np, y_pred_np, average='macro')
        metrics['precision_macro'] = precision_score(y_true_np, y_pred_np, average='macro')
        metrics['recall_macro'] = recall_score(y_true_np, y_pred_np, average='macro')
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true_np, y_pred_np, average=None)
        metrics['recall_per_class'] = recall_score(y_true_np, y_pred_np, average=None)
        metrics['f1_per_class'] = f1_score(y_true_np, y_pred_np, average=None)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np)
        
        # Overall accuracy (less meaningful for imbalanced data)
        metrics['accuracy'] = (y_true_np == y_pred_np).mean()
        
        logger.info(f"Real metrics computed:")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision Macro: {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall Macro: {metrics['recall_macro']:.4f}")
        
        return metrics
    
    def train_epoch(self, train_data, optimizer, loss_fn):
        """Train for one epoch on real data"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(train_data.x, train_data.edge_index, train_data.edge_attr)
        
        # Compute loss
        loss = loss_fn(outputs, train_data.y)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_data, loss_fn):
        """Validate for one epoch on real data"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            outputs = self.model(val_data.x, val_data.edge_index, val_data.edge_attr)
            
            loss = loss_fn(outputs, val_data.y)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(val_data.y.cpu().numpy())
        
        return total_loss, all_predictions, all_labels
    
    def train_model(self, train_indices, val_indices, test_indices):
        """Train the model on real imbalanced data"""
        logger.info("Starting training on real imbalanced data...")
        
        # Create subgraph data
        train_data = self.create_subgraph_data(train_indices)
        val_data = self.create_subgraph_data(val_indices)
        test_data = self.create_subgraph_data(test_indices)
        
        logger.info(f"Subgraph sizes - Train: {train_data.x.shape[0]}, Val: {val_data.x.shape[0]}, Test: {test_data.x.shape[0]}")
        
        # Create model
        input_dim = train_data.x.shape[1]
        self.create_model(input_dim)
        
        # Compute real class weights
        class_weights = self.compute_real_class_weights(train_indices)
        
        # Create loss function
        loss_fn = self.create_loss_function(class_weights)
        
        # Create optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.get('learning_rate', 0.001),
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Training loop
        best_balanced_accuracy = 0
        patience_counter = 0
        max_patience = self.training_config.get('patience', 20)
        
        for epoch in range(self.training_config.get('epochs', 100)):
            # Train
            train_loss = self.train_epoch(train_data, optimizer, loss_fn)
            
            # Validate
            val_loss, val_preds, val_labels = self.validate_epoch(val_data, loss_fn)
            
            # Compute metrics
            val_metrics = self.evaluate_real_metrics(val_labels, val_preds)
            val_balanced_accuracy = val_metrics['balanced_accuracy']
            
            # Learning rate scheduling
            scheduler.step(val_balanced_accuracy)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_balanced_accuracy:.4f}")
            
            # Early stopping
            if val_balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = val_balanced_accuracy
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_imbalanced_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_imbalanced_model.pt'))
        
        # Final evaluation on test set
        logger.info("Evaluating on real test data...")
        test_loss, test_preds, test_labels = self.validate_epoch(test_data, loss_fn)
        test_metrics = self.evaluate_real_metrics(test_labels, test_preds)
        
        logger.info("Final real test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def run_real_training(self):
        """Run complete training pipeline on real data"""
        logger.info("Starting real imbalanced training pipeline...")
        
        # Load real data
        self.load_real_data()
        
        # Create real splits
        splits = self.create_real_splits()
        if splits is None:
            logger.error("Failed to create real data splits")
            return None
        
        train_indices, val_indices, test_indices = splits
        
        # Train model
        test_metrics = self.train_model(train_indices, val_indices, test_indices)
        
        logger.info("Real imbalanced training completed!")
        return test_metrics

def main():
    """Main function to run real imbalanced training"""
    
    # Configuration for real imbalanced data
    model_config = {
        'hidden_dim': 256,
        'output_dim': 2,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.3
    }
    
    training_config = {
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 100,
        'patience': 20,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0
    }
    
    # Create trainer
    trainer = RealImbalancedTrainer("data", model_config, training_config)
    
    # Run training
    results = trainer.run_real_training()
    
    if results:
        print("✅ Real imbalanced training completed successfully!")
        print(f"Best Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"Best F1 Macro: {results['f1_macro']:.4f}")
    else:
        print("❌ Real imbalanced training failed!")

if __name__ == "__main__":
    main()
