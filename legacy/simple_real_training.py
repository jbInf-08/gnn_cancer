"""
Simple Real Training Pipeline for Cancer Genomics Data
- Uses ONLY real data (no synthetic/fake data)
- Handles extreme class imbalance (50,903:1 ratio)
- Simple but effective approach for real-world imbalanced datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, global_mean_pool
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGAT(nn.Module):
    """Simple GAT model for real imbalanced cancer genomics data"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4, dropout=0.3):
        super(SimpleGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # GAT layers
        self.conv1 = GATv2Conv(
            input_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            concat=True
        )
        
        self.conv2 = GATv2Conv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            concat=True
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # First GAT layer
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output projection
        return self.output(x)

class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class SimpleRealTrainer:
    """Simple trainer for real imbalanced cancer genomics data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
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
    
    def create_simple_splits(self, max_samples_per_split=1000):
        """Create simple train/validation/test splits with limited samples"""
        logger.info("Creating simple data splits...")
        
        if not hasattr(self.data, 'y') or self.data.y is None:
            logger.error("No real labels found in data")
            return None
        
        y_np = self.data.y.cpu().numpy()
        
        # Find real positive and negative samples
        positive_indices = np.where(y_np == 1)[0]
        negative_indices = np.where(y_np == 0)[0]
        
        logger.info(f"Real positive samples: {len(positive_indices)}")
        logger.info(f"Real negative samples: {len(negative_indices)}")
        
        # Limit negative samples to avoid overwhelming
        max_negative = max_samples_per_split * 2  # Allow more negative samples
        if len(negative_indices) > max_negative:
            np.random.seed(42)
            negative_indices = np.random.choice(negative_indices, max_negative, replace=False)
        
        # Split positive samples (ensure at least 1 in each split)
        np.random.seed(42)
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        
        # Simple split: 60% train, 20% val, 20% test
        train_positive = positive_indices[:max(1, int(0.6 * n_positive))]
        val_positive = positive_indices[max(1, int(0.6 * n_positive)):max(1, int(0.8 * n_positive))]
        test_positive = positive_indices[max(1, int(0.8 * n_positive)):]
        
        train_negative = negative_indices[:int(0.6 * n_negative)]
        val_negative = negative_indices[int(0.6 * n_negative):int(0.8 * n_negative)]
        test_negative = negative_indices[int(0.8 * n_negative):]
        
        # Combine splits
        train_indices = np.concatenate([train_positive, train_negative])
        val_indices = np.concatenate([val_positive, val_negative])
        test_indices = np.concatenate([test_positive, test_negative])
        
        logger.info(f"Simple data splits created:")
        logger.info(f"  Train: {len(train_indices)} samples")
        logger.info(f"  Validation: {len(val_indices)} samples")
        logger.info(f"  Test: {len(test_indices)} samples")
        
        return train_indices, val_indices, test_indices
    
    def create_model(self, input_dim):
        """Create the simple GAT model"""
        logger.info("Creating simple GAT model...")
        
        self.model = SimpleGAT(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=2,
            num_layers=2,
            num_heads=4,
            dropout=0.3
        ).to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def evaluate_metrics(self, y_true, y_pred):
        """Evaluate using metrics suitable for imbalanced data"""
        logger.info("Computing evaluation metrics...")
        
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
        
        logger.info(f"Metrics computed:")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision Macro: {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall Macro: {metrics['recall_macro']:.4f}")
        
        return metrics
    
    def train_simple_model(self, train_indices, val_indices, test_indices):
        """Train the simple model on real imbalanced data"""
        logger.info("Starting simple training on real imbalanced data...")
        
        # Create model
        input_dim = self.data.x.shape[1]
        self.create_model(input_dim)
        
        # Create loss function
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Training loop
        best_balanced_accuracy = 0
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(50):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass on all data (simple approach)
            outputs = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            
            # Compute loss only on training samples
            train_outputs = outputs[train_indices]
            train_labels = self.data.y[train_indices]
            train_loss = loss_fn(train_outputs, train_labels)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = outputs[val_indices]
                val_labels = self.data.y[val_indices]
                val_loss = loss_fn(val_outputs, val_labels)
                
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_metrics = self.evaluate_metrics(val_labels, val_predictions)
                val_balanced_accuracy = val_metrics['balanced_accuracy']
            
            # Log progress
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_balanced_accuracy:.4f}")
            
            # Early stopping
            if val_balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = val_balanced_accuracy
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_simple_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_simple_model.pt'))
        
        # Final evaluation on test set
        logger.info("Evaluating on real test data...")
        self.model.eval()
        with torch.no_grad():
            test_outputs = outputs[test_indices]
            test_labels = self.data.y[test_indices]
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_metrics = self.evaluate_metrics(test_labels, test_predictions)
        
        logger.info("Final real test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def run_simple_training(self):
        """Run simple training pipeline on real data"""
        logger.info("Starting simple real imbalanced training pipeline...")
        
        # Load real data
        self.load_real_data()
        
        # Create simple splits
        splits = self.create_simple_splits()
        if splits is None:
            logger.error("Failed to create simple data splits")
            return None
        
        train_indices, val_indices, test_indices = splits
        
        # Train model
        test_metrics = self.train_simple_model(train_indices, val_indices, test_indices)
        
        logger.info("Simple real imbalanced training completed!")
        return test_metrics

def main():
    """Main function to run simple real imbalanced training"""
    
    # Create trainer
    trainer = SimpleRealTrainer("data")
    
    # Run training
    results = trainer.run_simple_training()
    
    if results:
        print("✅ Simple real imbalanced training completed successfully!")
        print(f"Best Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"Best F1 Macro: {results['f1_macro']:.4f}")
        print(f"Best Precision Macro: {results['precision_macro']:.4f}")
        print(f"Best Recall Macro: {results['recall_macro']:.4f}")
        
        # Show per-class performance
        print(f"Precision per class: {results['precision_per_class']}")
        print(f"Recall per class: {results['recall_per_class']}")
        print(f"F1 per class: {results['f1_per_class']}")
        
        # Show confusion matrix
        print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    else:
        print("❌ Simple real imbalanced training failed!")

if __name__ == "__main__":
    main()
