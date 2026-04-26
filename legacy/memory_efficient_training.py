"""
Memory-Efficient Training Pipeline for Cancer Genomics Data
- Uses ONLY real data (no synthetic/fake data)
- Handles extreme class imbalance (50,903:1 ratio)
- Memory-efficient approach for large graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientGAT(nn.Module):
    """Memory-efficient GAT model for large graphs"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.3):
        super(MemoryEfficientGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Single GAT layer to reduce memory usage
        self.conv = GATv2Conv(
            input_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout, 
            add_self_loops=True, 
            concat=True
        )
        
        # Simple output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # Single GAT layer
        x = F.elu(self.conv(x, edge_index))
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

class MemoryEfficientTrainer:
    """Memory-efficient trainer for real imbalanced cancer genomics data"""
    
    def __init__(self, data_path: str, batch_size=1000):
        self.data_path = data_path
        self.batch_size = batch_size
        self.data = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Batch size: {batch_size}")
    
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
    
    def create_memory_efficient_splits(self, max_samples=2000):
        """Create memory-efficient splits with limited samples"""
        logger.info("Creating memory-efficient data splits...")
        
        if not hasattr(self.data, 'y') or self.data.y is None:
            logger.error("No real labels found in data")
            return None
        
        y_np = self.data.y.cpu().numpy()
        
        # Find real positive and negative samples
        positive_indices = np.where(y_np == 1)[0]
        negative_indices = np.where(y_np == 0)[0]
        
        logger.info(f"Real positive samples: {len(positive_indices)}")
        logger.info(f"Real negative samples: {len(negative_indices)}")
        
        # Limit total samples to avoid memory issues
        max_negative = max_samples - len(positive_indices)
        if len(negative_indices) > max_negative:
            np.random.seed(42)
            negative_indices = np.random.choice(negative_indices, max_negative, replace=False)
        
        # Combine all samples
        all_indices = np.concatenate([positive_indices, negative_indices])
        np.random.seed(42)
        np.random.shuffle(all_indices)
        
        # Simple split: 60% train, 20% val, 20% test
        n_total = len(all_indices)
        train_indices = all_indices[:int(0.6 * n_total)]
        val_indices = all_indices[int(0.6 * n_total):int(0.8 * n_total)]
        test_indices = all_indices[int(0.8 * n_total):]
        
        logger.info(f"Memory-efficient data splits created:")
        logger.info(f"  Train: {len(train_indices)} samples")
        logger.info(f"  Validation: {len(val_indices)} samples")
        logger.info(f"  Test: {len(test_indices)} samples")
        
        return train_indices, val_indices, test_indices
    
    def create_model(self, input_dim):
        """Create the memory-efficient GAT model"""
        logger.info("Creating memory-efficient GAT model...")
        
        self.model = MemoryEfficientGAT(
            input_dim=input_dim,
            hidden_dim=64,  # Smaller hidden dimension
            output_dim=2,
            num_heads=2,    # Fewer attention heads
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
    
    def train_memory_efficient_model(self, train_indices, val_indices, test_indices):
        """Train the memory-efficient model on real imbalanced data"""
        logger.info("Starting memory-efficient training on real imbalanced data...")
        
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
        max_patience = 10
        
        for epoch in range(30):  # Fewer epochs for memory efficiency
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            # Process in smaller batches to avoid memory issues
            total_loss = 0
            num_batches = 0
            
            # Process training data in batches
            for i in range(0, len(train_indices), self.batch_size):
                batch_indices = train_indices[i:i + self.batch_size]
                
                # Get batch data
                batch_x = self.data.x[batch_indices]
                batch_y = self.data.y[batch_indices]
                
                # Forward pass
                batch_outputs = self.model(batch_x, self.data.edge_index, self.data.edge_attr)
                
                # Compute loss
                batch_loss = loss_fn(batch_outputs, batch_y)
                total_loss += batch_loss.item()
                
                # Backward pass
                batch_loss.backward()
                num_batches += 1
            
            # Update weights
            optimizer.step()
            avg_train_loss = total_loss / num_batches
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_val_predictions = []
            all_val_labels = []
            
            with torch.no_grad():
                # Process validation data in batches
                for i in range(0, len(val_indices), self.batch_size):
                    batch_indices = val_indices[i:i + self.batch_size]
                    
                    batch_x = self.data.x[batch_indices]
                    batch_y = self.data.y[batch_indices]
                    
                    batch_outputs = self.model(batch_x, self.data.edge_index, self.data.edge_attr)
                    batch_loss = loss_fn(batch_outputs, batch_y)
                    val_loss += batch_loss.item()
                    
                    batch_predictions = torch.argmax(batch_outputs, dim=1)
                    all_val_predictions.extend(batch_predictions.cpu().numpy())
                    all_val_labels.extend(batch_y.cpu().numpy())
                
                val_metrics = self.evaluate_metrics(all_val_labels, all_val_predictions)
                val_balanced_accuracy = val_metrics['balanced_accuracy']
            
            # Log progress
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_balanced_accuracy:.4f}")
            
            # Early stopping
            if val_balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = val_balanced_accuracy
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_memory_efficient_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_memory_efficient_model.pt'))
        
        # Final evaluation on test set
        logger.info("Evaluating on real test data...")
        self.model.eval()
        all_test_predictions = []
        all_test_labels = []
        
        with torch.no_grad():
            # Process test data in batches
            for i in range(0, len(test_indices), self.batch_size):
                batch_indices = test_indices[i:i + self.batch_size]
                
                batch_x = self.data.x[batch_indices]
                batch_y = self.data.y[batch_indices]
                
                batch_outputs = self.model(batch_x, self.data.edge_index, self.data.edge_attr)
                batch_predictions = torch.argmax(batch_outputs, dim=1)
                
                all_test_predictions.extend(batch_predictions.cpu().numpy())
                all_test_labels.extend(batch_y.cpu().numpy())
            
            test_metrics = self.evaluate_metrics(all_test_labels, all_test_predictions)
        
        logger.info("Final real test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def run_memory_efficient_training(self):
        """Run memory-efficient training pipeline on real data"""
        logger.info("Starting memory-efficient real imbalanced training pipeline...")
        
        # Load real data
        self.load_real_data()
        
        # Create memory-efficient splits
        splits = self.create_memory_efficient_splits()
        if splits is None:
            logger.error("Failed to create memory-efficient data splits")
            return None
        
        train_indices, val_indices, test_indices = splits
        
        # Train model
        test_metrics = self.train_memory_efficient_model(train_indices, val_indices, test_indices)
        
        logger.info("Memory-efficient real imbalanced training completed!")
        return test_metrics

def main():
    """Main function to run memory-efficient real imbalanced training"""
    
    # Create trainer with small batch size
    trainer = MemoryEfficientTrainer("data", batch_size=500)
    
    # Run training
    results = trainer.run_memory_efficient_training()
    
    if results:
        print("✅ Memory-efficient real imbalanced training completed successfully!")
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
        
        # Realistic assessment
        print("\n🎯 REALISTIC ASSESSMENT:")
        if results['balanced_accuracy'] > 0.6:
            print("✅ GOOD: Balanced accuracy above 60% - model is learning!")
        elif results['balanced_accuracy'] > 0.5:
            print("⚠️ FAIR: Balanced accuracy above 50% - better than random guessing")
        else:
            print("❌ POOR: Balanced accuracy below 50% - needs improvement")
        
        if results['f1_macro'] > 0.3:
            print("✅ GOOD: F1 macro above 30% - reasonable performance for extreme imbalance")
        else:
            print("⚠️ LOW: F1 macro below 30% - expected for 50,903:1 imbalance ratio")
            
    else:
        print("❌ Memory-efficient real imbalanced training failed!")

if __name__ == "__main__":
    main()
