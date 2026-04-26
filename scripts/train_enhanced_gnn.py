import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import time

# Import our models
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from gnn_cancer.models.enhanced_gnn_models import get_enhanced_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGNNTrainer:
    def __init__(self, data_path="data/enhanced/torch_geometric_data.pt",
                 output_dir="results/enhanced_gnn",
                 model_type="gat",
                 hidden_dim=128,
                 num_layers=3,
                 dropout=0.3,
                 lr=0.001,
                 weight_decay=5e-4,
                 epochs=100,
                 batch_size=1,  # For small dataset
                 use_wandb=True):
        
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_wandb = use_wandb
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="cancer-gnn-enhanced",
                config={
                    "model_type": model_type,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "batch_size": batch_size
                }
            )
        
        # Load data
        self.load_data()
        
        # Setup model
        self.setup_model()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def load_data(self):
        """Load the enhanced dataset."""
        logger.info("Loading enhanced dataset...")
        
        # Load PyTorch Geometric data with weights_only=False for compatibility
        self.data = torch.load(self.data_path, map_location=self.device, weights_only=False)
        
        logger.info(f"Dataset loaded:")
        logger.info(f"  - Node features: {self.data.x.shape}")
        logger.info(f"  - Edge index: {self.data.edge_index.shape}")
        logger.info(f"  - Labels: {self.data.y.shape}")
        
        # Split data into train/val/test
        n_samples = self.data.x.size(0)
        indices = list(range(n_samples))
        
        # Split: 60% train, 20% val, 20% test
        train_indices, temp_indices = train_test_split(indices, test_size=0.4, random_state=42, stratify=self.data.y.cpu().numpy())
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, 
                                                   stratify=self.data.y[temp_indices].cpu().numpy())
        
        # Create masks
        self.train_mask = torch.zeros(n_samples, dtype=torch.bool)
        self.val_mask = torch.zeros(n_samples, dtype=torch.bool)
        self.test_mask = torch.zeros(n_samples, dtype=torch.bool)
        
        self.train_mask[train_indices] = True
        self.val_mask[val_indices] = True
        self.test_mask[test_indices] = True
        
        logger.info(f"Data split:")
        logger.info(f"  - Train: {self.train_mask.sum().item()}")
        logger.info(f"  - Validation: {self.val_mask.sum().item()}")
        logger.info(f"  - Test: {self.test_mask.sum().item()}")
        
        # Create data loaders (for batch processing if needed)
        self.train_loader = DataLoader([self.data], batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader([self.data], batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader([self.data], batch_size=self.batch_size, shuffle=False)
        
    def setup_model(self):
        """Setup the GNN model."""
        logger.info(f"Setting up {self.model_type.upper()} model...")
        
        input_dim = self.data.x.size(1)
        output_dim = len(torch.unique(self.data.y))
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Output dimension: {output_dim}")
        
        # Create model
        self.model = get_enhanced_model(
            model_type=self.model_type,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_edge_attr=False  # We don't have edge attributes
        ).to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index)
            
            # Get predictions for training nodes only
            train_out = out[self.train_mask]
            train_y = batch.y[self.train_mask]
            
            # Calculate loss
            loss = self.criterion(train_out, train_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = train_out.argmax(dim=1)
            correct += pred.eq(train_y).sum().item()
            total += train_y.size(0)
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                out = self.model(batch.x, batch.edge_index)
                
                # Get predictions for validation nodes only
                val_out = out[self.val_mask]
                val_y = batch.y[self.val_mask]
                
                # Calculate loss
                loss = self.criterion(val_out, val_y)
                
                # Statistics
                total_loss += loss.item()
                pred = val_out.argmax(dim=1)
                correct += pred.eq(val_y).sum().item()
                total += val_y.size(0)
        
        return total_loss / len(self.val_loader), correct / total
    
    def test(self):
        """Test the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                out = self.model(batch.x, batch.edge_index)
                
                # Get predictions for test nodes only
                test_out = out[self.test_mask]
                test_y = batch.y[self.test_mask]
                
                # Get predictions
                pred = test_out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(test_y.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC (if binary classification)
        if len(np.unique(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = 0.0
        else:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, test_metrics, test_preds, test_labels):
        """Save training results."""
        # Save model
        torch.save(self.model.state_dict(), self.output_dir / f"{self.model_type}_best.pt")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        with open(self.output_dir / "training_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        # Save test results
        results = {
            'test_metrics': test_metrics,
            'test_predictions': test_preds,
            'test_labels': test_labels
        }
        
        with open(self.output_dir / "test_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([test_metrics])
        metrics_df.to_csv(self.output_dir / "test_metrics.csv", index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.2f}s): "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.output_dir / f"{self.model_type}_best.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(self.output_dir / f"{self.model_type}_best.pt"))
        
        # Test
        logger.info("Testing model...")
        test_preds, test_labels = self.test()
        test_metrics = self.calculate_metrics(test_labels, test_preds)
        
        # Print test results
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Save results
        self.save_results(test_metrics, test_preds, test_labels)
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Training complete!")
        return test_metrics

def main():
    """Main function to train the enhanced GNN."""
    logger.info("Starting enhanced GNN training...")
    
    # Create trainer
    trainer = EnhancedGNNTrainer(
        model_type="gat",  # Try different models: "gat", "sage", "gcn"
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        lr=0.001,
        weight_decay=5e-4,
        epochs=100,
        use_wandb=False  # Set to True if you have wandb set up
    )
    
    # Train model
    test_metrics = trainer.train()
    
    logger.info("Enhanced GNN training complete!")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main() 