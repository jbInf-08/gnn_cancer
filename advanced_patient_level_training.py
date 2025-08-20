#!/usr/bin/env python3
"""
Advanced Patient-Level Training with All Critical Improvements
Implements all changes needed to close and surpass paper performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedGATModel(nn.Module):
    """Advanced GAT model matching paper specifications exactly"""
    
    def __init__(self, num_features, num_classes, hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1):
        super(AdvancedGATModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GAT layers with exact paper specifications
        self.gat_layers = nn.ModuleList()
        
        # Input layer
        self.gat_layers.append(GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Output layer
        self.gat_layers.append(GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Skip connections: only between hidden layers of the same size
        self.skip_connections = nn.ModuleList()
        for _ in range(num_layers - 3):
            self.skip_connections.append(nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention weights for interpretability
        self.register_buffer('attention_weights', torch.tensor([]))
        
    def forward(self, x, edge_index, edge_attr=None):
        # Store attention weights for analysis
        attention_weights_list = []
        
        # Input layer
        x = self.gat_layers[0](x, edge_index, edge_attr)
        x = self.batch_norms[0](x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Hidden layers with skip connections (only after the first hidden layer)
        for i in range(1, self.num_layers - 1):
            # For hidden layers after the first, use skip connection
            if i > 1:
                skip = self.skip_connections[i-2](x)
                x = self.gat_layers[i](x, edge_index, edge_attr)
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = self.dropout(x)
                x = x + skip
            else:
                x = self.gat_layers[i](x, edge_index, edge_attr)
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = self.dropout(x)
            
            # Store attention weights from the last layer
            if i == len(self.gat_layers) - 2:
                if hasattr(self.gat_layers[i], 'att_src'):
                    attention_weights_list.append(self.gat_layers[i].att_src)
        
        # Output layer
        x = self.gat_layers[-1](x, edge_index, edge_attr)
        
        # Update attention weights buffer
        if attention_weights_list:
            self.attention_weights = torch.cat(attention_weights_list, dim=0)
        
        return F.log_softmax(x, dim=1)

class AdvancedTrainer:
    """Advanced trainer with all critical improvements"""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        
        # AdamW optimizer (critical improvement)
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler (critical improvement)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6)
        
        # Gradient clipping (critical improvement)
        self.max_grad_norm = 1.0
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 20
        self.patience_counter = 0
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def calculate_class_weights(self, labels):
        """Calculate balanced class weights"""
        class_counts = torch.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights
    
    def train_epoch(self, train_loader, class_weights):
        """Train for one epoch with advanced techniques"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Calculate loss with class weights
            loss = F.nll_loss(output, batch.y, weight=class_weights.to(self.device))
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.max(1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
        
        # Update learning rate
        self.scheduler.step()
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader, class_weights):
        """Validate with balanced metrics"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                output = self.model(batch.x, batch.edge_index, batch.edge_attr)
                loss = F.nll_loss(output, batch.y, weight=class_weights.to(self.device))
                
                total_loss += loss.item()
                pred = output.max(1)[1]
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # Calculate balanced accuracy
        balanced_acc = self.calculate_balanced_accuracy(all_labels, all_preds)
        
        return total_loss / len(val_loader), correct / total, balanced_acc
    
    def calculate_balanced_accuracy(self, y_true, y_pred):
        """Calculate balanced accuracy"""
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred)
    
    def train(self, train_loader, val_loader, num_epochs=200):
        """Train with early stopping and advanced techniques"""
        logger.info("Starting advanced training...")
        
        # Calculate class weights
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch.y.numpy())
        class_weights = self.calculate_class_weights(torch.tensor(all_labels))
        logger.info(f"Class weights: {class_weights}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, class_weights)
            
            # Validation
            val_loss, val_acc, val_balanced_acc = self.validate(val_loader, class_weights)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_advanced_gat.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_advanced_gat.pt'))
        logger.info("Training completed!")
    
    def evaluate(self, test_loader):
        """Comprehensive evaluation"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                output = self.model(batch.x, batch.edge_index, batch.edge_attr)
                
                probs = torch.exp(output)
                pred = output.max(1)[1]
                
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'balanced_accuracy': self.calculate_balanced_accuracy(all_labels, all_preds),
        }
        # Only compute ROC/PR AUC if both classes are present
        unique_labels = set(all_labels)
        if len(unique_labels) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(all_labels, [p[1] for p in all_probs])
            except Exception as e:
                metrics['roc_auc'] = None
            try:
                metrics['pr_auc'] = average_precision_score(all_labels, [p[1] for p in all_probs])
            except Exception as e:
                metrics['pr_auc'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        
        return metrics, all_preds, all_probs, all_labels

def load_patient_level_data():
    """Load the processed patient-level data"""
    data_path = Path("data/patient_level")
    
    # Load the PyTorch Geometric data
    data = torch.load(data_path / "patient_level_data.pt")
    
    logger.info(f"Loaded data with {data.num_nodes} nodes and {data.num_edges} edges")
    logger.info(f"Feature dimension: {data.num_features}")
    logger.info(f"Number of classes: {len(set(data.y.tolist()))}")
    logger.info(f"Label distribution: {dict(zip(*np.unique(data.y.numpy(), return_counts=True)))}")
    
    return data

def create_train_val_test_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create 70/15/15 train/validation/test splits"""
    num_nodes = data.num_nodes
    
    # Create indices
    indices = np.arange(num_nodes)
    
    # For small datasets, use simple random split
    if num_nodes < 10:
        # Simple random split for small datasets
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_size = max(1, int(train_ratio * num_nodes))
        val_size = max(1, int(val_ratio * num_nodes))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    else:
        # Stratified split for larger datasets
        from sklearn.model_selection import train_test_split
        
        # First split: train vs rest
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size=1-train_ratio, 
            stratify=data.y.numpy(),
            random_state=42
        )
        
        # Second split: validation vs test
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio/(val_ratio + test_ratio),
            stratify=data.y[temp_indices].numpy(),
            random_state=42
        )
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # Create separate data objects with proper edge index mapping
    def create_subgraph_data(data, node_mask):
        # Get nodes in this split
        split_nodes = torch.where(node_mask)[0]
        
        # Create mapping from original node indices to new indices
        new_node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(split_nodes)}
        
        # Filter edges where both nodes are in this split
        edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[:, i]
            if src.item() in new_node_mapping and dst.item() in new_node_mapping:
                edge_mask[i] = True
        
        # Get filtered edges and attributes
        filtered_edges = data.edge_index[:, edge_mask]
        filtered_attrs = data.edge_attr[edge_mask] if data.edge_attr is not None else None
        
        # Remap edge indices to new node indices
        new_edges = torch.zeros_like(filtered_edges)
        for i in range(filtered_edges.size(1)):
            src, dst = filtered_edges[:, i]
            new_edges[0, i] = new_node_mapping[src.item()]
            new_edges[1, i] = new_node_mapping[dst.item()]
        
        return Data(
            x=data.x[split_nodes],
            edge_index=new_edges,
            edge_attr=filtered_attrs,
            y=data.y[split_nodes]
        )
    
    train_data = create_subgraph_data(data, train_mask)
    val_data = create_subgraph_data(data, val_mask)
    test_data = create_subgraph_data(data, test_mask)
    
    logger.info(f"Train: {len(train_indices)} samples")
    logger.info(f"Validation: {len(val_indices)} samples")
    logger.info(f"Test: {len(test_indices)} samples")
    
    return train_data, val_data, test_data

def main():
    """Main training function"""
    logger.info("Starting Advanced Patient-Level Training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Load data
    data = load_patient_level_data()
    
    # Create splits
    train_data, val_data, test_data = create_train_val_test_splits(data)
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=1, shuffle=False)
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False)
    
    # Initialize model with paper specifications
    num_classes = len(set(data.y.tolist()))
    model = AdvancedGATModel(
        num_features=data.num_features,
        num_classes=num_classes,
        hidden_dim=128,  # Paper specification
        num_layers=4,    # Paper specification
        num_heads=8,     # Paper specification
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, device, learning_rate=0.001, weight_decay=1e-5)
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=200)
    
    # Evaluate model
    metrics, preds, probs, labels = trainer.evaluate(test_loader)
    
    # Print results
    logger.info("=== FINAL RESULTS ===")
    for metric, value in metrics.items():
        if value is None:
            logger.info(f"{metric}: N/A")
        else:
            logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    def to_py(val):
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val
    results = {
        'metrics': {k: (v if v is None else float(v)) for k, v in metrics.items()},
        'predictions': [to_py(p) for p in preds],
        'probabilities': [to_py(p) for p in probs],
        'labels': [to_py(l) for l in labels],
        'training_history': {
            'train_losses': [float(x) for x in trainer.train_losses],
            'val_losses': [float(x) for x in trainer.val_losses],
            'train_accuracies': [float(x) for x in trainer.train_accuracies],
            'val_accuracies': [float(x) for x in trainer.val_accuracies]
        }
    }
    
    with open('results/advanced_patient_level_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_results_visualization(results, trainer)
    
    logger.info("Advanced training completed successfully!")

def create_results_visualization(results, trainer):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training history
    axes[0, 0].plot(trainer.train_losses, label='Train Loss')
    axes[0, 0].plot(trainer.val_losses, label='Validation Loss')
    axes[0, 0].set_title('Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy history
    axes[0, 1].plot(trainer.train_accuracies, label='Train Accuracy')
    axes[0, 1].plot(trainer.val_accuracies, label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy History')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Metrics comparison (filter out None values)
    metrics = results['metrics']
    metric_names = [k for k, v in metrics.items() if v is not None]
    metric_values = [v for v in metrics.values() if v is not None]
    
    axes[1, 0].bar(metric_names, metric_values)
    axes[1, 0].set_title('Final Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('results/advanced_patient_level_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 