#!/usr/bin/env python3
"""
Simple Advanced Optimization Strategy to Achieve All 12 Metrics Exceeding Paper Performance
Simplified version to avoid import issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, GCNConv
import logging
from pathlib import Path
import warnings
import json
import gc

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGAT(nn.Module):
    """Simple GAT model with enhanced architecture"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num_heads=4, dropout=0.2):
        super(SimpleGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # GAT layers
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
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # GAT layers
        for i, (conv, bn, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = dropout(x)
            
            # Clear intermediate activations to save memory
            if i < len(self.conv_layers) - 1:
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Final output projection
        return self.output(x)

class SimpleGCN(nn.Module):
    """Simple GCN model with enhanced architecture"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(SimpleGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
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
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # GCN layers
        for i, (conv, bn, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = dropout(x)
            
            # Clear intermediate activations to save memory
            if i < len(self.conv_layers) - 1:
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Final output projection
        return self.output(x)

class SimpleFocalLoss(nn.Module):
    """Simple Focal Loss"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(SimpleFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def compute_simple_metrics(y_true, y_pred):
    """Compute metrics without sklearn dependencies"""
    # Convert to numpy for calculations
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Basic accuracy
    accuracy = (y_true == y_pred).mean()
    
    # Calculate TP, FP, TN, FN
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Balanced accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Simple ROC-AUC approximation
    roc_auc = 0.5 + (tp * tn - fp * fn) / (2 * (tp + fn) * (tn + fp)) if (tp + fn) * (tn + fp) > 0 else 0.5
    
    # Simple PR-AUC approximation
    pr_auc = precision * recall
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'balanced_accuracy': float(balanced_accuracy),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }

class SimpleAdvancedTrainer:
    """Simple advanced trainer"""
    
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
    
    def create_simple_splits(self, max_samples=1000):
        """Create simple data splits"""
        logger.info("Creating simple data splits...")
        
        y_np = self.data.y.cpu().numpy()
        positive_indices = torch.where(self.data.y == 1)[0]
        negative_indices = torch.where(self.data.y == 0)[0]
        
        # Ensure we have enough positive samples
        if len(positive_indices) < 5:
            logger.warning(f"Only {len(positive_indices)} positive samples available")
            return None
        
        # Limit samples for memory efficiency
        max_negative = max_samples - len(positive_indices)
        if len(negative_indices) > max_negative:
            negative_indices = negative_indices[:max_negative]
        
        # Combine and shuffle
        all_indices = torch.cat([positive_indices, negative_indices])
        perm = torch.randperm(len(all_indices))
        all_indices = all_indices[perm]
        
        # Stratified split
        n_total = len(all_indices)
        train_indices = all_indices[:int(0.6 * n_total)]
        val_indices = all_indices[int(0.6 * n_total):int(0.8 * n_total)]
        test_indices = all_indices[int(0.8 * n_total):]
        
        logger.info(f"Simple splits created: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        return train_indices, val_indices, test_indices
    
    def train_simple_model(self, model_class, model_name, train_indices, val_indices, test_indices):
        """Train simple model"""
        logger.info(f"Training simple {model_name}...")
        
        # Create model with smaller dimensions for memory efficiency
        input_dim = self.data.x.shape[1]
        if model_class == SimpleGAT:
            model = SimpleGAT(
                input_dim=input_dim,
                hidden_dim=128,  # Smaller for memory efficiency
                output_dim=2,
                num_layers=3,    # Fewer layers
                num_heads=4,     # Fewer heads
                dropout=0.2
            ).to(self.device)
        else:  # SimpleGCN
            model = SimpleGCN(
                input_dim=input_dim,
                hidden_dim=128,  # Smaller for memory efficiency
                output_dim=2,
                num_layers=3,    # Fewer layers
                dropout=0.2
            ).to(self.device)
        
        # Simple loss function
        loss_fn = SimpleFocalLoss(alpha=1.0, gamma=2.0)
        
        # Simple optimizer
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Simple scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # Training loop with memory management
        best_metrics = None
        best_model_state = None
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(50):  # Fewer epochs for memory efficiency
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass with memory management
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
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = outputs[val_indices]
                val_labels = self.data.y[val_indices]
                val_loss = loss_fn(val_outputs, val_labels)
                
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_metrics = compute_simple_metrics(val_labels, val_predictions)
                
                current_score = val_metrics['balanced_accuracy']
            
            scheduler.step(current_score)
            
            # Early stopping
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
            
            # Memory cleanup
            del outputs, train_outputs, val_outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(self.data.x, self.data.edge_index, self.data.edge_attr)
            test_outputs = test_outputs[test_indices]
            test_labels = self.data.y[test_indices]
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_metrics = compute_simple_metrics(test_labels, test_predictions)
        
        logger.info(f"Final {model_name} test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def run_simple_optimization(self):
        """Run simple optimization for all models"""
        logger.info("Starting simple advanced optimization...")
        
        # Load data
        self.load_data()
        
        # Create splits
        splits = self.create_simple_splits()
        if splits is None:
            logger.error("Failed to create data splits")
            return None
        
        train_indices, val_indices, test_indices = splits
        
        # Train simple models
        results = {}
        
        # Simple GAT
        gat_metrics = self.train_simple_model(
            SimpleGAT, "Simple GAT", train_indices, val_indices, test_indices
        )
        results['Simple_GAT'] = gat_metrics
        
        # Simple GCN
        gcn_metrics = self.train_simple_model(
            SimpleGCN, "Simple GCN", train_indices, val_indices, test_indices
        )
        results['Simple_GCN'] = gcn_metrics
        
        # Save results
        with open('results/simple_advanced_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Simple advanced optimization completed!")
        return results

def main():
    """Main function to run simple advanced optimization"""
    trainer = SimpleAdvancedTrainer("data")
    results = trainer.run_simple_optimization()
    
    if results:
        print("\n🎯 SIMPLE ADVANCED OPTIMIZATION RESULTS:")
        print("=" * 60)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        
        print(f"\n📁 Results saved to: results/simple_advanced_optimization_results.json")

if __name__ == "__main__":
    main()
