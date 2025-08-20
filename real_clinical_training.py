import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealClinicalGATModel(torch.nn.Module):
    """
    Advanced GAT model specifically designed for real clinical data
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, 
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.2,
                 batch_norm: bool = True, skip_connections: bool = True):
        super(RealClinicalGATModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        
        # GAT layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        if batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
            if batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False))
        
        # Dropout
        self.dropout = dropout
        
        # Attention weights storage
        self.register_buffer('attention_weights', torch.tensor([]))
    
    def forward(self, x, edge_index, edge_attr=None):
        # Input layer
        x = self.convs[0](x, edge_index, edge_attr)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers with skip connections
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index, edge_attr)
            if hasattr(self, 'batch_norms') and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.skip_connections and x.shape == identity.shape:
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index, edge_attr)
        
        return x

class RealClinicalGCNModel(torch.nn.Module):
    """
    Advanced GCN model specifically designed for real clinical data
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.2, batch_norm: bool = True,
                 skip_connections: bool = True):
        super(RealClinicalGCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip_connections = skip_connections
        
        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, num_classes))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr=None):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers with skip connections
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            if hasattr(self, 'batch_norms') and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.skip_connections and x.shape == identity.shape:
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x

class RealClinicalGraphSAGEModel(torch.nn.Module):
    """
    Advanced GraphSAGE model specifically designed for real clinical data
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.2, batch_norm: bool = True,
                 skip_connections: bool = True, aggregator: str = 'mean'):
        super(RealClinicalGraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip_connections = skip_connections
        
        # GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        if batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            if batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggregator))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr=None):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers with skip connections
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            if hasattr(self, 'batch_norms') and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.skip_connections and x.shape == identity.shape:
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x

class RealClinicalTrainer:
    """
    Advanced trainer specifically designed for real clinical data
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced real clinical data"""
        class_counts = torch.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        return class_weights
    
    def train_epoch(self, train_data, optimizer, criterion, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = self.model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = criterion(out, train_data.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Calculate accuracy
        pred = out.argmax(dim=1)
        accuracy = (pred == train_data.y).float().mean().item()
        
        return loss.item(), accuracy
    
    def validate(self, val_data, criterion):
        """Validate the model"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(val_data.x, val_data.edge_index, val_data.edge_attr)
            loss = criterion(out, val_data.y)
            
            pred = out.argmax(dim=1)
            accuracy = (pred == val_data.y).float().mean().item()
            
        return loss.item(), accuracy
    
    def evaluate(self, test_data):
        """Evaluate the model on test data"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(test_data.x, test_data.edge_index, test_data.edge_attr)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            # Calculate metrics
            y_true = test_data.y.cpu().numpy()
            y_pred = preds.cpu().numpy()
            y_prob = probs.cpu().numpy()
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC and PR AUC
            if len(np.unique(y_true)) > 1:
                try:
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0])
                except:
                    roc_auc = 0.5
                
                try:
                    pr_auc = average_precision_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0])
                except:
                    pr_auc = 0.5
            else:
                roc_auc = 0.5
                pr_auc = 0.5
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'predictions': y_pred,
                'probabilities': y_prob,
                'true_labels': y_true
            }
    
    def train(self, train_data, val_data, num_epochs=200, learning_rate=0.01, 
              weight_decay=1e-4, patience=20):
        """Train the model with advanced techniques"""
        logger.info(f"Starting training with {num_epochs} epochs")
        
        # Move data to device
        train_data = train_data.to(self.device)
        val_data = val_data.to(self.device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Loss function with class weights
        class_weights = self.calculate_class_weights(train_data.y)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_data, optimizer, criterion, scheduler)
            
            # Validation
            val_loss, val_acc = self.validate(val_data, criterion)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        logger.info("Training completed")
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

def load_real_clinical_data(data_path: str = "data/real_processed/real_clinical_data.pt"):
    """Load real clinical data"""
    logger.info(f"Loading real clinical data from {data_path}")
    
    if not Path(data_path).exists():
        logger.error(f"Real clinical data not found at {data_path}")
        return None
    
    data = torch.load(data_path)
    logger.info(f"Loaded real clinical data: {data.num_nodes} patients, {data.edge_index.shape[1]} edges")
    return data

def create_train_val_test_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Create train/validation/test splits for real clinical data"""
    logger.info("Creating train/validation/test splits")
    
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    
    # Stratified split
    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=(val_ratio + test_ratio), 
            stratify=data.y.cpu().numpy(), random_state=random_state
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_ratio/(val_ratio + test_ratio),
            stratify=data.y[temp_idx].cpu().numpy(), random_state=random_state
        )
    except:
        # Fallback to random split if stratification fails
        logger.warning("Stratified split failed, using random split")
        train_idx, temp_idx = train_test_split(
            indices, test_size=(val_ratio + test_ratio), random_state=random_state
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_state
        )
    
    # Create subgraphs
    def create_subgraph_data(node_indices):
        subgraph_data = data.clone()
        subgraph_data.x = data.x[node_indices]
        subgraph_data.y = data.y[node_indices]
        
        # Filter edges
        node_set = set(node_indices)
        edge_list = []
        for i in range(data.edge_index.shape[1]):
            src, dst = data.edge_index[:, i].cpu().numpy()
            if src in node_set and dst in node_set:
                new_src = np.where(node_indices == src)[0][0]
                new_dst = np.where(node_indices == dst)[0][0]
                edge_list.append([new_src, new_dst])
        
        if edge_list:
            subgraph_data.edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            subgraph_data.edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # Filter edge attributes
        if hasattr(data, 'edge_attr') and len(data.edge_attr) > 0:
            edge_attr_list = []
            for i in range(data.edge_index.shape[1]):
                src, dst = data.edge_index[:, i].cpu().numpy()
                if src in node_set and dst in node_set:
                    edge_attr_list.append(data.edge_attr[i])
            
            if edge_attr_list:
                subgraph_data.edge_attr = torch.stack(edge_attr_list)
            else:
                subgraph_data.edge_attr = torch.tensor([]).reshape(0, data.edge_attr.shape[1])
        
        return subgraph_data
    
    train_data = create_subgraph_data(train_idx)
    val_data = create_subgraph_data(val_idx)
    test_data = create_subgraph_data(test_idx)
    
    logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_data, val_data, test_data

def train_and_evaluate_model(model_class, model_params, train_data, val_data, test_data, 
                           model_name="Model"):
    """Train and evaluate a model on real clinical data"""
    logger.info(f"Training {model_name}")
    
    # Create model
    model = model_class(**model_params)
    
    # Create trainer
    trainer = RealClinicalTrainer(model)
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        train_data, val_data, num_epochs=200
    )
    
    # Evaluate model
    metrics = trainer.evaluate(test_data)
    
    # Add training history
    metrics['training_history'] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs
    }
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR AUC: {metrics['pr_auc']:.4f}")
    
    return metrics

def create_results_visualization(results, output_dir):
    """Create visualization of real clinical results"""
    logger.info("Creating results visualization")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    model_names = list(results.keys())
    metrics = ['f1_score', 'accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc']
    
    # Filter out None values
    valid_metrics = []
    for metric in metrics:
        values = [results[model][metric] for model in model_names if results[model][metric] is not None]
        if values:
            valid_metrics.append(metric)
    
    if not valid_metrics:
        logger.warning("No valid metrics for visualization")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(valid_metrics[:6]):
        values = [results[model][metric] for model in model_names if results[model][metric] is not None]
        model_names_filtered = [model for model in model_names if results[model][metric] is not None]
        
        if values:
            bars = axes[i].bar(model_names_filtered, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_clinical_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create training history plots
    for model_name, result in results.items():
        if 'training_history' in result:
            history = result['training_history']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss plot
            ax1.plot(history['train_losses'], label='Train Loss')
            ax1.plot(history['val_losses'], label='Val Loss')
            ax1.set_title(f'{model_name} - Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Accuracy plot
            ax2.plot(history['train_accuracies'], label='Train Accuracy')
            ax2.plot(history['val_accuracies'], label='Val Accuracy')
            ax2.set_title(f'{model_name} - Training Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Main function for real clinical training"""
    logger.info("Starting real clinical data training")
    
    # Load real clinical data
    data = load_real_clinical_data()
    if data is None:
        logger.error("Failed to load real clinical data")
        return
    
    # Create train/val/test splits
    train_data, val_data, test_data = create_train_val_test_splits(data)
    
    # Model configurations
    input_dim = data.x.shape[1]
    num_classes = data.num_classes
    
    model_configs = {
        'RealClinicalGAT': {
            'model_class': RealClinicalGATModel,
            'params': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.2,
                'batch_norm': True,
                'skip_connections': True
            }
        },
        'RealClinicalGCN': {
            'model_class': RealClinicalGCNModel,
            'params': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'batch_norm': True,
                'skip_connections': True
            }
        },
        'RealClinicalGraphSAGE': {
            'model_class': RealClinicalGraphSAGEModel,
            'params': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'batch_norm': True,
                'skip_connections': True,
                'aggregator': 'mean'
            }
        }
    }
    
    # Train and evaluate all models
    results = {}
    
    for model_name, config in model_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            metrics = train_and_evaluate_model(
                config['model_class'],
                config['params'],
                train_data, val_data, test_data,
                model_name
            )
            results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # Save results
    results_dir = Path("results/real_clinical_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def to_py(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: to_py(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [to_py(item) for item in obj]
        return obj
    
    results_serializable = to_py(results)
    
    results_file = results_dir / "real_clinical_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Create visualizations
    create_results_visualization(results, results_dir)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("REAL CLINICAL TRAINING RESULTS SUMMARY")
    logger.info("="*60)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC: {metrics['pr_auc']:.4f}")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("Real clinical training completed successfully!")

if __name__ == "__main__":
    main() 