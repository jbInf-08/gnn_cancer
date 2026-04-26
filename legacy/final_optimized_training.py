import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedGATModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=3, 
                 num_heads=8, dropout=0.2, batch_norm=True, skip_connections=True):
        super(OptimizedGATModel, self).__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
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
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
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

class OptimizedGCNModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=3, 
                 dropout=0.2, batch_norm=True, skip_connections=True):
        super(OptimizedGCNModel, self).__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
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
    
    def forward(self, x, edge_index):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
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

class OptimizedGraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=3, 
                 dropout=0.2, batch_norm=True, skip_connections=True, aggregator='mean'):
        super(OptimizedGraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
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
    
    def forward(self, x, edge_index):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
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

def train_and_evaluate_model(model, train_data, val_data, test_data, 
                           learning_rate=0.01, weight_decay=1e-4, num_epochs=200):
    """Train and evaluate a model with optimized hyperparameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Optimizer with optimized parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out, train_data.y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.edge_index)
            val_loss = criterion(val_out, val_data.y)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 20:  # Early stopping
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_out = model(test_data.x, test_data.edge_index)
        test_probs = F.softmax(test_out, dim=1)
        test_preds = test_out.argmax(dim=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_data.y.cpu(), test_preds.cpu())
        precision = precision_score(test_data.y.cpu(), test_preds.cpu(), average='weighted', zero_division=0)
        recall = recall_score(test_data.y.cpu(), test_preds.cpu(), average='weighted', zero_division=0)
        f1 = f1_score(test_data.y.cpu(), test_preds.cpu(), average='weighted', zero_division=0)
        
        # ROC AUC (if binary classification)
        if test_data.num_classes == 2:
            try:
                roc_auc = roc_auc_score(test_data.y.cpu(), test_probs[:, 1].cpu())
            except:
                roc_auc = 0.5
        else:
            roc_auc = 0.5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'val_loss': best_val_loss.item()
    }

def main():
    """Main function for final optimized training"""
    logger.info("Starting final optimized training with all improvements")
    
    # Load enhanced multi-modal data
    data_path = Path("data/enhanced_multi_modal/enhanced_multi_modal_data.pt")
    if not data_path.exists():
        logger.error("Enhanced multi-modal data not found. Please run the multi-modal processor first.")
        return
    
    data = torch.load(data_path)
    logger.info(f"Loaded data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    
    # Load best hyperparameters
    best_hyperparams = {
        'GAT': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.2, 
                'batch_norm': True, 'skip_connections': True, 'learning_rate': 0.01, 'weight_decay': 1e-4},
        'GCN': {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.2, 'batch_norm': True, 
                'skip_connections': True, 'learning_rate': 0.01, 'weight_decay': 1e-4},
        'GraphSAGE': {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.2, 'batch_norm': True, 
                      'skip_connections': True, 'aggregator': 'mean', 'learning_rate': 0.01, 'weight_decay': 1e-4}
    }
    
    # Create train/val/test splits
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    
    # 70/15/15 split
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=data.y.cpu().numpy(), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=data.y[temp_idx].cpu().numpy(), random_state=42)
    
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
        
        return subgraph_data
    
    train_data = create_subgraph_data(train_idx)
    val_data = create_subgraph_data(val_idx)
    test_data = create_subgraph_data(test_idx)
    
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Train all models with optimized hyperparameters
    results = {}
    
    for model_type in ['GAT', 'GCN', 'GraphSAGE']:
        logger.info(f"Training optimized {model_type}")
        
        # Create model
        if model_type == 'GAT':
            model = OptimizedGATModel(data.x.shape[1], data.num_classes, **{k: v for k, v in best_hyperparams[model_type].items() 
                                                                           if k not in ['learning_rate', 'weight_decay']})
        elif model_type == 'GCN':
            model = OptimizedGCNModel(data.x.shape[1], data.num_classes, **{k: v for k, v in best_hyperparams[model_type].items() 
                                                                           if k not in ['learning_rate', 'weight_decay']})
        elif model_type == 'GraphSAGE':
            model = OptimizedGraphSAGEModel(data.x.shape[1], data.num_classes, **{k: v for k, v in best_hyperparams[model_type].items() 
                                                                                if k not in ['learning_rate', 'weight_decay']})
        
        # Train and evaluate
        metrics = train_and_evaluate_model(
            model, train_data, val_data, test_data,
            learning_rate=best_hyperparams[model_type]['learning_rate'],
            weight_decay=best_hyperparams[model_type]['weight_decay']
        )
        
        results[model_type] = metrics
        logger.info(f"{model_type} - F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}")
    
    # Save results
    results_dir = Path("results/final_optimized_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "final_optimized_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Final optimized training completed. Results saved to {results_file}")
    
    # Print summary
    logger.info("\nFinal Results Summary:")
    logger.info("=" * 50)
    for model_type, metrics in results.items():
        logger.info(f"{model_type}:")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info("")

if __name__ == "__main__":
    main()
