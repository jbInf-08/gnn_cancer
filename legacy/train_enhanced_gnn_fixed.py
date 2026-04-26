import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
import wandb
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools
from sklearn.model_selection import StratifiedKFold
from scripts.integrate_real_data import RealDataIntegrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

class SimpleGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleGAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=0.3))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.3))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * 4, output_dim, heads=1, dropout=0.3))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleGCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleGraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

def get_model(model_type, input_dim, hidden_dim, output_dim, num_layers=2):
    """Get model based on type."""
    if model_type.upper() == 'GAT':
        return SimpleGAT(input_dim, hidden_dim, output_dim, num_layers)
    elif model_type.upper() == 'GCN':
        return SimpleGCN(input_dim, hidden_dim, output_dim, num_layers)
    elif model_type.upper() == 'GRAPHSAGE':
        return SimpleGraphSAGE(input_dim, hidden_dim, output_dim, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model, data, train_mask, val_mask, device, num_epochs=50):
    """Train model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_f1 = 0
    patience_counter = 0
    patience = 10
    history = {'train_loss': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[val_mask].argmax(dim=1)
            val_true = data.y[val_mask]
            val_f1 = f1_score(val_true.cpu().numpy(), val_pred.cpu().numpy(), average='binary', zero_division=0)
            
            scheduler.step(val_f1)
            history['train_loss'].append(loss.item())
            history['val_f1'].append(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    return history, best_val_f1

def evaluate_model(model, data, test_mask, device):
    """Evaluate model on test set."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = out[test_mask].argmax(dim=1)
        test_true = data.y[test_mask]
        test_probs = torch.softmax(out[test_mask], dim=1)[:, 1]  # Probability of positive class
        
        # Convert to numpy
        test_pred_np = test_pred.cpu().numpy()
        test_true_np = test_true.cpu().numpy()
        test_probs_np = test_probs.cpu().numpy()
        
        # Calculate metrics
        accuracy = (test_pred == test_true).float().mean().item()
        precision = precision_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        recall = recall_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        f1 = f1_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(test_true_np, test_probs_np)
        except ValueError:
            roc_auc = 0.0
        
        try:
            pr_auc = average_precision_score(test_true_np, test_probs_np)
        except ValueError:
            pr_auc = 0.0
        
        cm = confusion_matrix(test_true_np, test_pred_np)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'class_counts': {'positives': int(sum(test_true_np==1)), 'negatives': int(sum(test_true_np==0))}
        }

def plot_learning_curves(history, fold, model_type, ablation_name):
    """Plot learning curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / F1')
    plt.title(f'Learning Curves - {model_type} {ablation_name} (Fold {fold+1})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/learning_curve_{model_type.lower()}_{ablation_name}_fold{fold+1}.png')
    plt.close()

def ablation_configs():
    """Define ablation configurations."""
    configs = [
        {'name': 'full', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_ppi', 'use_ppi': False, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_pathway', 'use_ppi': True, 'use_pathway': False, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_coexp', 'use_ppi': True, 'use_pathway': True, 'use_coexp': False, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_expr', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': False, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_cnv', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': False, 'use_mut': True},
        {'name': 'no_mut', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': False},
    ]
    return configs

def sota_models():
    """Define SOTA models to test."""
    return ['GAT', 'GCN', 'GRAPHSAGE']

def main():
    logger.info("Starting ablation and SOTA experiment sweep with fixed architecture...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    integrator = RealDataIntegrator()
    
    for ablation in ablation_configs():
        logger.info(f"Running ablation config: {ablation['name']}")
        
        # Integrate data with ablation toggles
        integrator.integrate_real_data(
            use_ppi=ablation['use_ppi'],
            use_pathway=ablation['use_pathway'],
            use_coexp=ablation['use_coexp'],
            use_expr=ablation['use_expr'],
            use_cnv=ablation['use_cnv'],
            use_mut=ablation['use_mut']
        )
        
        data_path = Path("data/enhanced/real_only_torch_geometric_data.pt")
        if not data_path.exists():
            logger.error(f"Enhanced data not found for ablation {ablation['name']}.")
            continue
        
        data = torch.load(data_path, map_location=device, weights_only=False)
        logger.info(f"Loaded data: {data.x.shape}, {data.edge_index.shape}")
        
        num_nodes = data.x.size(0)
        num_features = data.x.size(1)
        y_np = data.y.cpu().numpy()
        indices = np.arange(num_nodes)
        
        # Use smaller number of folds for faster execution
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Smaller hyperparameter grid
        param_grid = {
            'hidden_dim': [32, 64],  # Small models to fit in memory
            'dropout': [0.3],
            'learning_rate': [0.001],
        }
        
        grid = list(itertools.product(param_grid['hidden_dim'], param_grid['dropout'], param_grid['learning_rate']))
        
        for model_type in sota_models():
            logger.info(f"Running {model_type} for ablation {ablation['name']}")
            
            best_overall_f1 = 0
            best_overall_params = None
            best_overall_results = None
            
            for hidden_dim, dropout, lr in grid:
                logger.info(f"Testing params: hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}")
                
                fold_f1s = []
                fold_results = []
                
                for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, y_np)):
                    logger.info(f"Processing fold {fold+1}/3")
                    
                    # Split train/validation
                    train_idx, val_idx = next(StratifiedKFold(n_splits=3, shuffle=True, random_state=fold).split(train_val_idx, y_np[train_val_idx]))
                    train_indices = train_val_idx[train_idx]
                    val_indices = train_val_idx[val_idx]
                    test_indices = test_idx
                    
                    # Create masks
                    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    train_mask[train_indices] = True
                    val_mask[val_indices] = True
                    test_mask[test_indices] = True
                    
                    # Create model
                    model = get_model(model_type, num_features, hidden_dim, 2, num_layers=2).to(device)
                    
                    # Train model
                    history, best_val_f1 = train_model(model, data, train_mask, val_mask, device, num_epochs=30)
                    
                    # Plot learning curves
                    plot_learning_curves(history, fold, model_type, ablation['name'])
                    
                    # Evaluate on test set
                    test_results = evaluate_model(model, data, test_mask, device)
                    
                    logger.info(f"Fold {fold+1} Results: F1={test_results['f1']:.4f}, ROC_AUC={test_results['roc_auc']:.4f}, PR_AUC={test_results['pr_auc']:.4f}")
                    
                    fold_f1s.append(test_results['f1'])
                    fold_results.append(test_results)
                
                # Calculate average performance
                avg_f1 = np.mean(fold_f1s)
                logger.info(f"Average F1 for {model_type} {ablation['name']}: {avg_f1:.4f}")
                
                if avg_f1 > best_overall_f1:
                    best_overall_f1 = avg_f1
                    best_overall_params = (hidden_dim, dropout, lr)
                    best_overall_results = fold_results
            
            # Save best results
            if best_overall_results:
                logger.info(f"Best {model_type} {ablation['name']}: F1={best_overall_f1:.4f}")
                
                import json
                with open(f'data/processed/best_{model_type.lower()}_{ablation["name"]}_results.json', 'w') as f:
                    json.dump({
                        'best_params': {
                            'hidden_dim': best_overall_params[0],
                            'dropout': best_overall_params[1],
                            'learning_rate': best_overall_params[2]
                        },
                        'average_f1': best_overall_f1,
                        'fold_results': best_overall_results
                    }, f, indent=2)
    
    logger.info("Ablation and SOTA experiment sweep complete!")

if __name__ == "__main__":
    main() 