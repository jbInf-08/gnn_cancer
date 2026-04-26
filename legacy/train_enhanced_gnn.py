import sys
from pathlib import Path as _Path
# Repo root (parent of `legacy/`) for `import gnn_cancer`
_root = _Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.loader import NeighborLoader  # Add this import
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gnn_cancer.models.enhanced_gnn_models import EnhancedGATModel
import itertools
from sklearn.model_selection import StratifiedKFold
from scripts.integrate_real_data import RealDataIntegrator
from gnn_cancer.models.enhanced_gnn_models import get_enhanced_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def train_enhanced_model(model, data, train_mask, val_mask, device, num_epochs=200):
    """Train enhanced model with paper hyperparameters and improved loss function."""
    
    # Calculate class weights for imbalanced dataset (paper approach)
    num_drivers = (data.y == 1).sum().item()
    num_non_drivers = (data.y == 0).sum().item()
    
    # Weight for minority class (drivers) to balance the loss
    pos_weight = torch.tensor([num_non_drivers / num_drivers], device=device)
    
    # Use weighted BCE loss for binary classification with class imbalance (paper approach)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Paper hyperparameters: AdamW optimizer, learning rate 0.001, weight decay 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Paper hyperparameters: ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )
    
    best_val_f1 = 0
    best_val_acc = 0
    patience_counter = 0
    patience = 30  # Paper early stopping patience
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Convert labels to float for BCE loss
        train_labels = data.y[train_mask].float()
        
        # Binary classification loss - use only positive class logits
        loss = criterion(out[train_mask, 1], train_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (paper approach)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = (torch.sigmoid(val_out[val_mask, 1]) > 0.5).float()
            val_true = data.y[val_mask].float()
            val_acc = (val_pred == val_true).float().mean()
            
            # Calculate F1 score for validation
            val_pred_np = val_pred.cpu().numpy()
            val_true_np = val_true.cpu().numpy()
            val_f1 = f1_score(val_true_np, val_pred_np, average='binary', zero_division=0)
            
            # Calculate precision and recall
            val_precision = precision_score(val_true_np, val_pred_np, average='binary', zero_division=0)
            val_recall = recall_score(val_true_np, val_pred_np, average='binary', zero_division=0)
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': loss.item(),
                'val_accuracy': val_acc.item(),
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Learning rate scheduling based on F1 score (paper approach)
            scheduler.step(val_f1)
            
            # Save best model based on F1 score (better for imbalanced data)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/best_enhanced_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping (paper approach)
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    return best_val_acc, best_val_f1

def evaluate_model(model, data, test_mask, device):
    """Comprehensive model evaluation."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = (torch.sigmoid(out[test_mask, 1]) > 0.5).float()
        test_true = data.y[test_mask].float()
        test_probs = torch.sigmoid(out[test_mask, 1])
        
        # Calculate metrics
        accuracy = (test_pred == test_true).float().mean()
        
        # Convert to numpy for sklearn metrics
        test_pred_np = test_pred.cpu().numpy()
        test_true_np = test_true.cpu().numpy()
        test_probs_np = test_probs.cpu().numpy()
        
        # Binary classification metrics
        precision = precision_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        recall = recall_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        f1 = f1_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        
        # ROC AUC for binary classification
        try:
            roc_auc = roc_auc_score(test_true_np, test_probs_np)
        except ValueError:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(test_true_np, test_pred_np)
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': test_pred_np,
            'probabilities': test_probs_np,
            'true_labels': test_true_np
        }

def plot_results(results, model_name, save_path):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Metrics bar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1'], results['roc_auc']]
    
    axes[0, 0].bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[0, 0].set_title(f'{model_name} Performance Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Confusion matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
    axes[1, 0].plot(fpr, tpr, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Probability distribution
    axes[1, 1].hist(results['probabilities'][results['true_labels'] == 0], 
                   alpha=0.7, label='Non-drivers', bins=20, density=True)
    axes[1, 1].hist(results['probabilities'][results['true_labels'] == 1], 
                   alpha=0.7, label='Drivers', bins=20, density=True)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Probability Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curves(history, fold, params):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / F1')
    plt.title(f'Learning Curves (Fold {fold+1}, hidden_dim={params[0]}, dropout={params[1]}, lr={params[2]})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/learning_curve_fold{fold+1}_hd{params[0]}_do{params[1]}_lr{params[2]}.png')
    plt.close()

def ablation_configs():
    # Each config is a dict of ablation toggles
    configs = [
        {'name': 'full', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_ppi', 'use_ppi': False, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_pathway', 'use_ppi': True, 'use_pathway': False, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_coexp', 'use_ppi': True, 'use_pathway': True, 'use_coexp': False, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_expr', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': False, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_cnv', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': False, 'use_mut': True},
        {'name': 'no_mut', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': False},
        {'name': 'no_ppi_no_pathway', 'use_ppi': False, 'use_pathway': False, 'use_coexp': True, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_expr_no_cnv', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': False, 'use_cnv': False, 'use_mut': True},
        {'name': 'no_all_edges', 'use_ppi': False, 'use_pathway': False, 'use_coexp': False, 'use_expr': True, 'use_cnv': True, 'use_mut': True},
        {'name': 'no_all_features', 'use_ppi': True, 'use_pathway': True, 'use_coexp': True, 'use_expr': False, 'use_cnv': False, 'use_mut': False},
        # Add more as needed
    ]
    return configs

def sota_models():
    return ['GAT', 'GRAPHSAGE', 'GCN']

def main():
    logger.info("Starting ablation and SOTA experiment sweep...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        data_path = Path(f"data/enhanced/real_only_torch_geometric_data.pt")
        if not data_path.exists():
            logger.error(f"Enhanced data not found for ablation {ablation['name']}.")
            continue
        data = torch.load(data_path, map_location=device, weights_only=False)
        num_nodes = data.x.size(0)
        num_features = data.x.size(1)
        y_np = data.y.cpu().numpy()
        indices = np.arange(num_nodes)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        param_grid = {
            'hidden_dim': [64],  # Smaller model to fit in memory
            'dropout': [0.3],
            'learning_rate': [0.001],
        }
        grid = list(itertools.product(param_grid['hidden_dim'], param_grid['dropout'], param_grid['learning_rate']))
        for model_type in sota_models():
            logger.info(f"Running SOTA model: {model_type} for ablation {ablation['name']}")
            best_overall_f1 = 0
            best_overall_params = None
            best_overall_results = None
            for hidden_dim, dropout, lr in grid:
                fold_f1s = []
                fold_results = []
                for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, y_np)):
                    train_idx, val_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=fold).split(train_val_idx, y_np[train_val_idx]))
                    train_indices = train_val_idx[train_idx]
                    val_indices = train_val_idx[val_idx]
                    test_indices = test_idx
                    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    train_mask[train_indices] = True
                    val_mask[val_indices] = True
                                         test_mask[test_indices] = True
                     model = get_enhanced_model(
                         model_type,
                         input_dim=num_features,
                         hidden_dim=hidden_dim,
                         output_dim=2,
                         num_layers=3,  # Smaller model to fit in memory
                         dropout=dropout,
                         use_edge_attr=True,
                         num_edge_types=8
                     ).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    best_val_f1 = 0
                    patience_counter = 0
                    patience = 10
                    num_epochs = 100
                    history = {'train_loss': [], 'val_f1': []}
                    
                    for epoch in range(num_epochs):
                        model.train()
                        optimizer.zero_grad()
                        
                        # Forward pass on full graph (smaller model should fit)
                        out = model(data.x, data.edge_index)
                        
                        # Calculate loss for training nodes
                        train_labels = data.y[train_mask].float()
                        loss = criterion(out[train_mask, 1], train_labels)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        model.eval()
                        with torch.no_grad():
                            val_out = model(data.x, data.edge_index)
                            val_pred = (torch.sigmoid(val_out[val_mask, 1]) > 0.5).float()
                            val_true = data.y[val_mask].float()
                            val_f1 = f1_score(val_true.cpu().numpy(), val_pred.cpu().numpy(), average='binary', zero_division=0)
                            scheduler.step(val_f1)
                            history['train_loss'].append(loss.item())
                            history['val_f1'].append(val_f1)
                            if val_f1 > best_val_f1:
                                best_val_f1 = val_f1
                                torch.save(model.state_dict(), f'models/best_{model_type.lower()}_{ablation["name"]}_fold{fold}.pt')
                                patience_counter = 0
                            else:
                                patience_counter += 1
                            if patience_counter >= patience:
                                logger.info(f"Early stopping at epoch {epoch}")
                                break
                    plot_learning_curves(history, fold, (hidden_dim, dropout, lr))
                    try:
                        # Try the specific naming pattern first
                        model_path = f'models/best_{model_type.lower()}_{ablation["name"]}_fold{fold}.pt'
                        if not os.path.exists(model_path):
                            # Fallback to the enhanced model naming
                            model_path = 'models/best_enhanced_model.pt'
                        model.load_state_dict(torch.load(model_path))
                    except FileNotFoundError:
                        logger.warning(f"Best model file not found for {model_type} {ablation['name']} fold {fold}. Skipping evaluation.")
                        continue
                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index)
                        test_pred = (torch.sigmoid(out[test_mask, 1]) > 0.5).float()
                        test_true = data.y[test_mask].float()
                        test_probs = torch.sigmoid(out[test_mask, 1])
                        test_pred_np = test_pred.cpu().numpy()
                        test_true_np = test_true.cpu().numpy()
                        test_probs_np = test_probs.cpu().numpy()
                        precision = precision_score(test_true_np, test_pred_np, average='binary', zero_division=0)
                        recall = recall_score(test_true_np, test_pred_np, average='binary', zero_division=0)
                        f1 = f1_score(test_true_np, test_pred_np, average='binary', zero_division=0)
                        try:
                            roc_auc = roc_auc_score(test_true_np, test_probs_np)
                        except ValueError:
                            roc_auc = 0.0
                        from sklearn.metrics import average_precision_score
                        try:
                            pr_auc = average_precision_score(test_true_np, test_probs_np)
                        except ValueError:
                            pr_auc = 0.0
                        cm = confusion_matrix(test_true_np, test_pred_np)
                        logger.info(f"Fold {fold+1} Test Results: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc:.4f}, PR_AUC={pr_auc:.4f}")
                        np.save(f'results/confusion_matrix_{model_type.lower()}_{ablation["name"]}_fold{fold}_hd{hidden_dim}_do{dropout}_lr{lr}.npy', cm)
                        with open(f'results/class_counts_{model_type.lower()}_{ablation["name"]}_fold{fold}_hd{hidden_dim}_do{dropout}_lr{lr}.txt', 'w') as fcc:
                            fcc.write(f"Positives: {sum(test_true_np==1)}, Negatives: {sum(test_true_np==0)}\n")
                        fold_f1s.append(f1)
                        fold_results.append({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc,
                            'pr_auc': pr_auc,
                            'confusion_matrix': cm.tolist(),
                            'class_counts': {'positives': int(sum(test_true_np==1)), 'negatives': int(sum(test_true_np==0))}
                        })
                avg_f1 = np.mean(fold_f1s)
                logger.info(f"Grid search params: hidden_dim={hidden_dim}, dropout={dropout}, lr={lr} | Avg F1={avg_f1:.4f}")
                if avg_f1 > best_overall_f1:
                    best_overall_f1 = avg_f1
                    best_overall_params = (hidden_dim, dropout, lr)
                    best_overall_results = fold_results
            logger.info(f"Best hyperparameters for {model_type} {ablation['name']}: hidden_dim={best_overall_params[0]}, dropout={best_overall_params[1]}, lr={best_overall_params[2]}")
            logger.info(f"Best average F1: {best_overall_f1:.4f}")
            import json
            with open(f'data/processed/best_{model_type.lower()}_{ablation["name"]}_results.json', 'w') as f:
                json.dump({
                    'best_params': {
                        'hidden_dim': best_overall_params[0],
                        'dropout': best_overall_params[1],
                        'learning_rate': best_overall_params[2]
                    },
                    'fold_results': best_overall_results
                }, f, indent=2)
    logger.info("Ablation and SOTA experiment sweep complete!")

if __name__ == "__main__":
    main()

