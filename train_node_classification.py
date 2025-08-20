import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv, GCNConv
from torch_geometric.nn import HeteroConv, Linear
import logging
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeteroGNN(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, num_heads=4):
        super().__init__()
        
        # Define the heterogeneous convolution layers
        self.conv1 = HeteroConv({
            ('gene', 'associated_with', 'go'): GATConv((in_channels_dict['gene'], in_channels_dict['go']), hidden_channels, heads=num_heads, add_self_loops=False),
            ('go', 'rev_associated_with', 'gene'): GATConv((in_channels_dict['go'], in_channels_dict['gene']), hidden_channels, heads=num_heads, add_self_loops=False),
            ('gene', 'cited_in', 'pubmed'): GATConv((in_channels_dict['gene'], in_channels_dict['pubmed']), hidden_channels, heads=num_heads, add_self_loops=False),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv((in_channels_dict['pubmed'], in_channels_dict['gene']), hidden_channels, heads=num_heads, add_self_loops=False),
        })
        
        self.conv2 = HeteroConv({
            ('gene', 'associated_with', 'go'): GATConv((hidden_channels * num_heads, hidden_channels * num_heads), hidden_channels, heads=num_heads, add_self_loops=False),
            ('go', 'rev_associated_with', 'gene'): GATConv((hidden_channels * num_heads, hidden_channels * num_heads), hidden_channels, heads=num_heads, add_self_loops=False),
            ('gene', 'cited_in', 'pubmed'): GATConv((hidden_channels * num_heads, hidden_channels * num_heads), hidden_channels, heads=num_heads, add_self_loops=False),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv((hidden_channels * num_heads, hidden_channels * num_heads), hidden_channels, heads=num_heads, add_self_loops=False),
        })
        
        # Add projection layers for each node type
        self.proj = torch.nn.ModuleDict({
            'gene': torch.nn.Linear(in_channels_dict['gene'], hidden_channels * num_heads),
            'go': torch.nn.Linear(in_channels_dict['go'], hidden_channels * num_heads),
            'pubmed': torch.nn.Linear(in_channels_dict['pubmed'], hidden_channels * num_heads),
        })
        # Output layer for gene classification
        self.lin = Linear(hidden_channels * num_heads, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        # First layer
        prev_x_dict = x_dict.copy()
        x_dict = self.conv1(x_dict, edge_index_dict)
        # For any missing node type, project previous features to correct size
        for key in prev_x_dict:
            if x_dict.get(key) is None:
                x_dict[key] = self.proj[key](prev_x_dict[key])
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        # Second layer
        prev_x_dict2 = x_dict.copy()
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x for key, x in x_dict.items() if x is not None}
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        # Ensure 'gene' is present for output
        gene_features = x_dict['gene'] if 'gene' in x_dict else prev_x_dict2['gene']
        # Return only gene node predictions
        return self.lin(gene_features)

def train_model(model, data, train_mask, val_mask, device, num_epochs=100):
    # Calculate class weights for imbalanced dataset
    num_drivers = (data['gene'].y == 1).sum().item()
    num_non_drivers = (data['gene'].y == 0).sum().item()
    
    # Weight for minority class (drivers) to balance the loss
    pos_weight = torch.tensor([num_non_drivers / num_drivers], device=device)
    
    # Use weighted BCE loss for binary classification with class imbalance
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_acc = 0
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)
        
        # Convert labels to float for BCE loss
        train_labels = data['gene'].y[train_mask].float()
        
        # Binary classification loss - use only positive class logits
        # out has shape [batch_size, 2], we need [batch_size] for BCE
        loss = criterion(out[train_mask, 1], train_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x_dict, data.edge_index_dict)
            val_pred = (torch.sigmoid(val_out[val_mask, 1]) > 0.5).float()
            val_true = data['gene'].y[val_mask].float()
            val_acc = (val_pred == val_true).float().mean()
            
            # Calculate F1 score for validation
            val_pred_np = val_pred.cpu().numpy()
            val_true_np = val_true.cpu().numpy()
            val_f1 = f1_score(val_true_np, val_pred_np, average='binary', zero_division=0)
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': loss.item(),
                'val_accuracy': val_acc.item(),
                'val_f1': val_f1,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Learning rate scheduling based on F1 score
            scheduler.step(val_f1)
            
            # Save best model based on F1 score (better for imbalanced data)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/best_hetero_model.pt')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    return best_val_acc

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load the heterogeneous graph data
    try:
        # Try to load the expanded cancer drivers dataset first
        data = torch.load('data/processed/cancer_drivers/heterogeneous_graph.pt', weights_only=False)
        logger.info("Loaded expanded cancer drivers dataset")
    except FileNotFoundError:
        # Fallback to BRCA1 dataset
        data = torch.load('data/processed/brca1/heterogeneous_graph.pt', weights_only=False)
        logger.info("Loaded BRCA1 dataset (fallback)")
    
    data = data.to(device)
    
    # Print feature shapes for debugging
    print('gene.x shape:', data['gene'].x.shape)
    print('go.x shape:', data['go'].x.shape)
    print('pubmed.x shape:', data['pubmed'].x.shape)

    # Print expected in_channels for each GATConv in conv1
    print('conv1 expected in_channels:')
    print('  gene->go:', data['gene'].x.size(1))
    print('  go->gene:', data['go'].x.size(1))
    print('  gene->pubmed:', data['gene'].x.size(1))
    print('  pubmed->gene:', data['pubmed'].x.size(1))
    # Print expected in_channels for each GATConv in conv2
    hidden_channels = 128
    num_heads = 4
    print('conv2 expected in_channels (should be hidden_channels * num_heads =', hidden_channels * num_heads, ')')
    print('  gene->go:', hidden_channels * num_heads)
    print('  go->gene:', hidden_channels * num_heads)
    print('  gene->pubmed:', hidden_channels * num_heads)
    print('  pubmed->gene:', hidden_channels * num_heads)

    # Print label distribution
    print('Label distribution:')
    print('  Drivers (1):', (data['gene'].y == 1).sum().item())
    print('  Non-drivers (0):', (data['gene'].y == 0).sum().item())
    
    # Create stratified train/val/test masks for gene nodes
    gene_labels = data['gene'].y.cpu().numpy()
    gene_indices = np.arange(len(gene_labels))
    
    # Use stratified sampling to ensure driver genes appear in all splits
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_val_indices, test_indices = next(sss.split(gene_indices, gene_labels))
    
    # Split train_val into train and validation
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.7, random_state=42)  # 15% of total for validation
    train_indices, val_indices = next(sss_val.split(gene_indices[train_val_indices], gene_labels[train_val_indices]))
    train_indices = train_val_indices[train_indices]
    val_indices = train_val_indices[val_indices]
    
    # Create masks
    train_mask = torch.zeros(len(gene_labels), dtype=torch.bool)
    val_mask = torch.zeros(len(gene_labels), dtype=torch.bool)
    test_mask = torch.zeros(len(gene_labels), dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # Print split information
    print(f'\nData split information:')
    print(f'  Training set: {train_mask.sum().item()} genes ({train_mask.sum().item()/len(gene_labels)*100:.1f}%)')
    print(f'    - Drivers: {(data["gene"].y[train_mask] == 1).sum().item()}')
    print(f'    - Non-drivers: {(data["gene"].y[train_mask] == 0).sum().item()}')
    print(f'  Validation set: {val_mask.sum().item()} genes ({val_mask.sum().item()/len(gene_labels)*100:.1f}%)')
    print(f'    - Drivers: {(data["gene"].y[val_mask] == 1).sum().item()}')
    print(f'    - Non-drivers: {(data["gene"].y[val_mask] == 0).sum().item()}')
    print(f'  Test set: {test_mask.sum().item()} genes ({test_mask.sum().item()/len(gene_labels)*100:.1f}%)')
    print(f'    - Drivers: {(data["gene"].y[test_mask] == 1).sum().item()}')
    print(f'    - Non-drivers: {(data["gene"].y[test_mask] == 0).sum().item()}')
    
    # Initialize model
    in_channels_dict = {
        'gene': data['gene'].x.size(1),
        'go': data['go'].x.size(1),
        'pubmed': data['pubmed'].x.size(1)
    }
    
    # Binary classification: 2 classes (0: non-driver, 1: driver)
    model = HeteroGNN(in_channels_dict, hidden_channels=128, out_channels=2, num_heads=4).to(device)
    
    # Initialize wandb
    wandb.init(project="breast-cancer-gnn", name="cancer-drivers-binary-classification")
    
    logger.info("Training heterogeneous GNN on cancer driver data...")
    
    # Train the model
    best_val_acc = train_model(model, data, train_mask, val_mask, device)
    
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('models/best_hetero_model.pt'))
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        test_pred = (torch.sigmoid(out[test_mask, 1]) > 0.5).float()
        test_true = data['gene'].y[test_mask].float()
        
        # Calculate binary classification metrics
        accuracy = (test_pred == test_true).float().mean()
        
        # Convert to numpy for sklearn metrics
        test_pred_np = test_pred.cpu().numpy()
        test_true_np = test_true.cpu().numpy()
        test_probs_np = torch.sigmoid(out[test_mask, 1]).cpu().numpy()
        
        # Binary classification metrics
        precision = precision_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        recall = recall_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        f1 = f1_score(test_true_np, test_pred_np, average='binary', zero_division=0)
        
        # ROC AUC for binary classification
        try:
            roc_auc = roc_auc_score(test_true_np, test_probs_np)
        except ValueError:
            roc_auc = 0.0  # Handle case with only one class
        
        print(f'\nTest Results:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')
        
        # Log final metrics
        wandb.log({
            'test_accuracy': accuracy.item(),
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc
        })
    
    wandb.finish()

if __name__ == "__main__":
    main() 