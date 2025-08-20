import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='GCN'):
        super(GNNModel, self).__init__()
        self.model_type = model_type
        
        # First layer
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
            self.conv2 = GATConv(hidden_channels * 4, hidden_channels)
        else:  # SAGE
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Additional layers for better feature extraction
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Dropout layers
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    return metrics

def train_model(model, train_loader, val_loader, device, num_epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Calculate class weights
    all_labels = []
    for data in train_loader:
        all_labels.extend(data.y.numpy())
    class_counts = np.bincount(all_labels)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    
    best_val_acc = 0
    best_model_path = 'best_model.pt'
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1)[:, 1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(prob.cpu().detach().numpy())
        
        train_metrics = calculate_metrics(all_labels, all_preds, all_probs)
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(out, data.y, weight=class_weights)
                val_loss += loss.item() * data.num_graphs
                
                pred = out.argmax(dim=1)
                prob = F.softmax(out, dim=1)[:, 1]
                all_val_preds.extend(pred.cpu().numpy())
                all_val_labels.extend(data.y.cpu().numpy())
                all_val_probs.extend(prob.cpu().numpy())
        
        val_metrics = calculate_metrics(all_val_labels, all_val_preds, all_val_probs)
        
        # Update learning rate
        scheduler.step(val_metrics['accuracy'])
        
        # Log metrics
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch + 1}/{num_epochs}:')
            logger.info(f'Train Loss: {total_loss/len(train_loader.dataset):.4f}, Train Acc: {train_metrics["accuracy"]:.4f}')
            logger.info(f'Val Loss: {val_loss/len(val_loader.dataset):.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), best_model_path)
        
        # Log to wandb
        wandb.log({
            'train_loss': total_loss/len(train_loader.dataset),
            'val_loss': val_loss/len(val_loader.dataset),
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_roc_auc': train_metrics['roc_auc'],
            'val_roc_auc': val_metrics['roc_auc'],
            'epoch': epoch + 1
        })
    
    return best_val_acc

def main():
    # Initialize wandb
    wandb.init(project="breast-cancer-gnn", config={
        "architecture": "GCN",
        "dataset": "Breast Cancer Wisconsin",
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "hidden_channels": 256
    })
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load processed data
    train_data = torch.load('data/processed/train_data.pt')
    test_data = torch.load('data/processed/test_data.pt')
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize model
    in_channels = train_data[0].x.size(1)
    model = GNNModel(
        in_channels=in_channels,
        hidden_channels=256,
        out_channels=2,
        model_type='GCN'
    ).to(device)
    
    # Train model
    best_val_acc = train_model(model, train_loader, test_loader, device)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1)[:, 1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    # Calculate final metrics
    final_metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    # Log final metrics
    logger.info('Final Metrics:')
    for metric_name, value in final_metrics.items():
        logger.info(f'{metric_name}: {value:.4f}')
        wandb.log({f'final_{metric_name}': value})
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})
    
    wandb.finish()

if __name__ == '__main__':
    main() 