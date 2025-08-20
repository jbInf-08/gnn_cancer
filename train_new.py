import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import argparse
from typing import Dict, List, Optional, Union

from models.models import GCNModel, GraphSAGEModel, GATModel
from utils.data_utils import load_data
from utils.visualization import plot_learning_curves, plot_roc_curves, plot_pr_curves, plot_confusion_matrix
from utils.cancer_types import get_cancer_type, get_all_cancer_types, DataSource
import pretrain

def parse_args():
    parser = argparse.ArgumentParser(description="GNN Cancer Training for Additional Cancer Types")
    parser.add_argument('--cancer_type', type=str, required=True, help='Cancer type code (e.g., LUAD, LUSC, COAD, READ)')
    parser.add_argument('--model', type=str, required=True, choices=['GCN', 'GraphSAGE', 'GAT'], help='Model type')
    parser.add_argument('--pretrain', action='store_true', help='Enable self-supervised pre-training')
    parser.add_argument('--augment', action='store_true', help='Enable advanced data augmentation during training')
    parser.add_argument('--data_source', type=str, default='TCGA', choices=[s.value for s in DataSource], help='Data source to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    return parser.parse_args()

def get_model(model_type: str, in_channels: int, hidden_channels: int, num_layers: int) -> torch.nn.Module:
    """Get the specified model type."""
    if model_type == 'GCN':
        return GCNModel(in_channels, hidden_channels, num_layers)
    elif model_type == 'GraphSAGE':
        return GraphSAGEModel(in_channels, hidden_channels, num_layers)
    elif model_type == 'GAT':
        return GATModel(in_channels, hidden_channels, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class ModelTrainer:
    def __init__(self, data_dir: Path, device: torch.device):
        self.data_dir = data_dir
        self.device = device
        self.data = None
        self.args = None
        self.config = {
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10
        }

    def load_data(self, cancer_type: str, data_source: str = 'TCGA'):
        """Load data for the specified cancer type and data source."""
        cancer_info = get_cancer_type(cancer_type)
        if not cancer_info:
            raise ValueError(f"Unknown cancer type: {cancer_type}")
        
        if data_source not in [s.value for s in cancer_info.data_sources]:
            raise ValueError(f"Data source {data_source} not available for cancer type {cancer_type}")
        
        self.data = load_data(self.data_dir, cancer_type, data_source)
        self.data = self.data.to(self.device)

    def train_model(self, model, optimizer, scheduler, train_loader, val_loader):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []

        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            avg_train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0
            predictions = []
            labels = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch.x, batch.edge_index)
                    val_loss += F.cross_entropy(out, batch.y).item() * batch.num_graphs
                    pred = out.argmax(dim=1)
                    predictions.extend(pred.cpu().numpy())
                    labels.extend(batch.y.cpu().numpy())
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(avg_val_loss)

            # Calculate metrics
            train_metric = self.calculate_metrics(predictions, labels)
            val_metric = self.calculate_metrics(predictions, labels)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"results/{self.args.cancer_type}_{self.args.model}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Log metrics
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_metric['accuracy'],
                'val_accuracy': val_metric['accuracy'],
                'train_f1': train_metric['f1'],
                'val_f1': val_metric['f1']
            })

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

    def calculate_metrics(self, predictions, labels):
        return {
            'accuracy': np.mean(np.array(predictions) == np.array(labels)),
            'f1': f1_score(labels, predictions, average='weighted')
        }

    def train(self):
        # Initialize model
        if self.args.pretrain:
            print(f"Pretraining {self.args.model} with self-supervised learning...")
            model, pretrain_losses = pretrain.pretrain_model(
                self.data,
                model_type=self.args.model,
                hidden_channels=self.config['hidden_channels'],
                num_layers=self.config['num_layers'],
                device=self.device
            )
            torch.save(model.state_dict(), f"results/{self.args.cancer_type}_{self.args.model}_pretrained.pt")
        else:
            model = get_model(
                self.args.model,
                in_channels=self.data.num_node_features,
                hidden_channels=self.config['hidden_channels'],
                num_layers=self.config['num_layers']
            ).to(self.device)

        # Setup optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Create data loaders
        train_loader = DataLoader(self.data, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(self.data, batch_size=self.config['batch_size'])

        # Train model
        results = self.train_model(model, optimizer, scheduler, train_loader, val_loader)

        # Generate visualizations
        self.visualize_results(model, results)

        return results

    def visualize_results(self, model, results):
        # Plot learning curves
        plt.figure(figsize=(10, 5))
        plt.plot(results['train_losses'], label='Train Loss')
        plt.plot(results['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Learning Curves for {self.args.cancer_type} - {self.args.model}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/{self.args.cancer_type}_{self.args.model}_learning_curves.png')
        plt.close()

        # Feature space visualization
        model.eval()
        with torch.no_grad():
            embeddings = model(self.data.x, self.data.edge_index).cpu().numpy()
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=self.data.y.cpu().numpy(), cmap='viridis', alpha=0.7)
        plt.colorbar()
        plt.title(f'Feature Space for {self.args.cancer_type} - {self.args.model}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.tight_layout()
        plt.savefig(f'results/{self.args.cancer_type}_{self.args.model}_feature_space.png')
        plt.close()

        # Log visualizations to wandb
        wandb.log({
            'learning_curves': wandb.Image(f'results/{self.args.cancer_type}_{self.args.model}_learning_curves.png'),
            'feature_space': wandb.Image(f'results/{self.args.cancer_type}_{self.args.model}_feature_space.png')
        })

def main():
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project="gnn-cancer",
        name=f"{args.cancer_type}_{args.model}",
        config=vars(args)
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize trainer
    trainer = ModelTrainer(data_dir=Path("data"), device=device)
    trainer.args = args
    trainer.config.update({
        'hidden_channels': args.hidden_channels,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience
    })
    
    # Load data
    trainer.load_data(args.cancer_type, args.data_source)
    
    # Train model
    results = trainer.train()
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 