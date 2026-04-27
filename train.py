# train.py
# Repo-root execution: add project root to path so `import gnn_cancer` works without pip install.
import sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parent
if str(_sys_root) not in sys.path:
    sys.path.insert(0, str(_sys_root))
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import train_test_split_edges, dropout_adj, dropout_node
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score, 
    roc_auc_score, 
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import json
from transformers import get_cosine_schedule_with_warmup
import wandb
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from imblearn.over_sampling import SMOTE
from collections import Counter
import argparse
from typing import Optional, Dict, Any, List, Union, Tuple
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import umap
import shap
import lime
import lime.lime_tabular

from gnn_cancer.models.models import GCNModel, GraphSAGEModel, GATModel
from gnn_cancer.models.gnn_models import get_model
import pretrain
from gnn_cancer.utils.data_utils import load_data
from gnn_cancer.utils.visualization import plot_learning_curves, plot_roc_curves, plot_pr_curves, plot_confusion_matrix
from gnn_cancer.utils.cancer_types import get_cancer_type, get_all_cancer_types, DataSource
from gnn_cancer.utils.train_model import train_model, calculate_metrics

# Set up logging for script progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_script")

def _get_wandb_api_key() -> Optional[str]:
    """Read WandB key from the environment or config/api_keys.json (optional)."""
    k = os.getenv("WANDB_API_KEY")
    if k:
        return k
    try:
        with open("config/api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
        return keys.get("WANDB_API_KEY")
    except Exception:
        return None


def _make_wandb_stub():
    class _Image:
        def __init__(self, path: str) -> None:
            self.path = path

    class _W:
        Image = _Image

        @staticmethod
        def init(*_a, **_k):
            return None

        @staticmethod
        def log(*_a, **_k):
            return None

        @staticmethod
        def finish():
            return None

    return _W()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ImprovedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, conv_type='SAGEConv', dropout=0.3):
        super().__init__()
        self.input_norm = torch.nn.BatchNorm1d(in_channels)
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.attention = nn.ModuleList()
        
        conv_layer = {'SAGEConv': SAGEConv, 'GATConv': GATConv, 'GCNConv': GCNConv}[conv_type]
        
        # First layer with larger hidden dimension
        self.convs.append(conv_layer(in_channels, hidden_channels))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.residual_projs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.attention.append(MultiHeadAttention(hidden_channels, num_heads=8, dropout=dropout))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
            self.residual_projs.append(torch.nn.Identity())
            self.attention.append(MultiHeadAttention(hidden_channels, num_heads=8, dropout=dropout))
        
        # Last layer
        self.convs.append(conv_layer(hidden_channels, out_channels))
        self.residual_projs.append(torch.nn.Linear(hidden_channels, out_channels))
        
        # Enhanced normalization and scaling
        self.layer_norm = torch.nn.LayerNorm(out_channels)
        self.output_scale = torch.nn.Parameter(torch.ones(1))
        self.output_bias = torch.nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if module.weight is not None:
                torch.nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, edge_index):
        x = self.input_norm(x)
        
        for i in range(self.num_layers):
            identity = self.residual_projs[i](x)
            x = self.convs[i](x, edge_index)
            
            if i < self.num_layers - 1:
                x = self.attention[i](x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
                x = x + identity  # Residual connection
        
        x = self.layer_norm(x)
        x = x * self.output_scale + self.output_bias
        return x

class EnsembleGNN(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x, edge_index):
        outputs = [model(x, edge_index) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class ModelTrainer:
    def __init__(self, data_dir: Path, device: torch.device):
        self.data_dir = data_dir
        self.device = device
        self.model_types = ['GCN', 'GraphSAGE', 'GAT']
        self.config = {
            'hidden_channels': 64,  # Paper: 64 hidden units per layer
            'num_layers': 3,        # Paper: 3 layers for all models
            'dropout': 0.5,         # Paper: 0.5 dropout rate
            'learning_rate': 0.001, # Paper: 0.001 learning rate
            'weight_decay': 5e-4,   # Paper: 5e-4 weight decay
            'epochs': 100,          # Paper: trained for sufficient epochs
            'batch_size': 32,       # Paper: batch processing
            'patience': 10,         # Paper: early stopping patience
            'gat_heads': 8,         # Paper: 8 attention heads for GAT
            'sage_neighbors': 25    # Paper: 25 neighbors for GraphSAGE
        }
        self.data = None
        self.args = None

    def load_data(self, cancer_type: str, data_source: str = 'TCGA'):
        """Load data for the specified cancer type and data source."""
        if cancer_type.upper() == 'BRCA' and data_source.upper() == 'BENCHMARK':
            from gnn_cancer.benchmark_datasets import load_uci_breast_cancer_graph

            self.data = load_uci_breast_cancer_graph(
                k_neighbors=10,
                seed=int(self.args.seed) if self.args and hasattr(self.args, "seed") else 42,
            )
            n = int(self.data.train_mask.sum()) + int(self.data.val_mask.sum()) + int(
                self.data.test_mask.sum()
            )
            print(
                f"[INFO] Loaded BENCHMARK (sklearn Wisconsin BC) graph: "
                f"{self.data.num_nodes} nodes, {self.data.num_edges} edges, "
                f"{self.data.num_node_features} features, train/val/test = "
                f"{int(self.data.train_mask.sum())}/{int(self.data.val_mask.sum())}/{int(self.data.test_mask.sum())}"
            )
            return
        # For BRCA/TCGA, load the comprehensive multi-omics graph
        if cancer_type.upper() == 'BRCA' and data_source.upper() == 'TCGA':
            data_path = self.data_dir / 'processed' / 'BRCA_comprehensive_data.pt'
            if not data_path.exists():
                raise FileNotFoundError(f"Expected data at {data_path}, but not found.")
            self.data = torch.load(data_path, weights_only=False)
            print(f"[INFO] Loaded BRCA comprehensive graph with {self.data.num_node_features} node features and {getattr(self.data, 'edge_attr', torch.empty(0)).size(0) if hasattr(self.data, 'edge_attr') else 'N/A'} edge attributes.")
            # Warn if model does not use edge_attr
            if not hasattr(self, 'model_uses_edge_attr') and hasattr(self.data, 'edge_attr'):
                print("[WARNING] edge_attr is present in the data. Ensure your GNN model uses edge_attr if appropriate (e.g., for edge-type aware GNNs).")
            # Always (re-)create train/val/test masks
            print("[INFO] Creating train/val/test masks for BRCA_comprehensive_data.pt graph...")
            num_nodes = self.data.num_nodes
            y = self.data.y.cpu().numpy() if hasattr(self.data.y, 'cpu') else np.array(self.data.y)
            print(f"[DEBUG] y shape: {y.shape}, num_nodes: {num_nodes}")
            idx = np.arange(num_nodes)
            idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y, test_size=0.3, stratify=y, random_state=42)
            idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[idx_train] = True
            val_mask[idx_val] = True
            test_mask[idx_test] = True
            self.data.train_mask = train_mask
            self.data.val_mask = val_mask
            self.data.test_mask = test_mask
            print(f"[INFO] Added train/val/test split: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
            print(f"[DEBUG] train_mask shape: {train_mask.shape}, val_mask shape: {val_mask.shape}, test_mask shape: {test_mask.shape}")
        else:
            # Fallback to original loader for other cancer types or sources
            cancer_info = get_cancer_type(cancer_type)
            if not cancer_info:
                raise ValueError(f"Unknown cancer type: {cancer_type}")
            if data_source not in [s.value for s in cancer_info.data_sources]:
                raise ValueError(f"Data source {data_source} not available for cancer type {cancer_type}")
            self.data = load_data(self.data_dir, cancer_type, data_source)

    def train_model(self, model, optimizer, scheduler, data, model_type):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []

        for epoch in range(self.config['epochs']):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass with debugging
            out = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
            
            # Debug: Print intermediate values
            _q = getattr(self, "quiet", False)
            if not _q and epoch == 0:
                print(f"[DEBUG] Input x shape: {data.x.shape}")
                print(f"[DEBUG] Input x stats: min={data.x.min():.4f}, max={data.x.max():.4f}, mean={data.x.mean():.4f}")
                print(f"[DEBUG] Input x has NaN: {torch.isnan(data.x).any()}")
                print(f"[DEBUG] Edge index shape: {data.edge_index.shape}")
                if hasattr(data, 'edge_attr'):
                    print(f"[DEBUG] Edge attr shape: {data.edge_attr.shape}")
                    print(f"[DEBUG] Edge attr stats: min={data.edge_attr.min():.4f}, max={data.edge_attr.max():.4f}, mean={data.edge_attr.mean():.4f}")
                    print(f"[DEBUG] Edge attr has NaN: {torch.isnan(data.edge_attr).any()}")
                print(f"[DEBUG] Output shape: {out.shape}")
                print(f"[DEBUG] Output stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
                print(f"[DEBUG] Output has NaN: {torch.isnan(out).any()}")
            
            # Print first 5 logits and check for NaNs/Infs
            if not _q or epoch == 0:
                first_5_logits = out[data.train_mask][:5].detach().cpu().numpy()
                has_nan = torch.isnan(out).any().item()
                has_inf = torch.isinf(out).any().item()
                print(f"Epoch {epoch}: first 5 logits (train): {first_5_logits}")
                print(f"  Any NaN in logits: {has_nan} | Any Inf in logits: {has_inf}")
            
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
                val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])
            
            if not _q or epoch % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'results/{model_type}_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if not getattr(self, "wandb_off", False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'val_loss': val_loss.item(),
                    'val_accuracy': self.calculate_metrics(val_out[data.val_mask].argmax(dim=1).cpu().numpy(), data.y[data.val_mask].cpu().numpy())['accuracy'],
                    'val_f1': self.calculate_metrics(val_out[data.val_mask].argmax(dim=1).cpu().numpy(), data.y[data.val_mask].cpu().numpy())['f1'],
                    'val_precision': self.calculate_metrics(val_out[data.val_mask].argmax(dim=1).cpu().numpy(), data.y[data.val_mask].cpu().numpy())['precision'],
                    'val_recall': self.calculate_metrics(val_out[data.val_mask].argmax(dim=1).cpu().numpy(), data.y[data.val_mask].cpu().numpy())['recall']
                })

        # Final test evaluation
        print(f"[INFO] Evaluating {model_type} on test set...")
        model.load_state_dict(torch.load(f"results/{model_type}_best.pt"))
        model.eval()
        
        with torch.no_grad():
            test_out = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
            test_pred = test_out[data.test_mask].argmax(dim=1)
            test_metrics = self.calculate_metrics(test_pred, data.y[data.test_mask], test_out[data.test_mask])
            
            print(f"[INFO] {model_type} Test Results:")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")
            print(f"  F1-Score: {test_metrics['f1']:.4f}")
            if test_metrics['auc_roc'] is not None:
                print(f"  ROC AUC: {test_metrics['auc_roc']:.4f}")
            print(f"  Test Loss: {test_metrics['loss']:.4f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

    def calculate_metrics(self, predictions, labels, logits=None):
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        
        # Convert to numpy arrays if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Calculate basic metrics
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Calculate ROC AUC if we have probabilities
        auc_roc = None
        if logits is not None:
            try:
                if isinstance(logits, torch.Tensor):
                    logits = logits.cpu().numpy()
                
                # For multi-class, we need to handle it properly
                if logits.shape[1] > 2:
                    # Multi-class case
                    auc_roc = roc_auc_score(labels, logits, multi_class='ovr', average='weighted')
                else:
                    # Binary case
                    if logits.shape[1] == 2:
                        auc_roc = roc_auc_score(labels, logits[:, 1])
                    else:
                        auc_roc = roc_auc_score(labels, logits)
            except Exception as e:
                print(f"[WARNING] Could not calculate ROC AUC: {e}")
                auc_roc = None
        
        # Calculate loss if logits are provided
        loss = None
        if logits is not None:
            try:
                if isinstance(logits, torch.Tensor):
                    logits_tensor = logits
                else:
                    logits_tensor = torch.tensor(logits, dtype=torch.float32)
                
                if isinstance(labels, torch.Tensor):
                    labels_tensor = labels
                else:
                    labels_tensor = torch.tensor(labels, dtype=torch.long)
                
                loss = F.cross_entropy(logits_tensor, labels_tensor).item()
            except Exception as e:
                print(f"[WARNING] Loss calculation failed: {e}")
                loss = float('nan')
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'loss': loss,
            'auc_roc': auc_roc,
            'confusion_matrix': cm
        }

    def train_all_models(self):
        results = {}
        for model_type in self.model_types:
            print(f"[INFO] Training model: {model_type}")
            if self.args.pretrain:
                print(f"[INFO] Pretraining {model_type} with self-supervised learning...")
                model, pretrain_losses = pretrain.pretrain_model(
                    self.data,
                    model_type=model_type,
                    hidden_channels=self.config['hidden_channels'],
                    num_layers=self.config['num_layers'],
                    device=self.device
                )
                torch.save(model.state_dict(), f"results/{model_type}_pretrained.pt")
            else:
                # Automatically set number of output classes
                num_classes = int(torch.max(self.data.y).item() + 1) if hasattr(self.data, 'y') else 2
                model = get_model(
                    model_type,
                    in_channels=self.data.num_node_features,
                    hidden_channels=self.config['hidden_channels'],
                    num_layers=self.config['num_layers'],
                    out_channels=num_classes
                ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            print(f"[INFO] Starting training for {model_type}...")
            result = self.train_model(model, optimizer, scheduler, self.data, model_type)
            print(f"[INFO] Finished training model: {model_type}")
            results[model_type] = result

        print("[INFO] All models trained.")
        return results, None

    def visualize_feature_space(self, model, data):
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index).cpu().numpy()
        
        # Handle NaN values in embeddings
        if np.isnan(embeddings).any():
            print("[WARNING] Found NaN values in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0)
        
        n_samples = embeddings.shape[0]
        print(f"[INFO] Visualizing {n_samples} samples")
        
        # For very small datasets, use PCA instead of t-SNE
        if n_samples < 10:
            print("[INFO] Using PCA for small dataset instead of t-SNE")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            method_name = "PCA"
        else:
            # Adjust perplexity based on number of samples
            perplexity = min(30, max(5, n_samples // 4))  # Between 5 and 30, or n_samples/4
            print(f"[INFO] Using t-SNE with {n_samples} samples and perplexity {perplexity}")
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)
            method_name = "t-SNE"
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data.y.cpu().numpy(), cmap='viridis', alpha=0.7)
        plt.colorbar()
        plt.title(f'{method_name} Visualization of Feature Space')
        plt.xlabel(f'{method_name} 1')
        plt.ylabel(f'{method_name} 2')
        plt.tight_layout()
        plt.savefig('results/feature_visualization.png')
        plt.close()

    def visualize_attention_maps(self, model, data):
        if hasattr(model, 'get_attention_weights'):
            model.eval()
            with torch.no_grad():
                attention_weights = model.get_attention_weights(data.x, data.edge_index)
            plt.figure(figsize=(10, 8))
            plt.imshow(attention_weights.cpu().numpy(), cmap='viridis')
            plt.colorbar()
            plt.title('Attention Map')
            plt.xlabel('Node Index')
            plt.ylabel('Node Index')
            plt.tight_layout()
            plt.savefig('results/attention_map.png')
            plt.close()

    def feature_importance_analysis(self, model, data):
        """Analyze feature importance using GNNExplainer, which is designed for GNNs."""
        try:
            from torch_geometric.explain import GNNExplainer
            import numpy as np
            import matplotlib.pyplot as plt
            
            model.eval()
            
            # Initialize GNNExplainer (fixed parameter passing)
            explainer = GNNExplainer(
                model, 
                lr=0.01,
                num_hops=3
            )
            
            # Get explanation for a sample of nodes
            num_nodes_to_explain = min(20, data.num_nodes)  # Reduced for efficiency
            node_indices = torch.randperm(data.num_nodes)[:num_nodes_to_explain]
            
            # Get explanations
            explanations = []
            for node_idx in node_indices:
                explanation = explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    target=node_idx,
                    index=node_idx
                )
                explanations.append(explanation)
            
            # Aggregate feature importance across nodes
            if explanations and hasattr(explanations[0], 'node_mask'):
                # Sum node masks across all explained nodes
                total_importance = torch.zeros(data.num_node_features)
                for exp in explanations:
                    if exp.node_mask is not None:
                        total_importance += exp.node_mask.sum(dim=0)
                
                # Average importance
                avg_importance = total_importance / len(explanations)
                
                # Plot feature importance
                plt.figure(figsize=(12, 6))
                importance_np = avg_importance.cpu().numpy()
                plt.bar(range(len(importance_np)), importance_np)
                plt.title("Feature Importance (GNNExplainer)")
                plt.xlabel("Feature Index")
                plt.ylabel("Average Node Mask Importance")
                plt.tight_layout()
                plt.savefig('results/feature_importance.png')
                plt.close()
                print("[INFO] GNNExplainer feature importance plot saved.")
                
                # Also save edge importance if available
                if hasattr(explanations[0], 'edge_mask'):
                    edge_importance = torch.zeros(data.num_edges)
                    for exp in explanations:
                        if exp.edge_mask is not None:
                            edge_importance += exp.edge_mask
                    edge_importance = edge_importance / len(explanations)
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(edge_importance.cpu().numpy(), bins=50, alpha=0.7)
                    plt.title("Edge Importance Distribution (GNNExplainer)")
                    plt.xlabel("Edge Importance")
                    plt.ylabel("Frequency")
                    plt.tight_layout()
                    plt.savefig('results/edge_importance.png')
                    plt.close()
                    print("[INFO] GNNExplainer edge importance plot saved.")
                    
            else:
                raise ValueError("No valid explanations generated")
                
        except ImportError:
            print("[WARNING] GNNExplainer not available. Falling back to weight-based importance.")
            self._fallback_feature_importance(model, data)
        except Exception as e:
            print(f"[WARNING] GNNExplainer failed: {e}. Falling back to weight-based importance.")
            self._fallback_feature_importance(model, data)
    
    def _fallback_feature_importance(self, model, data):
        """Fallback method using model weights for feature importance."""
        import numpy as np
        import matplotlib.pyplot as plt
        
        try:
            # Try different model architectures
            if hasattr(model, 'convs') and len(model.convs) > 0:
                # GCN/GraphSAGE style - check if it's a linear layer
                conv_layer = model.convs[0]
                if hasattr(conv_layer, 'weight'):
                    weights = conv_layer.weight.data.cpu().numpy()
                    importance = np.abs(weights).mean(axis=0)
                else:
                    # For GCNConv, use the linear layer inside
                    if hasattr(conv_layer, 'lin'):
                        weights = conv_layer.lin.weight.data.cpu().numpy()
                        importance = np.abs(weights).mean(axis=0)
                    else:
                        raise AttributeError('GCNConv layer has no accessible weights')
            elif hasattr(model, 'conv1'):
                # Alternative naming
                conv_layer = model.conv1
                if hasattr(conv_layer, 'weight'):
                    weights = conv_layer.weight.data.cpu().numpy()
                    importance = np.abs(weights).mean(axis=0)
                elif hasattr(conv_layer, 'lin'):
                    weights = conv_layer.lin.weight.data.cpu().numpy()
                    importance = np.abs(weights).mean(axis=0)
                else:
                    raise AttributeError('Conv layer has no accessible weights')
            elif hasattr(model, 'classifier'):
                # Final classifier layer
                weights = model.classifier.weight.data.cpu().numpy()
                importance = np.abs(weights).mean(axis=0)
            else:
                # Try to find any linear layer
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        weights = module.weight.data.cpu().numpy()
                        importance = np.abs(weights).mean(axis=0)
                        break
                else:
                    raise AttributeError('No suitable layer found for feature importance')
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(importance)), importance)
            plt.title("Feature Importance (Weight-based)")
            plt.xlabel("Feature Index")
            plt.ylabel("Mean Absolute Weight")
            plt.tight_layout()
            plt.savefig('results/feature_importance.png')
            plt.close()
            print("[INFO] Weight-based feature importance plot saved.")
            
        except Exception as e:
            print(f"[ERROR] Could not compute feature importance: {e}")
            # Create a simple placeholder plot
            plt.figure(figsize=(12, 6))
            plt.bar(range(data.num_node_features), [1.0] * data.num_node_features)
            plt.title("Feature Importance (Placeholder)")
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig('results/feature_importance.png')
            plt.close()
            print("[INFO] Placeholder feature importance plot saved.")

    def plot_learning_curves(self, results):
        for model_type, result in results.items():
            plt.figure(figsize=(10, 5))
            plt.plot(result['train_losses'], label='Train Loss')
            plt.plot(result['val_losses'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Learning Curves for {model_type}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/{model_type}_learning_curves.png')
            plt.close()

def main():
    global wandb
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Starting main() with args:", args)
    
    if args.no_wandb:
        wandb = _make_wandb_stub()
        print("[INFO] WandB disabled (--no-wandb).")
    else:
        key = _get_wandb_api_key()
        if not key:
            raise ValueError(
                "Weights & Biases is enabled but no API key was found. Set WANDB_API_KEY, "
                "add it to config/api_keys.json, or pass --no-wandb for offline / local runs."
            )
        os.environ["WANDB_API_KEY"] = key
        print("[INFO] Initializing wandb...")
        wandb.init(
            project="gnn-cancer-mutation",
            name=f"{args.cancer_type}_{args.model}",
            config={
                "cancer_type": args.cancer_type,
                "model": args.model,
                "hidden_channels": args.hidden_channels,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "pretrain": args.pretrain,
                "augment": args.augment,
                "data_source": args.data_source
            }
        )
        print("[INFO] Wandb initialized.")
    
    trainer = ModelTrainer(data_dir=Path(args.data_dir), device=device)
    trainer.args = args
    trainer.quiet = bool(args.quiet)
    trainer.wandb_off = bool(args.no_wandb)
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
    print("[INFO] Trainer initialized.")
    
    # Load data for the specified cancer type
    print("[INFO] Loading data...")
    trainer.load_data(args.cancer_type, args.data_source)
    print("[INFO] Data loaded.")

    if args.ablation:
        print("[INFO] Running ablation study...")
        ablation_variants = [
            ('full', None),
            ('no_ppi', 'remove_ppi_edges'),
            ('no_pathway', 'remove_pathway_information'),
            ('no_expression', 'remove_expression_data'),
            ('no_attention', 'remove_attention_mechanism')
        ]
        ablation_results = {}
        for name, ablation_func in ablation_variants:
            data_variant = trainer.data
            if ablation_func:
                from ablation_studies import remove_ppi_edges, remove_pathway_information, remove_expression_data, remove_attention_mechanism
                func = locals()[ablation_func]
                data_variant = func(data_variant)
            trainer.data = data_variant
            results, _ = trainer.train_all_models()
            best_model_name = _best_model_name_from_results(results)
            best_f1_score = results[best_model_name]["test_metrics"].get("f1", 0.0)
            if not args.no_wandb:
                wandb.log({f"ablation_{name}_f1": best_f1_score})
            ablation_results[name] = results
        print('Ablation study results:', ablation_results)
        results = ablation_results['full']
        print("[INFO] Ablation study complete.")
    elif args.grid_search:
        print("[INFO] Running grid search...")
        import itertools
        grid = {
            'learning_rate': [1e-2, 1e-3, 5e-4, 1e-4],
            'hidden_channels': [64, 128, 256],
            'dropout': [0.1, 0.3, 0.5],
            'num_layers': [2, 3, 4],
            'gat_heads': [4, 8, 16] if args.model == 'GAT' else [None],
        }
        keys, values = zip(*grid.items())
        best_f1 = -1
        best_config = None
        best_results = None
        for v in itertools.product(*values):
            config = dict(zip(keys, v))
            trainer.config.update({
                'learning_rate': config['learning_rate'],
                'hidden_channels': config['hidden_channels'],
                'dropout': config['dropout'],
                'num_layers': config['num_layers'],
            })
            if args.model == 'GAT':
                trainer.config['gat_heads'] = config['gat_heads']
            results, _ = trainer.train_all_models()
            best_model_name = _best_model_name_from_results(results)
            best_f1_score = results[best_model_name]["test_metrics"].get("f1", 0.0)
            if not args.no_wandb:
                wandb.log({"config": str(config), "f1": best_f1_score})
            if best_f1_score > best_f1:
                best_f1 = best_f1_score
                best_config = config
                best_results = results
        print(f'Best config: {best_config}, Best F1: {best_f1}')
        results = best_results
        print("[INFO] Grid search complete.")
    else:
        print("[INFO] Training models...")
        results, ensemble = trainer.train_all_models()
        print("[INFO] Training complete.")

    # Generate and save visualizations for the best model
    print("[INFO] Generating visualizations...")
    best_model_name = _best_model_name_from_results(results)
    num_classes = int(torch.max(trainer.data.y).item() + 1) if hasattr(trainer.data, 'y') else 2
    best_model = get_model(
        best_model_name,
        in_channels=trainer.data.num_node_features,
        hidden_channels=trainer.config['hidden_channels'],
        num_layers=trainer.config['num_layers'],
        out_channels=num_classes,
    ).to(device)
    
    # Only load state dict if the file exists
    model_file = f"results/{best_model_name}_pretrained.pt" if args.pretrain else f"results/{best_model_name}_best.pt"
    if os.path.exists(model_file):
        best_model.load_state_dict(torch.load(model_file))
    else:
        print(f"Warning: Model file {model_file} not found. Using untrained model for visualization.")
    
    # Generate visualizations
    vis_data = trainer.data
    if isinstance(vis_data, list):
        # If it's a list of lists, get the first element of the first list
        if len(vis_data) > 0 and isinstance(vis_data[0], list):
            vis_data = vis_data[0][0]
        else:
            vis_data = vis_data[0]
    trainer.visualize_feature_space(best_model, vis_data)
    if hasattr(best_model, 'get_attention_weights'):
        trainer.visualize_attention_maps(best_model, vis_data)
    trainer.feature_importance_analysis(best_model, vis_data)
    trainer.plot_learning_curves(results)
    print("[INFO] Visualizations complete.")

    # Log to wandb
    if not args.no_wandb:
        print("[INFO] Logging results to wandb...")
        wandb.log({
            'tsne_visualization': wandb.Image('results/feature_visualization.png'),
            'attention_map': wandb.Image('results/attention_map.png') if hasattr(best_model, 'get_attention_weights') else None,
            'feature_importance': wandb.Image('results/feature_importance.png'),
            'learning_curves': wandb.Image(f'results/{best_model_name}_learning_curves.png')
        })
        print("[INFO] wandb logging complete.")
        wandb.finish()
    if args.export_results:
        export_payload: Dict[str, Any] = {
            "args": vars(args),
            "dataset": {},
            "test_metrics": {},
        }
        d = trainer.data
        for attr in ("dataset_name", "dataset_description", "k_neighbors"):
            if hasattr(d, attr):
                export_payload["dataset"][attr] = getattr(d, attr)
        if hasattr(d, "num_nodes"):
            export_payload["dataset"]["num_nodes"] = int(d.num_nodes)
        if hasattr(d, "num_edges"):
            export_payload["dataset"]["num_edges"] = int(d.num_edges)
        for model_name, res in results.items():
            tm = res.get("test_metrics", {})
            row: Dict[str, Any] = {}
            for k, v in tm.items():
                if k == "confusion_matrix":
                    row[k] = np.asarray(v).tolist()
                elif isinstance(v, (float, int, str, bool, type(None))):
                    row[k] = v
                elif hasattr(v, "item"):
                    row[k] = v.item() if v.size == 1 else v.tolist()
            export_payload["test_metrics"][model_name] = row
        os.makedirs(os.path.dirname(os.path.abspath(args.export_results)) or ".", exist_ok=True)
        with open(args.export_results, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2, default=str)
        print(f"[INFO] Wrote result summary to {args.export_results}")
    print("[INFO] Script finished successfully.")


def _best_model_name_from_results(results: Dict[str, Any]) -> str:
    """Pick a model to visualize; use test F1 (val_metrics may be empty in this trainer)."""
    def f1_of(name: str) -> float:
        tm = results[name].get("test_metrics", {})
        v = tm.get("f1", 0.0)
        return float(v) if v is not None else 0.0

    return max(results.keys(), key=f1_of)

def parse_args():
    parser = argparse.ArgumentParser(description="GNN Cancer Training")
    parser.add_argument('--cancer_type', type=str, required=True, help='Cancer type code (e.g., BRCA, LUAD, LUSC)')
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
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--grid_search', action='store_true', help='Enable grid search over hyperparameters')
    parser.add_argument('--ablation', action='store_true', help='Enable ablation study automation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing data files')
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Do not use Weights & Biases (no API key required).",
    )
    parser.add_argument(
        "--export-results",
        dest="export_results",
        type=str,
        default="",
        help="Write JSON with test metrics and dataset info to this path (e.g. results/reproducible_run.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose per-epoch output.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
