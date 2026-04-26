import sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Optional, Tuple
import logging
from gnn_cancer.models.gnn_models import get_model
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import shap
import lime
import lime.lime_tabular

class Visualizer:
    def __init__(self, data_dir: Path = Path("data/processed")):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
    def plot_graph_structure(self, G: nx.Graph, output_path: Path) -> None:
        """Plot the graph structure with node features."""
        plt.figure(figsize=(12, 8))
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G)
        
        # Draw nodes with mutation status
        node_colors = [G.nodes[node]['mutation_status'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='coolwarm')
        
        # Draw edges with different colors based on edge type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            if data['edge_type'] == 'ppi':
                edge_colors.append('blue')
            elif data['edge_type'] == 'pathway':
                edge_colors.append('green')
            else:  # coexpression
                edge_colors.append('red')
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5)
        
        plt.title("Graph Structure with Node Features and Edge Types")
        plt.savefig(output_path)
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(output_path)
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: Path) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.close()
    
    def plot_feature_importance(self, model: torch.nn.Module, feature_names: List[str], output_path: Path) -> None:
        """Plot feature importance based on model weights and SHAP values."""
        # Get weights from first layer
        weights = model.convs[0].weight.data.cpu().numpy()
        importance = np.abs(weights).mean(axis=1)
        
        # Sort features by importance
        idx = np.argsort(importance)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot weight-based importance
        ax1.barh(range(len(importance)), importance[idx])
        ax1.set_yticks(range(len(importance)))
        ax1.set_yticklabels([feature_names[i] for i in idx])
        ax1.set_xlabel('Feature Importance (Weights)')
        ax1.set_title('Weight-based Feature Importance')
        
        # Calculate SHAP values
        explainer = shap.DeepExplainer(model, torch.randn(100, len(feature_names)))
        shap_values = explainer.shap_values(torch.randn(100, len(feature_names)))
        
        # Plot SHAP values
        shap.summary_plot(shap_values, feature_names=feature_names, show=False, ax=ax2)
        ax2.set_title('SHAP Feature Importance')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]], output_path: Path) -> None:
        """Plot training history."""
        plt.figure(figsize=(10, 6))
        for metric, values in history.items():
            plt.plot(values, label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
    
    def plot_embedding_visualization(self, model: torch.nn.Module, data: torch.Tensor, output_path: Path) -> None:
        """Plot t-SNE visualization of node embeddings."""
        # Get node embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embeddings(data)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title('t-SNE Visualization of Node Embeddings')
        plt.savefig(output_path)
        plt.close()
    
    def plot_attention_weights(self, model: torch.nn.Module, data: torch.Tensor, output_path: Path) -> None:
        """Plot attention weights for selected nodes."""
        # Get attention weights
        model.eval()
        with torch.no_grad():
            attention_weights = model.get_attention_weights(data)
        
        # Plot attention heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights, cmap='viridis')
        plt.title('Attention Weights Heatmap')
        plt.savefig(output_path)
        plt.close()
    
    def plot_feature_correlations(self, data: pd.DataFrame, output_path: Path) -> None:
        """Plot feature correlation matrix."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig(output_path)
        plt.close()
    
    def plot_learning_curves(self, train_sizes: np.ndarray, train_scores: np.ndarray, 
                           val_scores: np.ndarray, output_path: Path) -> None:
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
        plt.plot(train_sizes, val_scores.mean(axis=1), label='Cross-validation score')
        plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Load graph and model predictions
    G = nx.read_gpickle(visualizer.data_dir / "graph.pkl")
    model = get_model("gcn", in_channels=G.number_of_nodes())
    model.load_state_dict(torch.load(visualizer.data_dir / "gcn_best.pt"))
    
    # Generate visualizations
    visualizer.plot_graph_structure(G, visualizer.data_dir / "graph_structure.png")
    
    # Load predictions and true labels
    y_true = np.load(visualizer.data_dir / "y_true.npy")
    y_pred = np.load(visualizer.data_dir / "y_pred.npy")
    y_pred_proba = np.load(visualizer.data_dir / "y_pred_proba.npy")
    
    visualizer.plot_confusion_matrix(y_true, y_pred, visualizer.data_dir / "confusion_matrix.png")
    visualizer.plot_roc_curve(y_true, y_pred_proba, visualizer.data_dir / "roc_curve.png")
    
    # Plot feature importance
    feature_names = list(G.nodes())
    visualizer.plot_feature_importance(model, feature_names, visualizer.data_dir / "feature_importance.png")
    
    # Plot training history
    history = {
        "train_loss": np.load(visualizer.data_dir / "train_loss.npy"),
        "val_loss": np.load(visualizer.data_dir / "val_loss.npy"),
        "train_f1": np.load(visualizer.data_dir / "train_f1.npy"),
        "val_f1": np.load(visualizer.data_dir / "val_f1.npy")
    }
    visualizer.plot_training_history(history, visualizer.data_dir / "training_history.png")
    
    # Plot embedding visualization
    data = torch.load(visualizer.data_dir / "data.pt")
    visualizer.plot_embedding_visualization(model, data, visualizer.data_dir / "embeddings.png")
    
    # Plot attention weights
    visualizer.plot_attention_weights(model, data, visualizer.data_dir / "attention_weights.png")
    
    # Plot feature correlations
    feature_data = pd.read_csv(visualizer.data_dir / "feature_matrix.csv")
    visualizer.plot_feature_correlations(feature_data, visualizer.data_dir / "feature_correlations.png")
    
    # Plot learning curves
    train_sizes = np.load(visualizer.data_dir / "train_sizes.npy")
    train_scores = np.load(visualizer.data_dir / "train_scores.npy")
    val_scores = np.load(visualizer.data_dir / "val_scores.npy")
    visualizer.plot_learning_curves(train_sizes, train_scores, val_scores, 
                                  visualizer.data_dir / "learning_curves.png") 
