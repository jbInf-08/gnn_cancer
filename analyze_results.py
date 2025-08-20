import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wandb
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import confusion_matrix, classification_report
import logging
from models.models import GATModel

# Add safe globals for PyTorch Geometric
add_safe_globals([Data])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Add SimpleGCN class from train.py ---
import torch.nn as nn
import torch.nn.functional as F
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ResultsAnalyzer:
    def __init__(self, data_dir=Path("data")):
        self.data_dir = data_dir
        self.metrics_file = data_dir / "processed" / "metrics.json"
        
    def load_metrics(self):
        """Load metrics from JSON file."""
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def analyze_class_distribution(self, data):
        """Analyze class distribution in the dataset."""
        labels = data.y.numpy()
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        logger.info("Class Distribution:")
        for cls, count in distribution.items():
            logger.info(f"Class {cls}: {count} samples ({count/len(labels)*100:.2f}%)")
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(distribution.keys()), y=list(distribution.values()))
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.savefig(self.data_dir / "class_distribution.png")
        plt.close()
        
        return distribution
    
    def analyze_feature_importance(self, data, model):
        """Analyze feature importance using model weights (GAT classifier layer)."""
        # Use classifier weights for feature importance
        weights = model.classifier.weight.data.cpu().numpy()
        importance = np.mean(np.abs(weights), axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance)), importance)
        plt.title("Feature Importance (GAT Classifier Layer)")
        plt.xlabel("Feature Index")
        plt.ylabel("Mean Absolute Weight")
        plt.savefig(self.data_dir / "feature_importance.png")
        plt.close()
        return importance
    
    def analyze_training_curves(self, history):
        """Analyze and plot training curves."""
        metrics = ['loss', 'accuracy', 'f1']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
        
        for i, metric in enumerate(metrics):
            train_metric = f'train_{metric}'
            val_metric = f'val_{metric}'
            
            axes[i].plot(history[train_metric], label=f'Train {metric.capitalize()}')
            axes[i].plot(history[val_metric], label=f'Validation {metric.capitalize()}')
            axes[i].set_title(f'{metric.capitalize()} Curves')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.data_dir / "training_curves.png")
        plt.close()
    
    def analyze_confusion_matrix(self, y_true, y_pred):
        """Analyze and plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(self.data_dir / "confusion_matrix.png")
        plt.close()
        
        return cm
    
    def analyze_model_performance(self, metrics):
        """Analyze and log model performance metrics"""
        logger.info("\nModel Performance Analysis:")
        
        # Access metrics from the test_metrics dictionary
        test_metrics = metrics['test_metrics']
        
        # Log basic metrics
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")
        logger.info(f"F1-score: {test_metrics['f1_score']:.4f}")
        logger.info(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        logger.info(f"AUPRC: {test_metrics['auprc']:.4f}")
        logger.info(f"Matthews Correlation Coefficient: {test_metrics['mcc']:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_metrics['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.data_dir / 'confusion_matrix.png')
        plt.close()
    
    def analyze_wandb_runs(self):
        """Analyze Weights & Biases runs."""
        api = wandb.Api()
        runs = api.runs("jvboy19-university-of-minnesota/gnn-cancer")
        
        metrics_df = pd.DataFrame()
        for run in runs:
            if run.state == "finished":
                metrics = run.summary
                metrics['run_id'] = run.id
                metrics['model'] = run.config.get('model', 'unknown')
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        
        # Plot comparison of different models
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=metrics_df, x='model', y='val_accuracy')
        plt.title("Model Performance Comparison")
        plt.xlabel("Model")
        plt.ylabel("Validation Accuracy")
        plt.savefig(self.data_dir / "model_comparison.png")
        plt.close()
        
        return metrics_df

def main():
    analyzer = ResultsAnalyzer()
    
    # Load data
    data = torch.load(analyzer.data_dir / "processed" / "BRCA_data.pt", weights_only=False)
    
    # Reconstruct and load the GAT model
    in_channels = data.num_node_features
    out_channels = len(torch.unique(data.y))
    hidden_channels = 64  # Use the same as in training
    num_heads = 8
    model = GATModel(in_channels, hidden_channels, out_channels, num_heads=num_heads)
    model.load_state_dict(torch.load(analyzer.data_dir / "best_model.pt"))
    model.eval()
    
    # Analyze class distribution
    analyzer.analyze_class_distribution(data)
    
    # Analyze feature importance
    analyzer.analyze_feature_importance(data, model)
    
    # Load and analyze metrics
    metrics = analyzer.load_metrics()
    analyzer.analyze_model_performance(metrics)
    
    # Analyze W&B runs
    metrics_df = analyzer.analyze_wandb_runs()
    
    logger.info("\nAnalysis complete. Check the data directory for visualization plots.")

if __name__ == "__main__":
    main() 