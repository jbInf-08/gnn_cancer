import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from typing import Dict, List, Tuple
import logging

class ProcessingVisualizer:
    def __init__(self, data_dir: Path = Path("data/raw")):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
    def plot_preprocessing_pipeline(self, output_path: Path) -> None:
        """Plot the preprocessing pipeline showing data integration steps."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Preprocessing Pipeline", fontsize=16)
        
        # Load example data
        mutation_data = pd.read_csv(self.data_dir / "tcga/BRCA/mutation/mutations.csv")
        expression_data = pd.read_csv(self.data_dir / "tcga/BRCA/expression/expression.csv")
        cnv_data = pd.read_csv(self.data_dir / "tcga/BRCA/cnv/cnv.csv")
        
        # Plot mutation data distribution
        sns.histplot(data=mutation_data, x='mutation_status', ax=axes[0,0])
        axes[0,0].set_title("Mutation Status Distribution")
        
        # Plot expression data distribution
        sns.histplot(data=expression_data, x='expression', ax=axes[0,1])
        axes[0,1].set_title("Expression Level Distribution")
        
        # Plot CNV data distribution
        sns.histplot(data=cnv_data, x='cnv', ax=axes[1,0])
        axes[1,0].set_title("CNV Value Distribution")
        
        # Plot correlation between features
        corr_data = pd.concat([
            mutation_data['mutation_status'],
            expression_data['expression'],
            cnv_data['cnv']
        ], axis=1)
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title("Feature Correlations")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_postprocessing_results(self, output_path: Path) -> None:
        """Plot postprocessing results showing model predictions and interpretations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Postprocessing Results", fontsize=16)
        
        # Load model predictions
        y_true = np.load(self.data_dir / "y_true.npy")
        y_pred = np.load(self.data_dir / "y_pred.npy")
        y_pred_proba = np.load(self.data_dir / "y_pred_proba.npy")
        
        # Plot prediction distribution
        sns.histplot(y_pred_proba, ax=axes[0,0])
        axes[0,0].set_title("Prediction Probability Distribution")
        axes[0,0].set_xlabel("Prediction Probability")
        axes[0,0].set_ylabel("Count")
        
        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title("Confusion Matrix")
        axes[0,1].set_xlabel("Predicted Label")
        axes[0,1].set_ylabel("True Label")
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1,0].set_xlim([0.0, 1.0])
        axes[1,0].set_ylim([0.0, 1.05])
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].legend(loc="lower right")
        
        # Plot feature importance
        feature_importance = np.load(self.data_dir / "feature_importance.npy")
        feature_names = np.load(self.data_dir / "feature_names.npy", allow_pickle=True)
        idx = np.argsort(feature_importance)
        axes[1,1].barh(range(len(feature_importance)), feature_importance[idx])
        axes[1,1].set_yticks(range(len(feature_importance)))
        axes[1,1].set_yticklabels([feature_names[i] for i in idx])
        axes[1,1].set_xlabel('Feature Importance')
        axes[1,1].set_title('Feature Importance Analysis')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_data_flow(self, output_path: Path) -> None:
        """Plot the complete data flow from raw data to predictions."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Define stages
        stages = [
            "Raw Data",
            "Data Integration",
            "Graph Construction",
            "Feature Engineering",
            "Model Training",
            "Prediction"
        ]
        
        # Define connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)
        ]
        
        # Create graph
        G = nx.DiGraph()
        for i, stage in enumerate(stages):
            G.add_node(i, label=stage)
        G.add_edges_from(connections)
        
        # Draw graph
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=2000, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                             arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, {i: G.nodes[i]['label'] 
                                       for i in G.nodes()}, ax=ax)
        
        plt.title("Data Flow Pipeline")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize visualizer
    visualizer = ProcessingVisualizer()
    
    # Generate preprocessing visualization
    visualizer.plot_preprocessing_pipeline(
        visualizer.data_dir / "preprocessing_pipeline.png"
    )
    
    # Generate postprocessing visualization
    visualizer.plot_postprocessing_results(
        visualizer.data_dir / "postprocessing_results.png"
    )
    
    # Generate data flow visualization
    visualizer.plot_data_flow(
        visualizer.data_dir / "data_flow.png"
    ) 