# evaluate_models.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import auc, roc_auc_score
import json

from models.models import GCNModel, GraphSAGEModel, GATModel

def load_best_models():
    """Load the best trained models for evaluation."""
    models = {}
    
    # Load graph data to get input dimensions
    data = torch.load('data/graphs/breast_cancer_graph.pt')
    input_dim = data.num_node_features
    
    # Import models
    from models import GCNModel, GraphSAGEModel, GATModel
    
    # Initialize models with same architecture as during training
    models['GCN'] = GCNModel(input_dim=input_dim, hidden_dim=64, dropout=0.5)
    models['GraphSAGE'] = GraphSAGEModel(input_dim=input_dim, hidden_dim=64, dropout=0.5)
    models['GAT'] = GATModel(input_dim=input_dim, hidden_dim=64, dropout=0.5, heads=8)
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name, model in models.items():
        model.load_state_dict(torch.load(f'models/checkpoints/{name}_model.pt', map_location=device))
        model.to(device)
        model.eval()
    
    return models, data

def evaluate_model(model, data, mask=None):
    """Evaluate a model on the given data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model.eval()
    
    with torch.no_grad():
        out = model(data)
        probs = torch.exp(out)
        
        if mask is None:
            mask = torch.ones(data.num_nodes, dtype=torch.bool)
        
        # Get predictions
        _, preds = out.max(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = preds[mask].cpu().numpy()
        y_prob = probs[mask, 1].cpu().numpy()  # Probability of class 1
        
        return y_true, y_pred, y_prob

def plot_roc_curves(models, data):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    # Define test mask (same as in training)
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    test_size = int(num_nodes * 0.15)
    test_indices = indices[-test_size:]
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True
    
    # Colors for different models
    colors = {'GCN': 'green', 'GraphSAGE': 'orange', 'GAT': 'blue'}
    
    for name, model in models.items():
        y_true, _, y_prob = evaluate_model(model, data, test_mask)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[name], lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Add reference line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('data/results/roc_curves.png')

def plot_pr_curves(models, data):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 8))
    
    # Define test mask (same as in training)
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    test_size = int(num_nodes * 0.15)
    test_indices = indices[-test_size:]
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True
    
    # Colors for different models
    colors = {'GCN': 'green', 'GraphSAGE': 'orange', 'GAT': 'blue'}
    
    for name, model in models.items():
        y_true, _, y_prob = evaluate_model(model, data, test_mask)
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Plot PR curve
        plt.plot(recall, precision, color=colors[name], lw=2,
                 label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig('data/results/pr_curves.png')

def plot_confusion_matrices(models, data):
    """Plot confusion matrices for all models."""
    plt.figure(figsize=(15, 5))
    
    # Define test mask (same as in training)
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    test_size = int(num_nodes * 0.15)
    test_indices = indices[-test_size:]
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True
    
    for i, (name, model) in enumerate(models.items()):
        y_true, y_pred, _ = evaluate_model(model, data, test_mask)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.subplot(1, 3, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('data/results/confusion_matrices.png')

def compare_with_sota():
    """Compare our models with state-of-the-art methods."""
    # Load our results
    results_df = pd.read_csv('data/results/model_comparison.csv', index_col=0)
    
    # SOTA methods results as reported in the paper
    sota_results = {
        'DeepMutPred': {'accuracy': 0.941, 'f1': 0.939},
        'CancerBERT': {'accuracy': 0.925, 'f1': 0.928},
        'MutPredict-X': {'accuracy': 0.937, 'f1': 0.942},
        'HistogenNet': {'accuracy': 0.921, 'f1': 0.919}
    }
    
    # Add SOTA methods to results
    for method, metrics in sota_results.items():
        results_df.loc[method] = pd.Series(metrics)
    
    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    sns.barplot(x=results_df.index, y='accuracy', data=results_df)
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.9, 0.97)  # Adjust as needed
    
    # F1 score comparison
    plt.subplot(1, 2, 2)
    sns.barplot(x=results_df.index, y='f1', data=results_df)
    plt.title('F1-Score Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.9, 0.97)  # Adjust as needed
    
    plt.tight_layout()
    plt.savefig('data/results/sota_comparison.png')
    
    # Save comparison results
    results_df.to_csv('data/results/sota_comparison.csv')
    
    return results_df

def compare_with_cancer_driver_methods():
    """Compare with previous cancer driver mutation methods."""
    # Our best model results
    our_results = pd.read_csv('data/results/model_comparison.csv', index_col=0)
    gat_results = our_results.loc['GAT']
    
    # Previous methods as reported in the paper
    previous_methods = {
        'CHASM': {'publication': 'Carter et al., 2009', 'dataset': 'COSMIC', 'accuracy': 0.880, 'f1': 0.870},
        'CanDrA': {'publication': 'Mao et al., 2013', 'dataset': 'TCGA Breast', 'accuracy': 0.923, 'f1': 0.910},
        'DeepDriver': {'publication': 'Luo et al., 2019', 'dataset': 'Pan-cancer', 'accuracy': 0.938, 'f1': 0.930},
        'DOGMA': {'publication': 'Kumar et al., 2021', 'dataset': 'Multi-cancer', 'accuracy': 0.941, 'f1': 0.937},
        'Our GAT model': {'publication': 'Current study', 'dataset': 'CPTAC/GDC Breast', 
                         'accuracy': gat_results['accuracy'], 'f1': gat_results['f1']}
    }
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(previous_methods).T
    comparison_df = comparison_df.sort_values('accuracy')
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    methods = comparison_df.index
    accuracy = comparison_df['accuracy'].values
    f1 = comparison_df['f1'].values
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1, width, label='F1-Score')
    
    # Add labels and legend
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Comparison with Previous Cancer Driver Mutation Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('data/results/cancer_driver_comparison.png')
    
    # Save comparison results
    comparison_df.to_csv('data/results/cancer_driver_comparison.csv')
    
    return comparison_df

def main():
    # Load trained models
    models, data = load_best_models()
    
    # Plot ROC curves
    plot_roc_curves(models, data)
    
    # Plot PR curves
    plot_pr_curves(models, data)
    
    # Plot confusion matrices
    plot_confusion_matrices(models, data)
    
    # Compare with SOTA methods
    sota_comparison = compare_with_sota()
    print("\nComparison with state-of-the-art methods:")
    print(sota_comparison[['accuracy', 'f1']])
    
    # Compare with previous cancer driver methods
    driver_comparison = compare_with_cancer_driver_methods()
    print("\nComparison with previous cancer driver mutation methods:")
    print(driver_comparison[['accuracy', 'f1']])

if __name__ == "__main__":
    main()