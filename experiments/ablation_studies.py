import sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
# ablation_studies.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from gnn_cancer.models.models import GCNModel, GraphSAGEModel, GATModel
from gnn_cancer.utils.train_model import train_model

def run_ablation_study(full_data, model_class, ablation_configs):
    """Run ablation studies by removing different components of the model/data."""
    results = {}
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training parameters
    learning_rate = 0.0001
    weight_decay = 1e-5
    epochs = 80
    
    # Base model and data
    base_model = model_class(input_dim=full_data.num_node_features, hidden_dim=64, dropout=0.5, heads=8)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print("Running ablation studies...")
    
    # Run full model first
    print("\nTraining full model...")
    full_metrics = train_model(base_model, full_data.clone(), base_optimizer, epochs=epochs)
    results['Full Model'] = full_metrics
    
    # Run ablation configurations
    for name, config_func in ablation_configs.items():
        print(f"\nTraining model without {name}...")
        
        # Create modified data or model
        modified_data = config_func(full_data.clone())
        
        # Create new model instance
        if name == 'Attention Mechanism':
            # For attention mechanism ablation, use GCN instead of GAT
            from gnn_cancer.models.models import GCNModel
            ablation_model = GCNModel(input_dim=modified_data.num_node_features, hidden_dim=64, dropout=0.5)
        else:
            ablation_model = model_class(input_dim=modified_data.num_node_features, hidden_dim=64, dropout=0.5, heads=8)
        
        ablation_optimizer = torch.optim.Adam(ablation_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Train and evaluate
        metrics = train_model(ablation_model, modified_data, ablation_optimizer, epochs=epochs)
        results[f"w/o {name}"] = metrics
    
    # Create results dataframe
    results_df = pd.DataFrame(results).T
    results_df = results_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auprc', 'mcc']]
    
    print("\nAblation Study Results:")
    print(results_df)
    
    # Save results
    results_df.to_csv('data/results/ablation_study.csv')
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['accuracy', 'f1']
    models = list(results.keys())
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 2, i+1)
        plt.bar(np.arange(len(models)), [results[m][metric] for m in models], width=0.6)
        plt.xticks(np.arange(len(models)), models, rotation=45, ha='right')
        plt.ylabel(f'{metric} Score')
        plt.title(f'Ablation Study - {metric}')
    
    plt.tight_layout()
    plt.savefig('data/results/ablation_study.png')
    
    return results_df

def remove_ppi_edges(data):
    """Remove protein-protein interaction edges (edge_type==1)."""
    if data.edge_attr is not None:
        # Keep only edges that are NOT PPI (edge_type != 1)
        keep_mask = data.edge_attr != 1
        data.edge_index = data.edge_index[:, keep_mask]
        data.edge_attr = data.edge_attr[keep_mask]
    return data

def remove_expression_data(data):
    """Remove expression data from node features."""
    # Assuming expression data is the second feature
    features = data.x.clone()
    features[:, 1] = 0  # Zero out expression features
    data.x = features
    
    return data

def remove_pathway_information(data):
    """Remove pathway-based edges (edge_type==2)."""
    if data.edge_attr is not None:
        # Keep only edges that are NOT pathway (edge_type != 2)
        keep_mask = data.edge_attr != 2
        data.edge_index = data.edge_index[:, keep_mask]
        data.edge_attr = data.edge_attr[keep_mask]
    return data

def remove_attention_mechanism(data):
    """Ablation for attention mechanism: no data change, model will be swapped to GCN."""
    return data

def main():
    # Load graph data
    data = torch.load('data/graphs/breast_cancer_graph.pt')
    
    # Define ablation configurations
    ablation_configs = {
        'PPI Edges': remove_ppi_edges,
        'Expression Data': remove_expression_data,
        'Pathway Information': remove_pathway_information,
        'Attention Mechanism': remove_attention_mechanism
    }
    
    # Run ablation studies on GAT model
    results = run_ablation_study(data, GATModel, ablation_configs)

if __name__ == "__main__":
    main()
