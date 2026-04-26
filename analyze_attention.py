import sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
# analyze_attention.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
from gnn_cancer.models.models import GCNModel, GraphSAGEModel, GATModel
import seaborn as sns

def load_gat_model_and_data():
    """Load the trained GAT model and data."""
    # Load graph data
    data = torch.load('data/graphs/breast_cancer_graph.pt')
    
    # Load node mapping
    with open('data/graphs/node_mapping.json', 'r') as f:
        node_mapping = json.load(f)
    
    # Reverse the mapping to get gene names from indices
    reverse_mapping = {int(v): k for k, v in node_mapping.items()}
    
    # Initialize GAT model
    input_dim = data.num_node_features
    model = GATModel(input_dim=input_dim, hidden_dim=64, dropout=0.5, heads=8)
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('models/checkpoints/GAT_model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    return model, data, reverse_mapping

def extract_attention_weights(model, data):
    """Extract attention weights from the GAT model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Register hook to get attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # GAT attention weights
        attention_weights.append(module.alpha.detach())
    
    # Register hooks for all attention layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.module.Module) and hasattr(module, 'alpha'):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass to get attention weights
    with torch.no_grad():
        _ = model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_weights

def analyze_high_attention_nodes(attention_weights, data, reverse_mapping):
    """Analyze nodes with high attention weights."""
    # Get edge index
    edge_index = data.edge_index.cpu().numpy()
    
    # Get attention from first layer (most interpretable)
    first_layer_attention = attention_weights[0].cpu().numpy()
    
    # Calculate average attention per node (across all heads)
    num_nodes = data.num_nodes
    node_attention_scores = np.zeros(num_nodes)
    
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i]
        target = edge_index[1, i]
        node_attention_scores[target] += np.mean(first_layer_attention[i, :])
    
    # Normalize by dividing by number of incoming edges
    in_degree = np.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        target = edge_index[1, i]
        in_degree[target] += 1
    
    # Avoid division by zero
    in_degree[in_degree == 0] = 1
    node_attention_scores = node_attention_scores / in_degree
    
    # Create dataframe with node scores
    node_scores = []
    for i in range(num_nodes):
        if i in reverse_mapping:
            gene_name = reverse_mapping[i]
            node_scores.append({
                'gene': gene_name,
                'attention_score': node_attention_scores[i],
                'node_idx': i
            })
    
    node_scores_df = pd.DataFrame(node_scores)
    node_scores_df = node_scores_df.sort_values('attention_score', ascending=False)
    
    # Save top 100 genes by attention score
    top_genes = node_scores_df.head(100)
    top_genes.to_csv('data/results/top_attention_genes.csv', index=False)
    
    # Plot top 20 genes
    plt.figure(figsize=(12, 8))
    sns.barplot(x='gene', y='attention_score', data=node_scores_df.head(20))
    plt.title('Top 20 Genes by Attention Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('data/results/top_attention_genes.png')
    
    return node_scores_df

def visualize_attention_subgraph(attention_weights, data, reverse_mapping, top_n=20):
    """Visualize a subgraph with the highest attention edges."""
    # Get edge index and attention weights
    edge_index = data.edge_index.cpu().numpy()
    first_layer_attention = attention_weights[0].cpu().numpy()
    
    # Calculate average attention per edge (across all heads)
    edge_attention = np.mean(first_layer_attention, axis=1)
    
    # Create edge list with attention scores
    edges = []
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i]
        target = edge_index[1, i]
        
        if source in reverse_mapping and target in reverse_mapping:
            source_gene = reverse_mapping[source]
            target_gene = reverse_mapping[target]
            
            edges.append({
                'source': source_gene,
                'target': target_gene,
                'attention': edge_attention[i],
                'source_idx': source,
                'target_idx': target
            })
    
    # Create dataframe and sort by attention
    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values('attention', ascending=False)
    
    # Save top 100 edges by attention
    top_edges = edges_df.head(100)
    top_edges.to_csv('data/results/top_attention_edges.csv', index=False)
    
    # Extract genes in top edges
    top_genes = set()
    for _, row in edges_df.head(top_n).iterrows():
        top_genes.add(row['source'])
        top_genes.add(row['target'])
    
    # Create NetworkX graph for visualization
    G = nx.Graph()
    
    # Add nodes
    for gene in top_genes:
        G.add_node(gene)
    
    # Add edges with attention weights
    for _, row in edges_df.iterrows():
        if row['source'] in top_genes and row['target'] in top_genes:
            G.add_edge(row['source'], row['target'], weight=row['attention'])
    
    # Visualize graph
    plt.figure(figsize=(12, 12))
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw edges with width proportional to attention
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title('High Attention Subgraph')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('data/results/attention_subgraph.png')
    
    return edges_df

def main():
    # Load model and data
    model, data, reverse_mapping = load_gat_model_and_data()
    
    # Extract attention weights
    attention_weights = extract_attention_weights(model, data)
    
    # Analyze high attention nodes
    node_scores = analyze_high_attention_nodes(attention_weights, data, reverse_mapping)
    print("Top 10 genes by attention score:")
    print(node_scores.head(10))
    
    # Visualize attention subgraph
    edge_scores = visualize_attention_subgraph(attention_weights, data, reverse_mapping)
    print("\nTop 10 gene interactions by attention weight:")
    print(edge_scores.head(10))
    
    # Check if known cancer genes are in top attention nodes
    known_cancer_genes = ['TP53', 'PIK3CA', 'BRCA1', 'BRCA2', 'PTEN', 'AKT1', 'CDH1', 'GATA3', 'RB1']
    
    found_genes = set(node_scores['gene'].head(100)) & set(known_cancer_genes)
    print(f"\nFound {len(found_genes)} known cancer genes in top 100 attention nodes:")
    print(found_genes)

if __name__ == "__main__":
    main()
