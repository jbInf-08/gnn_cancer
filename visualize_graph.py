import torch
import networkx as nx
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import numpy as np
from torch_geometric.utils import to_networkx
import seaborn as sns
import os
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_heterogeneous_graph(data, output_dir):
    """Visualize the heterogeneous graph structure."""
    logger.info("Creating graph visualizations...")
    
    # Create output directory
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Node Type Distribution
    node_types = {
        'gene': data['gene'].x.size(0),
        'go': data['go'].x.size(0),
        'pubmed': data['pubmed'].x.size(0)
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(node_types.keys(), node_types.values())
    plt.title('Distribution of Node Types')
    plt.ylabel('Number of Nodes')
    plt.savefig(output_dir / 'node_type_distribution.png')
    plt.close()
    
    # 2. Edge Type Distribution
    edge_types = {
        'gene-go': data['gene', 'associated_with', 'go'].edge_index.size(1),
        'gene-pubmed': data['gene', 'cited_in', 'pubmed'].edge_index.size(1)
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(edge_types.keys(), edge_types.values())
    plt.title('Distribution of Edge Types')
    plt.ylabel('Number of Edges')
    plt.savefig(output_dir / 'edge_type_distribution.png')
    plt.close()
    
    # 3. Create a subgraph for visualization (using only a subset of nodes)
    # Convert to networkx graph
    G = nx.Graph()
    
    # Add gene nodes
    for i in range(min(50, data['gene'].x.size(0))):
        G.add_node(f'gene_{i}', type='gene')
    
    # Add GO nodes
    for i in range(min(20, data['go'].x.size(0))):
        G.add_node(f'go_{i}', type='go')
    
    # Add PubMed nodes
    for i in range(min(30, data['pubmed'].x.size(0))):
        G.add_node(f'pubmed_{i}', type='pubmed')
    
    # Add edges
    gene_go_edges = data['gene', 'associated_with', 'go'].edge_index.t().cpu().numpy()
    gene_pubmed_edges = data['gene', 'cited_in', 'pubmed'].edge_index.t().cpu().numpy()
    
    for i, j in gene_go_edges[:100]:  # Limit to 100 edges for visualization
        if i < 50 and j < 20:  # Only include nodes in our subgraph
            G.add_edge(f'gene_{i}', f'go_{j}', type='gene-go')
    
    for i, j in gene_pubmed_edges[:100]:  # Limit to 100 edges for visualization
        if i < 50 and j < 30:  # Only include nodes in our subgraph
            G.add_edge(f'gene_{i}', f'pubmed_{j}', type='gene-pubmed')
    
    # 4. Plot the subgraph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    node_colors = {
        'gene': 'red',
        'go': 'blue',
        'pubmed': 'green'
    }
    
    for node_type in ['gene', 'go', 'pubmed']:
        nodes = [n for n, d in G.nodes(data=True) if d['type'] == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors[node_type],
                             label=node_type, node_size=100)
    
    # Draw edges
    edge_colors = {
        'gene-go': 'purple',
        'gene-pubmed': 'orange'
    }
    
    for edge_type in ['gene-go', 'gene-pubmed']:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors[edge_type],
                             label=edge_type, alpha=0.5)
    
    plt.title('BRCA1 Heterogeneous Graph (Subgraph)')
    plt.legend()
    plt.axis('off')
    plt.savefig(output_dir / 'graph_visualization.png')
    plt.close()
    
    # 5. Create a heatmap of node features
    plt.figure(figsize=(15, 5))
    
    # Gene features
    plt.subplot(1, 3, 1)
    sns.heatmap(data['gene'].x[:50].cpu().numpy(), cmap='viridis')
    plt.title('Gene Node Features')
    
    # GO features
    plt.subplot(1, 3, 2)
    sns.heatmap(data['go'].x[:20].cpu().numpy(), cmap='viridis')
    plt.title('GO Term Features')
    
    # PubMed features
    plt.subplot(1, 3, 3)
    sns.heatmap(data['pubmed'].x[:30].cpu().numpy(), cmap='viridis')
    plt.title('PubMed Features')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_heatmaps.png')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    try:
        # Create necessary directories
        os.makedirs('data/processed/brca1', exist_ok=True)
        os.makedirs('visualizations/brca1', exist_ok=True)
        
        # Load the heterogeneous graph data
        data = torch.load('data/processed/brca1/heterogeneous_graph.pt')
        
        # Create visualizations
        visualize_heterogeneous_graph(data, 'visualizations/brca1')
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        traceback.print_exc()
        print(f"Exception occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 