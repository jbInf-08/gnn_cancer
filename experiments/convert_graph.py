import pickle
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_networkx_graph(graph_path: Path) -> nx.Graph:
    """Load the NetworkX graph from pickle file."""
    logger.info(f"Loading graph from {graph_path}")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G

def convert_to_pytorch_geometric(G: nx.Graph) -> Data:
    """Convert NetworkX graph to PyTorch Geometric Data object with all node features and edge types."""
    logger.info("Converting graph to PyTorch Geometric format...")
    # Get node features
    node_features = []
    for node in G.nodes():
        features = [
            G.nodes[node].get('mutation_status', 0),
            G.nodes[node].get('expression_mean', 0.0),
            G.nodes[node].get('expression_std', 0.0),
            G.nodes[node].get('cnv', 0.0),
            G.nodes[node].get('protein_abundance', 0.0),
            G.nodes[node].get('driver_status', 0)
        ]
        node_features.append(features)
    # Convert to tensor and normalize
    x = torch.tensor(node_features, dtype=torch.float)
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(x), dtype=torch.float)
    # Edge type mapping
    edge_type_map = {'coexpression': 0, 'ppi': 1, 'pathway': 2}
    edge_index = []
    edge_types = []
    for edge in G.edges(data=True):
        n1, n2, attr = edge
        idx1 = list(G.nodes()).index(n1)
        idx2 = list(G.nodes()).index(n2)
        # Add both directions for undirected graph
        edge_index.append([idx1, idx2])
        edge_index.append([idx2, idx1])
        etype = edge_type_map.get(attr.get('edge_type', 'coexpression'), 0)
        edge_types.append(etype)
        edge_types.append(etype)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_types, dtype=torch.long)
    # Get node labels (mutation status)
    y = torch.tensor([G.nodes[node].get('mutation_status', 0) for node in G.nodes()], dtype=torch.long)
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    logger.info(f"Created PyTorch Geometric Data object with:")
    logger.info(f"- {data.num_nodes} nodes")
    logger.info(f"- {data.num_edges} edges")
    logger.info(f"- {data.num_node_features} node features")
    logger.info(f"- {data.edge_attr.size(0)} edge attributes (edge_type)")
    return data

def main():
    # Setup paths
    data_dir = Path("data/processed")
    graph_path = data_dir / "graph.pkl"
    output_path = data_dir / "BRCA_data.pt"
    
    # Load NetworkX graph
    G = load_networkx_graph(graph_path)
    
    # Convert to PyTorch Geometric Data
    data = convert_to_pytorch_geometric(G)
    
    # Save the data
    logger.info(f"Saving PyTorch Geometric Data to {output_path}")
    torch.save(data, output_path)
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_node_features}")
    print(f"Number of classes: {len(torch.unique(data.y))}")
    print(f"Class distribution: {torch.bincount(data.y).tolist()}")

if __name__ == "__main__":
    main() 