import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import NearestNeighbors
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_kaggle_data(data_path: Path):
    """Load and preprocess the Kaggle breast cancer dataset."""
    logger.info("Loading Kaggle breast cancer dataset...")
    df = pd.read_csv(data_path)
    
    # Remove the 'id' column and 'Unnamed: 32' column if they exist
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    
    # Convert diagnosis to binary (M=1, B=0)
    df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
    
    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, X.columns

def create_graph_data(X, y):
    """Convert tabular data to graph format."""
    logger.info("Converting data to graph format...")
    
    # Create a list to store individual graph data objects
    graph_list = []
    
    for i in range(len(X)):
        # Create a small graph for each sample (2 nodes: sample and a virtual node)
        n_nodes = 2
        adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        
        # Node features: original features for sample node, zeros for virtual node
        x = np.zeros((n_nodes, X.shape[1]))
        x[0] = X[i]  # Original features for sample node
        
        # Convert to PyTorch tensors
        x = torch.FloatTensor(x)
        y_tensor = torch.LongTensor([y[i]])
        edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_matrix))
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index, y=y_tensor)
        graph_list.append(data)
    
    return graph_list

def create_large_graph(X, y, feature_names, k_neighbors=5):
    """Create a single large graph for node classification using k-nearest neighbors."""
    logger.info(f"Creating a single large graph for node classification with {k_neighbors} nearest neighbors...")
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)  # +1 because each point is its own neighbor
    distances, indices = nbrs.kneighbors(X)
    
    # Remove self-loops (first neighbor is the point itself)
    indices = indices[:, 1:]  # Remove self-loops
    distances = distances[:, 1:]
    
    # Create edge index
    rows = np.repeat(np.arange(len(X)), k_neighbors)
    cols = indices.flatten()
    edge_index = torch.LongTensor([rows, cols])
    
    # Calculate edge features (similarity scores)
    edge_attr = torch.FloatTensor(1 / (1 + distances.flatten()))  # Convert distances to similarities
    
    # Create node features with additional structural information
    x = torch.FloatTensor(X)
    
    # Add node degree as an additional feature
    degrees = torch.zeros(len(X))
    for i in range(len(X)):
        degrees[i] = (edge_index[0] == i).sum() + (edge_index[1] == i).sum()
    x = torch.cat([x, degrees.unsqueeze(1)], dim=1)
    
    # Create labels
    y = torch.LongTensor(y)
    
    # Create PyTorch Geometric data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        feature_names=feature_names + ['degree']  # Store feature names for reference
    )
    
    return data

def main():
    # Set paths
    data_dir = Path("data/raw/kaggle")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    X, y, feature_names = load_kaggle_data(data_dir / "data.csv")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create graph data
    train_data = create_graph_data(X_train, y_train)
    test_data = create_graph_data(X_test, y_test)
    
    # Create a single large graph for node classification
    large_graph = create_large_graph(X, y, feature_names, k_neighbors=5)
    
    # Save processed data
    torch.save(train_data, output_dir / "train_data.pt")
    torch.save(test_data, output_dir / "test_data.pt")
    torch.save(large_graph, output_dir / "large_graph.pt")
    
    logger.info(f"Processed data saved to {output_dir}")
    logger.info(f"Training samples: {len(y_train)}")
    logger.info(f"Test samples: {len(y_test)}")
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    logger.info(f"Graph structure: {len(large_graph.x)} nodes, {large_graph.edge_index.size(1)} edges")

if __name__ == "__main__":
    main() 