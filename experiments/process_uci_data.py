import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_uci_data(data_dir: Path):
    """Load UCI breast cancer dataset."""
    logger.info("Loading UCI breast cancer dataset...")
    
    # Read the data file
    data_file = data_dir / "wdbc.data"
    
    # UCI feature names from documentation
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    # Read the data
    df = pd.read_csv(data_file, header=None, names=['id', 'diagnosis'] + feature_names)
    
    # Convert diagnosis to binary (M=1, B=0)
    df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
    logger.info(f"Unique values in diagnosis after conversion: {df['diagnosis'].unique()}")
    
    # Separate features and labels
    X = df.drop(['id', 'diagnosis'], axis=1).values
    y = df['diagnosis'].values
    logger.info(f"First 10 values of y: {y[:10]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names

def create_graph_data(X, y, k=5):
    """Create graph data from features using k-nearest neighbors."""
    logger.info(f"Creating graph with {k} nearest neighbors...")
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Create edge index
    edge_index = []
    edge_attr = []
    
    for i in range(len(X)):
        # Skip self-loops
        for j, idx in enumerate(indices[i][1:], 1):
            edge_index.append([i, idx])
            # Edge features: distance and feature similarity
            dist = distances[i][j]
            similarity = np.exp(-dist)  # Convert distance to similarity
            edge_attr.append([dist, similarity])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create node features
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    # Create graph data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data

def main():
    # Set up paths
    data_dir = Path("data/raw/uci/breast_cancer_wisconsin")
    processed_dir = Path("data/processed/uci")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    X, y, feature_names = load_uci_data(data_dir)
    
    # Create a single large graph for node classification
    data = create_graph_data(X, y)
    
    # Create train/test masks
    n_nodes = data.x.size(0)
    indices = torch.randperm(n_nodes)
    train_size = int(0.8 * n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    # Save processed data
    torch.save(data, processed_dir / "large_graph.pt")
    torch.save(train_mask, processed_dir / "train_mask.pt")
    torch.save(test_mask, processed_dir / "test_mask.pt")
    
    # Log dataset statistics
    logger.info(f"Dataset processed and saved to {processed_dir}")
    logger.info(f"Total nodes: {n_nodes}")
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    logger.info(f"Graph structure: {data.num_nodes} nodes, {data.num_edges} edges")

if __name__ == "__main__":
    main() 