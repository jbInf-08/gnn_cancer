import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(data_dir, cancer_type, data_source):
    """
    Load real breast cancer data from Kaggle and convert to graph structure.
    
    Args:
        data_dir: Directory containing the data
        cancer_type: Type of cancer (BRCA, etc.)
        data_source: Data source (kaggle, etc.)
    
    Returns:
        Data object with node features, edges, and labels
    """
    print(f"[INFO] Loading {cancer_type} data from {data_source}")
    
    # Load the Kaggle breast cancer dataset
    data_path = os.path.join(data_dir, 'raw', 'kaggle', 'breast-cancer.csv')
    df = pd.read_csv(data_path)
    
    print(f"[INFO] Loaded {len(df)} samples with {len(df.columns)-2} features")
    
    # Separate features and labels
    # Remove 'id' and 'diagnosis' columns, keep all other features
    feature_columns = [col for col in df.columns if col not in ['id', 'diagnosis']]
    X = df[feature_columns].values
    y = (df['diagnosis'] == 'M').astype(int)  # Convert M=1 (malignant), B=0 (benign)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(X_scaled)
    y = torch.LongTensor(y.values)
    
    # Create a fully connected graph (each node connects to all others)
    # This simulates a graph where each sample can influence every other sample
    num_nodes = len(df)
    edge_index = []
    
    # Create edges between all pairs of nodes (fully connected graph)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # No self-loops
                edge_index.append([i, j])
    
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    
    # Create train/val/test masks
    idx = np.arange(num_nodes)
    idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y, test_size=0.3, stratify=y, random_state=42)
    idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"[INFO] Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
    print(f"[INFO] Node features: {data.num_node_features}, Classes: {len(torch.unique(data.y))}")
    print(f"[INFO] Class distribution: {torch.bincount(data.y).tolist()}")
    print(f"[INFO] Train/Val/Test split: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
    
    return data 