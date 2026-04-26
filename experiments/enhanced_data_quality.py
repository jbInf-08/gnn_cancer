"""
Enhanced Data Quality Improvements to Surpass Paper Performance
- Feature normalization and selection
- Graph sparsification and augmentation
- Advanced data preprocessing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataQualityImprover:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        
    def load_data(self):
        """Load the data"""
        try:
            data_file = Path(self.data_path) / "enhanced" / "real_only_torch_geometric_data.pt"
            if data_file.exists():
                self.data = torch.load(data_file, weights_only=False)
                logger.info(f"Loaded data from {data_file}")
                logger.info(f"Data shape: {self.data.x.shape}, Edges: {self.data.edge_index.shape[1]}")
            else:
                raise FileNotFoundError(f"Data file not found: {data_file}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self):
        """Clean the data by removing NaN values and outliers"""
        logger.info("Cleaning data...")
        
        # Remove NaN values
        if torch.isnan(self.data.x).any():
            logger.info("Removing NaN values from node features...")
            self.data.x = torch.nan_to_num(self.data.x, nan=0.0)
        
        if torch.isnan(self.data.edge_attr).any():
            logger.info("Removing NaN values from edge attributes...")
            self.data.edge_attr = torch.nan_to_num(self.data.edge_attr, nan=0.0)
        
        # Remove outliers using IQR method
        logger.info("Removing outliers from node features...")
        x_np = self.data.x.numpy()
        for i in range(x_np.shape[1]):
            Q1 = np.percentile(x_np[:, i], 25)
            Q3 = np.percentile(x_np[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            x_np[:, i] = np.clip(x_np[:, i], lower_bound, upper_bound)
        
        self.data.x = torch.from_numpy(x_np).float()
        logger.info("Data cleaning completed")
    
    def normalize_features(self, method='standard'):
        """Normalize node features"""
        logger.info(f"Normalizing features using {method} scaling...")
        
        x_np = self.data.x.numpy()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        x_normalized = self.scaler.fit_transform(x_np)
        self.data.x = torch.from_numpy(x_normalized).float()
        
        logger.info(f"Feature normalization completed using {method} scaling")
    
    def select_features(self, method='mutual_info', k=None):
        """Select the most important features"""
        logger.info(f"Selecting features using {method}...")
        
        if k is None:
            k = min(10, self.data.x.shape[1])  # Select top 10 or all if less than 10
        
        x_np = self.data.x.numpy()
        
        if hasattr(self.data, 'y') and self.data.y is not None:
            y_np = self.data.y.numpy()
        else:
            logger.warning("No labels found, skipping feature selection")
            return
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        x_selected = self.feature_selector.fit_transform(x_np, y_np)
        self.data.x = torch.from_numpy(x_selected).float()
        
        # Get feature scores
        feature_scores = self.feature_selector.scores_
        selected_features = self.feature_selector.get_support()
        
        logger.info(f"Feature selection completed. Selected {k} features out of {len(feature_scores)}")
        logger.info(f"Top feature scores: {feature_scores[selected_features][:5]}")
    
    def apply_pca(self, n_components=None, explained_variance=0.95):
        """Apply PCA for dimensionality reduction"""
        logger.info("Applying PCA for dimensionality reduction...")
        
        x_np = self.data.x.numpy()
        
        if n_components is None:
            # Find number of components that explain 95% of variance
            pca_temp = PCA()
            pca_temp.fit(x_np)
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= explained_variance) + 1
        
        self.pca = PCA(n_components=n_components)
        x_pca = self.pca.fit_transform(x_np)
        self.data.x = torch.from_numpy(x_pca).float()
        
        explained_var_ratio = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA completed. Reduced from {x_np.shape[1]} to {n_components} features")
        logger.info(f"Explained variance ratio: {explained_var_ratio:.4f}")
    
    def optimize_graph_structure(self, sparsification_threshold=0.1):
        """Optimize graph structure by removing noisy edges"""
        logger.info("Optimizing graph structure...")
        
        # Convert to NetworkX for easier manipulation
        edge_index = self.data.edge_index.numpy()
        edge_attr = self.data.edge_attr.numpy() if self.data.edge_attr is not None else None
        
        G = nx.Graph()
        
        # Add edges with weights
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            weight = edge_attr[i, 0] if edge_attr is not None else 1.0
            G.add_edge(u, v, weight=weight)
        
        # Remove edges with low weights (sparsification)
        if edge_attr is not None:
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            threshold = np.percentile(weights, sparsification_threshold * 100)
            
            edges_to_remove = [(u, v) for u, v in G.edges() if G[u][v]['weight'] < threshold]
            G.remove_edges_from(edges_to_remove)
            
            logger.info(f"Removed {len(edges_to_remove)} edges with weight < {threshold:.4f}")
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        logger.info(f"Removed {len(isolated_nodes)} isolated nodes")
        
        # Convert back to PyTorch Geometric format
        edge_list = list(G.edges())
        if edge_list:
            new_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            new_edge_attr = torch.tensor([G[u][v]['weight'] for u, v in edge_list], dtype=torch.float).unsqueeze(1)
            
            self.data.edge_index = new_edge_index
            self.data.edge_attr = new_edge_attr
        else:
            logger.warning("No edges remaining after optimization!")
        
        logger.info(f"Graph optimization completed. New edge count: {self.data.edge_index.shape[1]}")
    
    def augment_graph(self, augmentation_factor=0.1):
        """Augment graph by adding synthetic edges"""
        logger.info("Augmenting graph with synthetic edges...")
        
        num_nodes = self.data.x.shape[0]
        num_new_edges = int(self.data.edge_index.shape[1] * augmentation_factor)
        
        # Skip augmentation for very large graphs to avoid memory issues
        if num_nodes > 100000:
            logger.warning(f"Graph too large ({num_nodes} nodes), skipping augmentation to avoid memory issues")
            return
        
        # Create synthetic edges based on feature similarity
        x_np = self.data.x.numpy()
        
        # Use a more memory-efficient approach for large graphs
        if num_nodes > 10000:
            # Sample a subset of nodes for similarity computation
            sample_size = min(10000, num_nodes)
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            x_sample = x_np[sample_indices]
            
            # Compute similarities only for sampled nodes
            x_normalized = x_sample / np.linalg.norm(x_sample, axis=1, keepdims=True)
            similarities = np.dot(x_normalized, x_normalized.T)
            
            logger.info(f"Using sampled similarity computation for {sample_size} nodes")
        else:
            # Compute pairwise cosine similarities for smaller graphs
            x_normalized = x_np / np.linalg.norm(x_np, axis=1, keepdims=True)
            similarities = np.dot(x_normalized, x_normalized.T)
        
        # Find pairs with high similarity that are not already connected
        existing_edges = set()
        for i in range(self.data.edge_index.shape[1]):
            u, v = self.data.edge_index[0, i].item(), self.data.edge_index[1, i].item()
            existing_edges.add((min(u, v), max(u, v)))
        
        # Find potential new edges
        potential_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (i, j) not in existing_edges and similarities[i, j] > 0.8:
                    potential_edges.append((i, j, similarities[i, j]))
        
        # Sort by similarity and take top ones
        potential_edges.sort(key=lambda x: x[2], reverse=True)
        new_edges = potential_edges[:num_new_edges]
        
        if new_edges:
            # Add new edges
            new_edge_index = torch.tensor([[u, v] for u, v, _ in new_edges], dtype=torch.long).t().contiguous()
            new_edge_attr = torch.tensor([[w] for _, _, w in new_edges], dtype=torch.float)
            
            # Combine with existing edges
            combined_edge_index = torch.cat([self.data.edge_index, new_edge_index], dim=1)
            combined_edge_attr = torch.cat([self.data.edge_attr, new_edge_attr], dim=0)
            
            self.data.edge_index = combined_edge_index
            self.data.edge_attr = combined_edge_attr
            
            logger.info(f"Added {len(new_edges)} synthetic edges")
        else:
            logger.info("No suitable synthetic edges found")
    
    def analyze_data_quality(self):
        """Analyze and report data quality metrics"""
        logger.info("Analyzing data quality...")
        
        # Node feature analysis
        x_np = self.data.x.numpy()
        logger.info(f"Node features:")
        logger.info(f"  Shape: {x_np.shape}")
        logger.info(f"  Mean: {x_np.mean():.4f}")
        logger.info(f"  Std: {x_np.std():.4f}")
        logger.info(f"  Min: {x_np.min():.4f}")
        logger.info(f"  Max: {x_np.max():.4f}")
        
        # Graph structure analysis
        num_nodes = self.data.x.shape[0]
        num_edges = self.data.edge_index.shape[1]
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        logger.info(f"Graph structure:")
        logger.info(f"  Nodes: {num_nodes}")
        logger.info(f"  Edges: {num_edges}")
        logger.info(f"  Density: {density:.6f}")
        
        # Edge attribute analysis
        if self.data.edge_attr is not None:
            edge_attr_np = self.data.edge_attr.numpy()
            logger.info(f"Edge attributes:")
            logger.info(f"  Shape: {edge_attr_np.shape}")
            logger.info(f"  Mean: {edge_attr_np.mean():.4f}")
            logger.info(f"  Std: {edge_attr_np.std():.4f}")
        
        # Label analysis
        if hasattr(self.data, 'y') and self.data.y is not None:
            y_np = self.data.y.numpy()
            unique_labels, counts = np.unique(y_np, return_counts=True)
            logger.info(f"Labels:")
            for label, count in zip(unique_labels, counts):
                logger.info(f"  Label {label}: {count} samples ({count/len(y_np)*100:.1f}%)")
    
    def save_enhanced_data(self, output_path="data/enhanced"):
        """Save the enhanced data"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        enhanced_file = output_dir / "enhanced_optimized_torch_geometric_data.pt"
        torch.save(self.data, enhanced_file)
        
        logger.info(f"Enhanced data saved to {enhanced_file}")
        
        # Save preprocessing information
        preprocessing_info = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'data_shape': self.data.x.shape,
            'edge_count': self.data.edge_index.shape[1]
        }
        
        info_file = output_dir / "preprocessing_info.pkl"
        import pickle
        with open(info_file, 'wb') as f:
            pickle.dump(preprocessing_info, f)
        
        logger.info(f"Preprocessing information saved to {info_file}")
    
    def run_full_enhancement(self):
        """Run the complete data enhancement pipeline"""
        logger.info("Starting complete data enhancement pipeline...")
        
        # Load data
        self.load_data()
        
        # Analyze original data
        logger.info("=== Original Data Analysis ===")
        self.analyze_data_quality()
        
        # Clean data
        self.clean_data()
        
        # Normalize features
        self.normalize_features(method='standard')
        
        # Select features (if labels are available)
        if hasattr(self.data, 'y') and self.data.y is not None:
            self.select_features(method='mutual_info', k=10)
        
        # Apply PCA for dimensionality reduction
        self.apply_pca(explained_variance=0.95)
        
        # Optimize graph structure
        self.optimize_graph_structure(sparsification_threshold=0.1)
        
        # Augment graph
        self.augment_graph(augmentation_factor=0.1)
        
        # Analyze enhanced data
        logger.info("=== Enhanced Data Analysis ===")
        self.analyze_data_quality()
        
        # Save enhanced data
        self.save_enhanced_data()
        
        logger.info("Data enhancement pipeline completed!")
        return self.data

def main():
    """Main function to run data enhancement"""
    enhancer = EnhancedDataQualityImprover("data")
    enhanced_data = enhancer.run_full_enhancement()
    
    print("Data enhancement completed successfully!")
    print(f"Final data shape: {enhanced_data.x.shape}")
    print(f"Final edge count: {enhanced_data.edge_index.shape[1]}")

if __name__ == "__main__":
    main()
