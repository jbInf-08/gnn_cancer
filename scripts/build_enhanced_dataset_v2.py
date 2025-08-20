import os
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDatasetBuilderV2:
    def __init__(self, expression_file="data/processed/expression_matrix_patients.csv",
                 cnv_file="data/processed/cnv_matrix_patients.csv",
                 output_dir="data/enhanced"):
        self.expression_file = expression_file
        self.cnv_file = cnv_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.expression_matrix = None
        self.cnv_matrix = None
        self.combined_features = None
        self.graph = None
        self.torch_data = None
        
    def load_data(self):
        """Load expression and CNV matrices."""
        logger.info("Loading expression and CNV matrices...")
        
        # Load matrices
        self.expression_matrix = pd.read_csv(self.expression_file, index_col=0)
        self.cnv_matrix = pd.read_csv(self.cnv_file, index_col=0)
        
        logger.info(f"Expression matrix: {self.expression_matrix.shape}")
        logger.info(f"CNV matrix: {self.cnv_matrix.shape}")
        
        # Ensure same samples
        common_samples = list(set(self.expression_matrix.columns) & set(self.cnv_matrix.columns))
        self.expression_matrix = self.expression_matrix[common_samples]
        self.cnv_matrix = self.cnv_matrix[common_samples]
        
        logger.info(f"Common samples: {len(common_samples)}")
        
    def preprocess_features(self, feature_selection='variance', n_genes=1000, n_segments=500):
        """Preprocess and select features."""
        logger.info("Preprocessing features...")
        
        # Transpose to get samples as rows
        expr_features = self.expression_matrix.T  # samples x genes
        cnv_features = self.cnv_matrix.T  # samples x segments
        
        # Feature selection for expression
        if feature_selection == 'variance':
            # Select top genes by variance
            gene_variances = expr_features.var()
            top_genes = gene_variances.nlargest(n_genes).index
            expr_selected = expr_features[top_genes]
        else:
            expr_selected = expr_features
            
        # Feature selection for CNV
        if feature_selection == 'variance':
            # Select top segments by variance
            segment_variances = cnv_features.var()
            top_segments = segment_variances.nlargest(n_segments).index
            cnv_selected = cnv_features[top_segments]
        else:
            cnv_selected = cnv_features
        
        # Combine features
        self.combined_features = pd.concat([expr_selected, cnv_selected], axis=1)
        
        # Fill missing values
        self.combined_features = self.combined_features.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        self.combined_features_scaled = pd.DataFrame(
            scaler.fit_transform(self.combined_features),
            index=self.combined_features.index,
            columns=self.combined_features.columns
        )
        
        logger.info(f"Combined features shape: {self.combined_features.shape}")
        logger.info(f"Expression features: {expr_selected.shape[1]}")
        logger.info(f"CNV features: {cnv_selected.shape[1]}")
        
        # Save scaler
        with open(self.output_dir / "feature_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
            
        return scaler
    
    def build_knn_graph(self, k=3):
        """Build k-nearest neighbors graph."""
        logger.info(f"Building k-nearest neighbors graph (k={k})...")
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(self.combined_features_scaled)
        
        # Use k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(self.combined_features_scaled)
        distances, indices = nbrs.kneighbors(self.combined_features_scaled)
        
        # Create adjacency matrix
        n_samples = len(self.combined_features_scaled)
        adjacency_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # Skip the first neighbor (self)
            for j in indices[i][1:]:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Make it undirected
        
        # Create NetworkX graph
        self.graph = nx.from_numpy_array(adjacency_matrix)
        
        # Add node attributes (patient IDs)
        patient_ids = list(self.combined_features_scaled.index)
        for i, patient_id in enumerate(patient_ids):
            self.graph.nodes[i]['patient_id'] = patient_id
        
        logger.info(f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        logger.info(f"Average degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}")
        
        # Save graph
        with open(self.output_dir / "patient_graph.pkl", 'wb') as f:
            pickle.dump(self.graph, f)
            
        return self.graph
    
    def create_torch_geometric_data(self, labels=None):
        """Create PyTorch Geometric Data object."""
        logger.info("Creating PyTorch Geometric Data object...")
        
        # Node features
        node_features = torch.FloatTensor(self.combined_features_scaled.values)
        
        # Edge indices
        edge_list = list(self.graph.edges())
        if len(edge_list) == 0:
            # If no edges, create a fully connected graph for demonstration
            logger.warning("No edges found, creating fully connected graph for demonstration")
            n_nodes = len(self.combined_features_scaled)
            edge_list = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        
        # Create labels (placeholder - you can add real clinical labels here)
        if labels is None:
            # Create synthetic labels for demonstration
            # In practice, you would load real clinical data
            n_samples = len(self.combined_features_scaled)
            labels = np.random.randint(0, 2, n_samples)  # Binary classification
            logger.warning("Using synthetic labels. Replace with real clinical data!")
        
        node_labels = torch.LongTensor(labels)
        
        # Create PyTorch Geometric Data object
        self.torch_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=node_labels
        )
        
        logger.info(f"PyTorch Geometric Data created:")
        logger.info(f"  - Node features: {self.torch_data.x.shape}")
        logger.info(f"  - Edge index: {self.torch_data.edge_index.shape}")
        logger.info(f"  - Labels: {self.torch_data.y.shape}")
        
        # Save PyTorch data
        torch.save(self.torch_data, self.output_dir / "torch_geometric_data.pt")
        
        return self.torch_data
    
    def visualize_graph(self):
        """Visualize the patient similarity graph."""
        logger.info("Creating graph visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Use spring layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color='lightblue', 
                             node_size=300, 
                             alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, 
                             alpha=0.3, 
                             edge_color='gray')
        
        # Add labels
        patient_ids = [self.graph.nodes[i]['patient_id'] for i in self.graph.nodes()]
        labels = {i: patient_ids[i][:8] for i in self.graph.nodes()}  # Truncate for readability
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Patient Similarity Graph (KNN)", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / "patient_similarity_graph.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graph visualization saved to {self.output_dir / 'patient_similarity_graph.png'}")
    
    def create_dataset_summary(self):
        """Create a summary of the enhanced dataset."""
        logger.info("Creating dataset summary...")
        
        summary = {
            'n_patients': len(self.combined_features_scaled),
            'n_expression_features': self.expression_matrix.shape[0],
            'n_cnv_features': self.cnv_matrix.shape[0],
            'n_combined_features': self.combined_features.shape[1],
            'n_graph_nodes': self.graph.number_of_nodes(),
            'n_graph_edges': self.graph.number_of_edges(),
            'graph_density': nx.density(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'patient_ids': list(self.combined_features_scaled.index),
            'expression_genes': list(self.expression_matrix.index),
            'cnv_segments': list(self.cnv_matrix.index)
        }
        
        # Save summary
        with open(self.output_dir / "dataset_summary.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        # Print summary
        logger.info("Dataset Summary:")
        logger.info(f"  - Patients: {summary['n_patients']}")
        logger.info(f"  - Expression features: {summary['n_expression_features']}")
        logger.info(f"  - CNV features: {summary['n_cnv_features']}")
        logger.info(f"  - Combined features: {summary['n_combined_features']}")
        logger.info(f"  - Graph nodes: {summary['n_graph_nodes']}")
        logger.info(f"  - Graph edges: {summary['n_graph_edges']}")
        logger.info(f"  - Graph density: {summary['graph_density']:.3f}")
        logger.info(f"  - Average degree: {summary['average_degree']:.2f}")
        
        return summary
    
    def build_dataset(self, feature_selection='variance', n_genes=1000, n_segments=500, k=3):
        """Build the complete enhanced dataset."""
        logger.info("Building enhanced dataset...")
        
        # Load data
        self.load_data()
        
        # Preprocess features
        scaler = self.preprocess_features(feature_selection, n_genes, n_segments)
        
        # Build k-nearest neighbors graph
        graph = self.build_knn_graph(k)
        
        # Create PyTorch Geometric data
        torch_data = self.create_torch_geometric_data()
        
        # Visualize graph
        self.visualize_graph()
        
        # Create summary
        summary = self.create_dataset_summary()
        
        logger.info("Enhanced dataset built successfully!")
        logger.info(f"Output directory: {self.output_dir}")
        
        return {
            'expression_matrix': self.expression_matrix,
            'cnv_matrix': self.cnv_matrix,
            'combined_features': self.combined_features,
            'combined_features_scaled': self.combined_features_scaled,
            'graph': self.graph,
            'torch_data': self.torch_data,
            'scaler': scaler,
            'summary': summary
        }

def main():
    """Main function to build the enhanced dataset."""
    logger.info("Starting enhanced dataset construction (v2)...")
    
    # Create dataset builder
    builder = EnhancedDatasetBuilderV2()
    
    # Build dataset
    dataset = builder.build_dataset(
        feature_selection='variance',
        n_genes=1000,  # Top 1000 genes by variance
        n_segments=500,  # Top 500 CNV segments by variance
        k=3  # 3-nearest neighbors
    )
    
    logger.info("Enhanced dataset construction complete!")
    logger.info("Files created:")
    logger.info("  - data/enhanced/feature_scaler.pkl")
    logger.info("  - data/enhanced/patient_graph.pkl")
    logger.info("  - data/enhanced/torch_geometric_data.pt")
    logger.info("  - data/enhanced/patient_similarity_graph.png")
    logger.info("  - data/enhanced/dataset_summary.pkl")

if __name__ == "__main__":
    main() 