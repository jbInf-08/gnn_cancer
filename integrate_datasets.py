import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_geometric.data import HeteroData
import networkx as nx
from typing import Dict, List, Tuple, Optional
import requests
import json
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetIntegrator:
    def __init__(self, data_dir: str = "data"):
        """Initialize the dataset integrator."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.tcga_data = None
        self.ncbi_data = None
        self.uci_data = None
        self.kaggle_data = None
        self.integrated_graph = None
        
        # Initialize preprocessing tools
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def load_tcga_data(self) -> None:
        """Load and preprocess TCGA data."""
        logger.info("Loading TCGA data...")
        
        # Load expression data (single file)
        expr_file = self.data_dir / "raw/BRCA_expression.tsv.gz"
        if not expr_file.exists():
            raise FileNotFoundError(f"Expression file not found: {expr_file}")
        
        expr_data = pd.read_csv(expr_file, sep='\t', compression=None, comment='#')

        # Load mutation data (multiple files)
        mutation_dir = self.data_dir / "raw/tcga"
        mutation_files = list(mutation_dir.glob("BRCA_mutation_*.maf"))
        if not mutation_files:
            raise FileNotFoundError("No TCGA mutation files found in data/raw/tcga/")
        
        mutation_data_list = []
        for f in mutation_files:
            try:
                # Read mutation file with gzip compression and skip comments
                mutation_data_list.append(pd.read_csv(f, sep='\t', compression='gzip', comment='#'))
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")
                continue
        
        if not mutation_data_list:
            raise ValueError("No mutation data could be loaded")
        
        mutation_data = pd.concat(mutation_data_list, ignore_index=True)

        # For now, create empty clinical data (we can add this later if needed)
        clinical_data = pd.DataFrame()

        # Process and integrate TCGA data
        self.tcga_data = {
            'expression': expr_data,
            'clinical': clinical_data,
            'mutation': mutation_data
        }

        logger.info(f"Loaded TCGA data with {len(expr_data)} expression samples, {len(clinical_data)} clinical samples, {len(mutation_data)} mutation records")
        
    def load_ncbi_data(self) -> None:
        """Load and preprocess NCBI data."""
        logger.info("Loading NCBI data...")
        ncbi_dir = self.data_dir / "raw/ncbi"
        
        # Define essential columns for gene_info
        essential_columns = ['GeneID', 'Symbol', 'description', 'type_of_gene']
        
        # Load gene information in chunks with memory optimization
        chunk_size = 100000  # Adjust based on available memory
        gene_info_chunks = []
        
        for chunk in pd.read_csv(
            ncbi_dir / "gene_info.gz",
            compression='gzip',
            sep='\t',
            usecols=essential_columns,
            chunksize=chunk_size,
            low_memory=True,
            dtype={
                'GeneID': 'int32',
                'Symbol': 'str',
                'description': 'str',
                'type_of_gene': 'str'
            }
        ):
            # Filter out non-protein coding genes to reduce data size
            chunk = chunk[chunk['type_of_gene'] == 'protein-coding']
            gene_info_chunks.append(chunk)
        
        gene_info = pd.concat(gene_info_chunks, ignore_index=True)
        
        # Load GO term associations with memory optimization
        go_terms = pd.read_csv(
            ncbi_dir / "gene2go.gz",
            compression='gzip',
            sep='\t',
            usecols=['GeneID', 'GO_ID', 'Category'],
            dtype={
                'GeneID': 'int32',
                'GO_ID': 'str',
                'Category': 'str'
            },
            low_memory=True
        )
        
        # Filter GO terms to only include genes we have in gene_info
        go_terms = go_terms[go_terms['GeneID'].isin(gene_info['GeneID'])]
        
        # Load PubMed citations with memory optimization
        pubmed = pd.read_csv(
            ncbi_dir / "gene2pubmed.gz",
            compression='gzip',
            sep='\t',
            usecols=['GeneID', 'PubMed_ID'],
            dtype={
                'GeneID': 'int32',
                'PubMed_ID': 'int32'
            },
            low_memory=True
        )
        
        # Filter PubMed citations to only include genes we have in gene_info
        pubmed = pubmed[pubmed['GeneID'].isin(gene_info['GeneID'])]
        
        # Process and integrate NCBI data
        self.ncbi_data = {
            'gene_info': gene_info,
            'go_terms': go_terms,
            'pubmed': pubmed
        }
        
        logger.info(f"Loaded NCBI data with {len(gene_info)} genes, {len(go_terms)} GO terms, and {len(pubmed)} PubMed citations")
        
    def load_uci_data(self) -> None:
        """Load and preprocess UCI data."""
        logger.info("Loading UCI data...")
        uci_dir = self.data_dir / "raw/uci"
        
        # Load UCI breast cancer dataset
        uci_data = pd.read_csv(uci_dir / "breast-cancer-wisconsin.data", 
                             names=['id', 'clump_thickness', 'uniformity_cell_size',
                                   'uniformity_cell_shape', 'marginal_adhesion',
                                   'single_epithelial_cell_size', 'bare_nuclei',
                                   'bland_chromatin', 'normal_nucleoli', 'mitoses',
                                   'class'])
        
        # Process UCI data
        self.uci_data = uci_data
        
        logger.info(f"Loaded UCI data with {len(uci_data)} samples")
        
    def load_kaggle_data(self) -> None:
        """Load and preprocess Kaggle data."""
        logger.info("Loading Kaggle data...")
        kaggle_dir = self.data_dir / "raw/kaggle"
        
        # Load Kaggle breast cancer dataset
        kaggle_data = pd.read_csv(kaggle_dir / "breast-cancer.csv")
        
        # Process Kaggle data
        self.kaggle_data = kaggle_data
        
        logger.info(f"Loaded Kaggle data with {len(kaggle_data)} samples")
        
    def create_heterogeneous_graph(self) -> None:
        """Create a heterogeneous graph integrating all datasets."""
        logger.info("Creating integrated heterogeneous graph...")
        
        # Initialize HeteroData object
        data = HeteroData()
        
        # Add gene nodes (from TCGA and NCBI)
        gene_features = self._process_gene_features()
        data['gene'].x = torch.tensor(gene_features, dtype=torch.float)
        
        # Add GO term nodes (from NCBI)
        go_features = self._process_go_features()
        data['go'].x = torch.tensor(go_features, dtype=torch.float)
        
        # Add patient nodes (from TCGA, UCI, and Kaggle)
        patient_features = self._process_patient_features()
        data['patient'].x = torch.tensor(patient_features, dtype=torch.float)
        
        # Add PubMed nodes (from NCBI)
        pubmed_features = self._process_pubmed_features()
        data['pubmed'].x = torch.tensor(pubmed_features, dtype=torch.float)
        
        # Add edges
        self._add_edges(data)
        
        # Save the integrated graph
        self.integrated_graph = data
        torch.save(data, self.data_dir / "processed/integrated_graph.pt")
        
        logger.info("Created and saved integrated heterogeneous graph")
        
    def _process_gene_features(self) -> np.ndarray:
        """Process and combine gene features from TCGA and NCBI."""
        # Get TCGA gene features
        tcga_gene_features = self._combine_tcga_gene_features()
        
        # Get NCBI gene features
        ncbi_gene_features = self._process_ncbi_gene_features()
        
        # For now, just return TCGA features to avoid dimension mismatch
        # In a full implementation, we would need to align gene sets properly
        logger.info(f"Using TCGA gene features with shape: {tcga_gene_features.shape}")
        return tcga_gene_features
        
    def _process_go_features(self) -> np.ndarray:
        """Process GO term features from NCBI."""
        go_terms = self.ncbi_data['go_terms']
        
        # Create one-hot encoding of GO categories
        categories = go_terms['Category'].unique()
        unique_go_ids = go_terms['GO_ID'].unique()
        
        go_features = np.zeros((len(unique_go_ids), len(categories)))
        
        # Create mappings
        go_id_to_idx = {go_id: idx for idx, go_id in enumerate(unique_go_ids)}
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        
        # Fill the feature matrix using groupby to avoid indexing issues
        for go_id, group in go_terms.groupby('GO_ID'):
            if go_id in go_id_to_idx:
                go_idx = go_id_to_idx[go_id]
                for category in group['Category'].unique():
                    if category in category_to_idx:
                        cat_idx = category_to_idx[category]
                        go_features[go_idx, cat_idx] = 1
            
        return go_features
        
    def _process_patient_features(self) -> np.ndarray:
        """Process and combine patient features from TCGA, UCI, and Kaggle."""
        # Process TCGA clinical data
        tcga_patient_features = self._process_tcga_patient_features()
        
        # Process UCI data
        uci_patient_features = self._process_uci_patient_features()
        
        # Process Kaggle data
        kaggle_patient_features = self._process_kaggle_patient_features()
        
        # Find the maximum number of columns
        n_cols = max(tcga_patient_features.shape[1], uci_patient_features.shape[1], kaggle_patient_features.shape[1])
        
        # Pad arrays to have the same number of columns
        def pad_features(arr, n_cols):
            if arr.shape[1] < n_cols:
                pad_width = n_cols - arr.shape[1]
                return np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')
            return arr
        tcga_patient_features = pad_features(tcga_patient_features, n_cols)
        uci_patient_features = pad_features(uci_patient_features, n_cols)
        kaggle_patient_features = pad_features(kaggle_patient_features, n_cols)
        
        # Vertically stack all patient features (each patient is a row)
        return np.vstack([tcga_patient_features, uci_patient_features, kaggle_patient_features])
        
    def _process_pubmed_features(self) -> np.ndarray:
        """Process PubMed features from NCBI."""
        pubmed_data = self.ncbi_data['pubmed']
        
        # Create features based on PubMed ID only
        pubmed_features = np.zeros((len(pubmed_data['PubMed_ID'].unique()), 1))
        
        for idx, pubmed_id in enumerate(pubmed_data['PubMed_ID'].unique()):
            # Use a simple feature based on PubMed ID
            pubmed_features[idx, 0] = pubmed_id % 1000  # Use last 3 digits as a feature
            
        return pubmed_features
        
    def _add_edges(self, data: HeteroData) -> None:
        """Add edges to the heterogeneous graph."""
        # Add gene-GO edges
        gene_go_edges = self._create_gene_go_edges()
        data['gene', 'associated_with', 'go'].edge_index = torch.tensor(gene_go_edges, dtype=torch.long).t()
        
        # Add gene-patient edges
        gene_patient_edges = self._create_gene_patient_edges()
        data['gene', 'expressed_in', 'patient'].edge_index = torch.tensor(gene_patient_edges, dtype=torch.long).t()
        
        # Add gene-pubmed edges
        gene_pubmed_edges = self._create_gene_pubmed_edges()
        data['gene', 'cited_in', 'pubmed'].edge_index = torch.tensor(gene_pubmed_edges, dtype=torch.long).t()
        
    def validate_integration(self) -> Dict[str, float]:
        """Validate the integrated dataset."""
        logger.info("Validating integrated dataset...")
        
        validation_metrics = {
            'node_counts': {
                'gene': self.integrated_graph['gene'].x.size(0),
                'go': self.integrated_graph['go'].x.size(0),
                'patient': self.integrated_graph['patient'].x.size(0),
                'pubmed': self.integrated_graph['pubmed'].x.size(0)
            },
            'edge_counts': {
                'gene-go': self.integrated_graph['gene', 'associated_with', 'go'].edge_index.size(1),
                'gene-patient': self.integrated_graph['gene', 'expressed_in', 'patient'].edge_index.size(1),
                'gene-pubmed': self.integrated_graph['gene', 'cited_in', 'pubmed'].edge_index.size(1)
            }
        }
        
        logger.info("Validation metrics:")
        for category, metrics in validation_metrics.items():
            logger.info(f"{category}: {metrics}")
            
        return validation_metrics
        
    def save_integrated_data(self) -> None:
        """Save the integrated dataset and metadata."""
        output_dir = self.data_dir / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the integrated graph
        torch.save(self.integrated_graph, output_dir / "integrated_graph.pt")
        
        # Save metadata
        metadata = {
            'data_sources': {
                'tcga': 'TCGA Breast Cancer Data',
                'ncbi': 'NCBI Gene and Literature Data',
                'uci': 'UCI Breast Cancer Dataset',
                'kaggle': 'Kaggle Breast Cancer Dataset'
            },
            'node_types': ['gene', 'go', 'patient', 'pubmed'],
            'edge_types': [
                ('gene', 'associated_with', 'go'),
                ('gene', 'expressed_in', 'patient'),
                ('gene', 'cited_in', 'pubmed')
            ],
            'feature_dimensions': {
                'gene': self.integrated_graph['gene'].x.size(1),
                'go': self.integrated_graph['go'].x.size(1),
                'patient': self.integrated_graph['patient'].x.size(1),
                'pubmed': self.integrated_graph['pubmed'].x.size(1)
            }
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved integrated data to {output_dir}")

    def _combine_tcga_gene_features(self) -> np.ndarray:
        """Combine gene features from TCGA expression and mutation data."""
        if not self.tcga_data:
            return np.array([])
            
        # Get expression data
        expr_data = self.tcga_data['expression']
        
        # Get mutation data
        mutation_data = self.tcga_data['mutation']
        
        # Create a feature matrix for genes
        # Use gene_id from expression and Hugo_Symbol from mutation
        expr_genes = set(expr_data['gene_id'].unique())
        mutation_genes = set(mutation_data['Hugo_Symbol'].unique())
        unique_genes = list(expr_genes.union(mutation_genes))
        
        # Initialize feature matrix
        gene_features = np.zeros((len(unique_genes), 2))  # 2 features: expression mean and mutation count
        
        # Calculate mean expression for each gene (using tpm_unstranded as expression value)
        expr_means = expr_data.groupby('gene_id')['tpm_unstranded'].mean()
        
        # Calculate mutation count for each gene
        mutation_counts = mutation_data.groupby('Hugo_Symbol').size()
        
        # Fill feature matrix
        for idx, gene_id in enumerate(unique_genes):
            # Try to get expression mean (handle both gene_id and Hugo_Symbol)
            expr_mean = expr_means.get(gene_id, 0)
            if expr_mean == 0:
                # Try to find by gene_name if gene_id not found
                gene_name_match = expr_data[expr_data['gene_name'] == gene_id]['tpm_unstranded'].mean()
                if not pd.isna(gene_name_match):
                    expr_mean = gene_name_match
            
            gene_features[idx, 0] = expr_mean  # Expression mean
            gene_features[idx, 1] = mutation_counts.get(gene_id, 0)  # Mutation count
            
        return gene_features

    def _process_ncbi_gene_features(self) -> np.ndarray:
        """Process NCBI gene features."""
        if not self.ncbi_data:
            return np.array([])
            
        gene_info = self.ncbi_data['gene_info']
        
        # Create simple features based on gene information
        # For now, create a simple feature matrix with gene ID and type encoding
        gene_features = np.zeros((len(gene_info), 2))
        
        # Feature 1: Gene ID (modulo to keep it manageable)
        gene_features[:, 0] = gene_info['GeneID'] % 10000
        
        # Feature 2: Gene type encoding (simple hash of type)
        gene_types = gene_info['type_of_gene'].fillna('unknown')
        type_encoding = {gene_type: idx for idx, gene_type in enumerate(gene_types.unique())}
        gene_features[:, 1] = gene_types.map(type_encoding)
        
        return gene_features

    def _process_tcga_patient_features(self) -> np.ndarray:
        """Extract basic numeric features from TCGA clinical data (if available)."""
        clinical_data = self.tcga_data.get('clinical', pd.DataFrame())
        if clinical_data.empty:
            # Return a dummy array if no clinical data
            return np.zeros((1, 1))
        # Select numeric columns only
        numeric_cols = clinical_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.zeros((len(clinical_data), 1))
        return clinical_data[numeric_cols].fillna(0).to_numpy()

    def _process_uci_patient_features(self) -> np.ndarray:
        """Extract basic numeric features from UCI data."""
        uci_data = self.uci_data
        if uci_data is None or uci_data.empty:
            return np.zeros((1, 1))
        # Drop non-numeric columns (like 'id' and 'class')
        numeric_cols = [col for col in uci_data.columns if col not in ['id', 'class']]
        return uci_data[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

    def _process_kaggle_patient_features(self) -> np.ndarray:
        """Extract basic numeric features from Kaggle data."""
        kaggle_data = self.kaggle_data
        if kaggle_data is None or kaggle_data.empty:
            return np.zeros((1, 1))
        # Drop non-numeric columns (like 'id' and 'diagnosis' if present)
        non_numeric = [col for col in kaggle_data.columns if kaggle_data[col].dtype == 'O']
        numeric_cols = [col for col in kaggle_data.columns if col not in non_numeric]
        if not numeric_cols:
            return np.zeros((len(kaggle_data), 1))
        return kaggle_data[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

    def _create_gene_go_edges(self):
        """Create edges between genes and GO terms based on NCBI gene2go associations."""
        # Map gene IDs to indices in the gene node list
        expr_data = self.tcga_data['expression']
        gene_ids = list(set(expr_data['gene_id'].unique()))
        gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
        go_terms = self.ncbi_data['go_terms']
        go_ids = list(go_terms['GO_ID'].unique())
        go_id_to_idx = {go_id: idx for idx, go_id in enumerate(go_ids)}
        
        # Only keep associations where both gene and GO term are present
        edges = []
        for _, row in go_terms.iterrows():
            gene_id = row['GeneID'] if 'GeneID' in row else row['gene_id']
            go_id = row['GO_ID']
            if gene_id in gene_id_to_idx and go_id in go_id_to_idx:
                edges.append([gene_id_to_idx[gene_id], go_id_to_idx[go_id]])
        if edges:
            return np.array(edges).T  # shape (2, num_edges)
        else:
            return np.zeros((2, 0), dtype=int)

    def _create_gene_patient_edges(self):
        """Create edges between genes and patients based on expression data (all genes measured in all patients)."""
        expr_data = self.tcga_data['expression']
        gene_ids = list(set(expr_data['gene_id'].unique()))
        gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
        # For simplicity, assume each row is a gene, each column after metadata is a patient/sample
        # If you have sample IDs, use them; otherwise, use index
        num_patients = expr_data.shape[0]
        edges = []
        for gene_idx in range(len(gene_ids)):
            for patient_idx in range(num_patients):
                edges.append([gene_idx, patient_idx])
        if edges:
            return np.array(edges).T
        else:
            return np.zeros((2, 0), dtype=int)

    def _create_gene_pubmed_edges(self):
        """Create edges between genes and PubMed articles based on NCBI gene2pubmed associations."""
        pubmed = self.ncbi_data['pubmed']
        expr_data = self.tcga_data['expression']
        gene_ids = list(set(expr_data['gene_id'].unique()))
        gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
        pubmed_ids = list(pubmed['PubMed_ID'].unique())
        pubmed_id_to_idx = {pmid: idx for idx, pmid in enumerate(pubmed_ids)}
        edges = []
        for _, row in pubmed.iterrows():
            gene_id = row['GeneID'] if 'GeneID' in row else row['gene_id']
            pubmed_id = row['PubMed_ID']
            if gene_id in gene_id_to_idx and pubmed_id in pubmed_id_to_idx:
                edges.append([gene_id_to_idx[gene_id], pubmed_id_to_idx[pubmed_id]])
        if edges:
            return np.array(edges).T
        else:
            return np.zeros((2, 0), dtype=int)

def main():
    # Initialize the integrator
    integrator = DatasetIntegrator()
    
    # Load all datasets
    integrator.load_tcga_data()
    integrator.load_ncbi_data()
    integrator.load_uci_data()
    integrator.load_kaggle_data()
    
    # Create integrated graph
    integrator.create_heterogeneous_graph()
    
    # Validate integration
    validation_metrics = integrator.validate_integration()
    
    # Save integrated data
    integrator.save_integrated_data()
    
    logger.info("Dataset integration completed successfully!")

if __name__ == "__main__":
    main() 