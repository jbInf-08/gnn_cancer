import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import requests
import gzip
import shutil
from gdc_client import GDCClient
import stringdb
import kegg
import reactome
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TCGA_Processor:
    def __init__(self, data_dir: Path = Path("data/raw")):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.gdc_client = GDCClient()
        
        # Create necessary directories
        self.mutation_dir = self.data_dir / "tcga/BRCA/mutation"
        self.expression_dir = self.data_dir / "tcga/BRCA/expression"
        self.cnv_dir = self.data_dir / "tcga/BRCA/cnv"
        self.processed_dir = self.data_dir / "processed"
        
        for directory in [self.mutation_dir, self.expression_dir, self.cnv_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def download_tcga_data(self):
        """Download TCGA breast cancer data using GDC client."""
        self.logger.info("Downloading TCGA breast cancer data...")
        
        # Download mutation data
        self.gdc_client.download_mutation_data(
            project_id="TCGA-BRCA",
            output_dir=self.mutation_dir
        )
        
        # Download expression data
        self.gdc_client.download_expression_data(
            project_id="TCGA-BRCA",
            output_dir=self.expression_dir
        )
        
        # Download CNV data
        self.gdc_client.download_cnv_data(
            project_id="TCGA-BRCA",
            output_dir=self.cnv_dir
        )
    
    def preprocess_mutation_data(self) -> pd.DataFrame:
        """Preprocess mutation data."""
        self.logger.info("Preprocessing mutation data...")
        
        # Read mutation data
        mutation_files = list(self.mutation_dir.glob("*.maf"))
        mutation_data = pd.concat([pd.read_csv(f, sep='\t') for f in mutation_files])
        
        # Filter for protein-coding mutations
        mutation_data = mutation_data[mutation_data['Variant_Classification'].isin([
            'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins'
        ])]
        
        # Create gene-level mutation matrix
        mutation_matrix = pd.crosstab(
            mutation_data['Hugo_Symbol'],
            mutation_data['Tumor_Sample_Barcode']
        )
        
        # Binarize mutations
        mutation_matrix = (mutation_matrix > 0).astype(int)
        
        return mutation_matrix
    
    def preprocess_expression_data(self) -> pd.DataFrame:
        """Preprocess expression data."""
        self.logger.info("Preprocessing expression data...")
        
        # Read expression data
        expression_files = list(self.expression_dir.glob("*.txt"))
        expression_data = pd.concat([pd.read_csv(f, sep='\t') for f in expression_files])
        
        # Log2 transform
        expression_data = np.log2(expression_data + 1)
        
        # Remove low variance genes
        gene_vars = expression_data.var(axis=1)
        expression_data = expression_data[gene_vars > gene_vars.quantile(0.1)]
        
        # Standardize
        scaler = StandardScaler()
        expression_data = pd.DataFrame(
            scaler.fit_transform(expression_data),
            index=expression_data.index,
            columns=expression_data.columns
        )
        
        return expression_data
    
    def preprocess_cnv_data(self) -> pd.DataFrame:
        """Preprocess CNV data."""
        self.logger.info("Preprocessing CNV data...")
        
        # Read CNV data
        cnv_files = list(self.cnv_dir.glob("*.txt"))
        cnv_data = pd.concat([pd.read_csv(f, sep='\t') for f in cnv_files])
        
        # Remove low quality segments
        cnv_data = cnv_data[cnv_data['Segment_Mean'].abs() > 0.1]
        
        # Create gene-level CNV matrix
        cnv_matrix = cnv_data.pivot_table(
            values='Segment_Mean',
            index='Gene_Symbol',
            columns='Sample_ID',
            aggfunc='mean'
        )
        
        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        cnv_matrix = pd.DataFrame(
            imputer.fit_transform(cnv_matrix),
            index=cnv_matrix.index,
            columns=cnv_matrix.columns
        )
        
        return cnv_matrix
    
    def get_ppi_network(self) -> nx.Graph:
        """Get protein-protein interaction network from STRING."""
        self.logger.info("Getting PPI network from STRING...")
        
        # Get PPI data with confidence > 0.7
        ppi_data = stringdb.get_network(confidence=0.7)
        
        # Create graph
        G = nx.Graph()
        for _, row in ppi_data.iterrows():
            G.add_edge(
                row['protein1'],
                row['protein2'],
                weight=row['combined_score'],
                edge_type='ppi'
            )
        
        return G
    
    def get_pathway_network(self) -> nx.Graph:
        """Get pathway co-occurrence network from KEGG and Reactome."""
        self.logger.info("Getting pathway network...")
        
        # Get KEGG pathways
        kegg_pathways = kegg.get_pathways()
        
        # Get Reactome pathways
        reactome_pathways = reactome.get_pathways()
        
        # Create graph
        G = nx.Graph()
        
        # Add edges for genes co-occurring in pathways
        for pathway in kegg_pathways + reactome_pathways:
            genes = pathway['genes']
            for i in range(len(genes)):
                for j in range(i+1, len(genes)):
                    if G.has_edge(genes[i], genes[j]):
                        G[genes[i]][genes[j]]['weight'] += 1
                    else:
                        G.add_edge(
                            genes[i],
                            genes[j],
                            weight=1,
                            edge_type='pathway'
                        )
        
        return G
    
    def get_coexpression_network(self, expression_data: pd.DataFrame) -> nx.Graph:
        """Get co-expression network."""
        self.logger.info("Getting co-expression network...")
        
        # Calculate correlation matrix
        corr_matrix = expression_data.corr()
        
        # Create graph
        G = nx.Graph()
        
        # Add edges for highly correlated genes
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i,j]) > 0.7:
                    G.add_edge(
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        weight=abs(corr_matrix.iloc[i,j]),
                        edge_type='coexpression'
                    )
        
        return G
    
    def integrate_networks(self, 
                         mutation_matrix: pd.DataFrame,
                         expression_data: pd.DataFrame,
                         cnv_matrix: pd.DataFrame) -> nx.Graph:
        """Integrate all networks into a single heterogeneous graph."""
        self.logger.info("Integrating networks...")
        
        # Get individual networks
        ppi_network = self.get_ppi_network()
        pathway_network = self.get_pathway_network()
        coexpression_network = self.get_coexpression_network(expression_data)
        
        # Combine networks
        G = nx.compose_all([ppi_network, pathway_network, coexpression_network])
        
        # Add node features
        for node in G.nodes():
            if node in mutation_matrix.index:
                G.nodes[node]['mutation_status'] = mutation_matrix.loc[node].mean()
            if node in expression_data.index:
                G.nodes[node]['expression'] = expression_data.loc[node].mean()
            if node in cnv_matrix.index:
                G.nodes[node]['cnv'] = cnv_matrix.loc[node].mean()
        
        return G
    
    def process_data(self):
        """Process all data and create the integrated graph."""
        # Download data
        self.download_tcga_data()
        
        # Preprocess data
        mutation_matrix = self.preprocess_mutation_data()
        expression_data = self.preprocess_expression_data()
        cnv_matrix = self.preprocess_cnv_data()
        
        # Create integrated graph
        G = self.integrate_networks(mutation_matrix, expression_data, cnv_matrix)
        
        # Save processed data
        nx.write_gpickle(G, self.processed_dir / "graph.pkl")
        mutation_matrix.to_csv(self.processed_dir / "mutation_matrix.csv")
        expression_data.to_csv(self.processed_dir / "expression_matrix.csv")
        cnv_matrix.to_csv(self.processed_dir / "cnv_matrix.csv")
        
        self.logger.info("Data processing complete!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = TCGA_Processor()
    
    # Process data
    processor.process_data() 