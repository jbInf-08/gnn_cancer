# build_graph.py
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from pathlib import Path
import logging
from tqdm import tqdm
import gzip
from sklearn.preprocessing import MinMaxScaler
import os

class GraphBuilder:
    def __init__(self, data_dir: Path = Path("data/processed")):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        
    def build_graph(self) -> nx.Graph:
        """Build the integrated graph for mutation analysis."""
        self.logger.info("Building integrated graph...")
        G = nx.Graph()
        # Load processed data
        expression_df = pd.read_csv(self.data_dir / "processed_expression.csv", index_col=0)
        mutation_df = pd.read_csv(self.data_dir / "processed_mutations.csv")
        # Load CNV data
        cnv_path = Path("data/raw/tcga/BRCA/cnv/cnv.csv")
        cnv_df = pd.read_csv(cnv_path) if cnv_path.exists() else None
        # Load protein abundance data
        protein_path = Path("data/raw/tcga/BRCA/protein/BREAST-CPTAC-TCGA_normalized_total_protein.xlsx")
        protein_df = pd.read_excel(protein_path) if protein_path.exists() else None
        # Load driver status
        driver_path = Path("data/processed/cancer_drivers/gene_info.csv")
        driver_df = pd.read_csv(driver_path) if driver_path.exists() else None
        # Add nodes with all features
        self._add_gene_nodes(G, expression_df, mutation_df, cnv_df, protein_df, driver_df)
        # Add co-expression edges
        self._add_coexpression_edges(G, expression_df)
        # Add PPI edges
        self._add_ppi_edges(G)
        # Add pathway edges
        self._add_pathway_edges(G)
        self.logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _add_gene_nodes(self, G: nx.Graph, expression_df: pd.DataFrame, mutation_df: pd.DataFrame, cnv_df: pd.DataFrame, protein_df: pd.DataFrame, driver_df: pd.DataFrame) -> None:
        """Add gene nodes with mutation, expression, CNV, protein abundance, and driver status."""
        # Get unique genes
        genes = expression_df.index.unique()
        
        # Process and add nodes
        for gene in tqdm(genes, desc="Adding gene nodes"):
            # Expression
            expression_values = expression_df.loc[gene].values
            # Mutation
            mutation_status = 1 if gene in mutation_df['Hugo_Symbol'].values else 0
            # CNV
            cnv = float(cnv_df[cnv_df['gene'] == gene]['cnv'].values[0]) if cnv_df is not None and gene in cnv_df['gene'].values else np.nan
            # Protein abundance
            protein = float(protein_df[protein_df['Gene'] == gene].iloc[0,1]) if protein_df is not None and 'Gene' in protein_df.columns and gene in protein_df['Gene'].values else np.nan
            # Driver status
            driver = int(driver_df[driver_df['Symbol'] == gene]['is_driver'].values[0]) if driver_df is not None and gene in driver_df['Symbol'].values else 0
            
            # Add node with features
            G.add_node(gene, 
                      mutation_status=mutation_status,
                      expression_mean=np.mean(expression_values),
                      expression_std=np.std(expression_values),
                      cnv=cnv,
                      protein_abundance=protein,
                      driver_status=driver)
    
    def _add_coexpression_edges(self, G: nx.Graph, expression_df: pd.DataFrame) -> None:
        """Add co-expression edges based on expression data."""
        self.logger.info("Adding co-expression edges...")
        
        # Calculate correlation matrix
        corr_matrix = expression_df.T.corr()
        
        # Add edges for highly correlated genes
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i,j]) > 0.7:  # Correlation threshold
                    gene1 = corr_matrix.columns[i]
                    gene2 = corr_matrix.columns[j]
                    if G.has_node(gene1) and G.has_node(gene2):
                        G.add_edge(gene1, gene2,
                                 weight=abs(corr_matrix.iloc[i,j]),
                                 edge_type='coexpression')
    
    def _add_ppi_edges(self, G: nx.Graph):
        self.logger.info("Adding PPI edges...")
        string_path = Path("data/external/string/protein_links.txt")
        if not string_path.exists():
            self.logger.warning("STRING PPI file not found.")
            return
        with open(string_path, 'r') as f:
            for line in tqdm(f, desc="PPI edges", total=6000000):
                if line.startswith('protein1'): continue
                parts = line.strip().split()
                if len(parts) < 3: continue
                gene1, gene2, score = parts[0], parts[1], float(parts[2])
                # Only add if both genes are in the graph and score > 700 (STRING uses 0-1000)
                if gene1 in G and gene2 in G and score > 700:
                    G.add_edge(gene1, gene2, weight=score/1000, edge_type='ppi')

    def _add_pathway_edges(self, G: nx.Graph):
        self.logger.info("Adding pathway edges...")
        pathway_files = [
            Path("data/external/pathways/KEGG_2021_Human.gmt"),
            Path("data/external/pathways/Reactome_2022.gmt")
        ]
        for pf in pathway_files:
            if not pf.exists():
                self.logger.warning(f"Pathway file {pf} not found.")
                continue
            with open(pf, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    genes = [g for g in parts[2:] if g in G]
                    # Add edges for all pairs in the pathway
                    for i in range(len(genes)):
                        for j in range(i+1, len(genes)):
                            g1, g2 = genes[i], genes[j]
                            if not G.has_edge(g1, g2):
                                G.add_edge(g1, g2, weight=1.0, edge_type='pathway')
    
    def save_graph(self, G: nx.Graph, output_path: Path) -> None:
        """Save the graph to disk."""
        self.logger.info(f"Saving graph to {output_path}")
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
    
    def load_graph(self, input_path: Path) -> nx.Graph:
        """Load the graph from disk."""
        self.logger.info(f"Loading graph from {input_path}")
        return nx.read_gpickle(input_path)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Build graph
    builder = GraphBuilder()
    G = builder.build_graph()
    
    # Save graph
    builder.save_graph(G, Path("data/processed/graph.pkl"))
    
    # Print some statistics
    print(f"\nGraph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # Count mutated genes
    mutated_genes = sum(1 for node in G.nodes() if G.nodes[node]['mutation_status'] == 1)
    print(f"Number of mutated genes: {mutated_genes}")

if __name__ == "__main__":
    main()