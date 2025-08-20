import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import gzip

def read_expression_data(file_path):
    """Read and process gene expression data."""
    print("Reading gene expression data...")
    # Read the file
    df = pd.read_csv(file_path, sep='\t')
    
    # Set gene name as index
    df.set_index('Gene', inplace=True)
    
    print(f"Loaded {len(df)} genes and {len(df.columns)} samples")
    return df

def read_mutation_data(file_path):
    """Read and process mutation data."""
    print("Reading mutation data...")
    # Read the gzipped file
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', comment='#')
    
    # Extract relevant columns
    df = df[['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Variant_Classification']]
    
    # Clean up sample barcodes
    df['Tumor_Sample_Barcode'] = df['Tumor_Sample_Barcode'].str[:12]
    
    print(f"Loaded {len(df)} mutations across {df['Tumor_Sample_Barcode'].nunique()} samples")
    return df

def create_gene_network(expression_df, mutation_df, top_n_genes=100):
    """Create a gene network based on expression correlation and mutations."""
    print("Creating gene network...")
    
    # Select top N most variable genes
    gene_vars = expression_df.var(axis=1)
    top_genes = gene_vars.nlargest(top_n_genes).index
    
    # Calculate correlation matrix for top genes
    corr_matrix = expression_df.loc[top_genes].T.corr()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (genes)
    for gene in top_genes:
        G.add_node(gene, type='gene')
    
    # Add edges based on correlation
    for i, gene1 in enumerate(top_genes):
        for gene2 in top_genes[i+1:]:
            corr = corr_matrix.loc[gene1, gene2]
            if abs(corr) > 0.5:  # Only add edges for strong correlations
                G.add_edge(gene1, gene2, weight=corr, type='correlation')
    
    # Add mutation information
    for gene in top_genes:
        if gene in mutation_df['Hugo_Symbol'].values:
            G.nodes[gene]['mutated'] = True
        else:
            G.nodes[gene]['mutated'] = False
    
    print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def visualize_network(G, output_path):
    """Visualize the gene network."""
    print("Visualizing network...")
    plt.figure(figsize=(12, 12))
    
    # Set up the layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    mutated_nodes = [n for n in G.nodes() if G.nodes[n].get('mutated', False)]
    non_mutated_nodes = [n for n in G.nodes() if not G.nodes[n].get('mutated', False)]
    
    nx.draw_networkx_nodes(G, pos, nodelist=mutated_nodes, node_color='red', 
                          node_size=100, alpha=0.6, label='Mutated')
    nx.draw_networkx_nodes(G, pos, nodelist=non_mutated_nodes, node_color='blue',
                          node_size=100, alpha=0.6, label='Non-mutated')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Gene Network (Red: Mutated, Blue: Non-mutated)')
    plt.legend()
    plt.axis('off')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {output_path}")

def main():
    # Setup paths
    data_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read data
    expression_df = read_expression_data(data_dir / "BRCA_expression.tsv")
    mutation_df = read_mutation_data(data_dir / "BRCA_mutation.maf.gz")
    
    # Create and visualize network
    G = create_gene_network(expression_df, mutation_df, top_n_genes=50)
    visualize_network(G, output_dir / "gene_network.png")
    
    # Save processed data
    print("Saving processed data...")
    expression_df.to_csv(output_dir / "processed_expression.csv")
    mutation_df.to_csv(output_dir / "processed_mutations.csv")
    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main() 