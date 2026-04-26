import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_geometric.data import HeteroData
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cancer_driver_data(data_dir: Path, census_path: Path):
    """Load cancer driver gene data from NCBI and Cancer Gene Census."""
    logger.info("Loading cancer driver gene data...")
    
    # Load Cancer Gene Census
    census_df = pd.read_csv(census_path)
    driver_entrez_ids = set(map(str, census_df['Entrez GeneId'].dropna().astype(int)))
    driver_symbols = set(census_df['Gene Symbol'].dropna().str.upper())
    
    logger.info(f"Found {len(driver_entrez_ids)} driver genes in Cancer Gene Census")
    
    # Load gene info in chunks and filter for driver genes
    gene_info_path = data_dir / "gene_info.gz"
    gene_info_chunks = pd.read_csv(gene_info_path, compression='gzip', sep='\t', chunksize=10000)
    
    # Collect all genes that are in the Cancer Gene Census
    driver_genes = []
    all_genes = []
    
    for chunk in gene_info_chunks:
        # Filter for genes that are in the Cancer Gene Census
        chunk_drivers = chunk[chunk['GeneID'].astype(str).isin(driver_entrez_ids)]
        driver_genes.append(chunk_drivers)
        all_genes.append(chunk)
    
    gene_info = pd.concat(all_genes, ignore_index=True)
    driver_gene_info = pd.concat(driver_genes, ignore_index=True)
    
    logger.info(f"Found {len(driver_gene_info)} driver genes in NCBI data")
    logger.info(f"Total genes in dataset: {len(gene_info)}")
    
    # Sample non-driver genes to create balanced dataset
    non_driver_genes = gene_info[~gene_info['GeneID'].astype(str).isin(driver_entrez_ids)]
    
    # Sample non-driver genes (aim for 2:1 ratio of non-drivers to drivers)
    target_non_drivers = min(len(driver_gene_info) * 2, len(non_driver_genes))
    sampled_non_drivers = non_driver_genes.sample(n=target_non_drivers, random_state=42)
    
    # Combine driver and sampled non-driver genes
    balanced_gene_info = pd.concat([driver_gene_info, sampled_non_drivers], ignore_index=True)
    
    logger.info(f"Final balanced dataset: {len(driver_gene_info)} drivers, {len(sampled_non_drivers)} non-drivers")
    
    # Load gene to GO mappings for all genes in our dataset
    gene_ids_in_dataset = set(balanced_gene_info['GeneID'].astype(str))
    gene2go_path = data_dir / "gene2go.gz"
    gene2go_chunks = pd.read_csv(gene2go_path, compression='gzip', sep='\t', chunksize=10000)
    gene2go = pd.concat([chunk[chunk['GeneID'].astype(str).isin(gene_ids_in_dataset)] for chunk in gene2go_chunks])
    
    # Load gene to PubMed mappings for all genes in our dataset
    gene2pubmed_path = data_dir / "gene2pubmed.gz"
    gene2pubmed_chunks = pd.read_csv(gene2pubmed_path, compression='gzip', sep='\t', chunksize=10000)
    gene2pubmed = pd.concat([chunk[chunk['GeneID'].astype(str).isin(gene_ids_in_dataset)] for chunk in gene2pubmed_chunks])
    
    logger.info(f"Found {len(gene2go)} GO term associations")
    logger.info(f"Found {len(gene2pubmed)} PubMed associations")
    
    return balanced_gene_info, gene2go, gene2pubmed, driver_entrez_ids

def load_brca1_data(data_dir: Path):
    """Load BRCA1 gene data from NCBI (fallback method)."""
    logger.info("Loading BRCA1 gene data...")
    
    # Load gene info in chunks and filter for BRCA1
    gene_info_path = data_dir / "gene_info.gz"
    gene_info_chunks = pd.read_csv(gene_info_path, compression='gzip', sep='\t', chunksize=10000)
    gene_info = pd.concat([chunk[chunk['Symbol'] == 'BRCA1'] for chunk in gene_info_chunks])
    logger.info(f"Found {len(gene_info)} BRCA1 gene entries")
    
    # Load gene to GO mappings in chunks and filter for BRCA1
    gene2go_path = data_dir / "gene2go.gz"
    gene2go_chunks = pd.read_csv(gene2go_path, compression='gzip', sep='\t', chunksize=10000)
    gene2go = pd.concat([chunk[chunk['GeneID'].isin(gene_info['GeneID'])] for chunk in gene2go_chunks])
    logger.info(f"Found {len(gene2go)} GO term associations for BRCA1")
    
    # Load gene to PubMed mappings in chunks and filter for BRCA1
    gene2pubmed_path = data_dir / "gene2pubmed.gz"
    gene2pubmed_chunks = pd.read_csv(gene2pubmed_path, compression='gzip', sep='\t', chunksize=10000)
    gene2pubmed = pd.concat([chunk[chunk['GeneID'].isin(gene_info['GeneID'])] for chunk in gene2pubmed_chunks])
    logger.info(f"Found {len(gene2pubmed)} PubMed associations for BRCA1")
    
    return gene_info, gene2go, gene2pubmed

def create_heterogeneous_graph(gene_info, gene2go, gene2pubmed, driver_entrez_ids):
    """Create a heterogeneous graph from cancer driver gene data."""
    logger.info("Creating heterogeneous graph from cancer driver gene data...")
    
    # Create node mappings
    gene_ids = {str(gene_id): idx for idx, gene_id in enumerate(gene_info['GeneID'])}
    go_ids = {str(go_id): idx for idx, go_id in enumerate(gene2go['GO_ID'].unique())}
    pubmed_ids = {str(pubmed_id): idx for idx, pubmed_id in enumerate(gene2pubmed['PubMed_ID'].unique())}
    
    logger.info(f"Created mappings for {len(gene_ids)} genes, {len(go_ids)} GO terms, and {len(pubmed_ids)} PubMed IDs")
    
    # Create gene features
    numerical_features = gene_info.select_dtypes(include=[np.number]).values
    if len(numerical_features) > 0:
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)
    else:
        numerical_features_scaled = np.array([])
    
    categorical_features = gene_info[['Symbol', 'description']].values
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_features_encoded = encoder.fit_transform(categorical_features)
    
    if len(numerical_features_scaled) > 0:
        gene_features = np.hstack([numerical_features_scaled, categorical_features_encoded])
    else:
        gene_features = categorical_features_encoded
    
    # Create GO term features (one-hot encoding of their categories)
    go_categories = gene2go['Category'].unique()
    go_category_to_idx = {cat: idx for idx, cat in enumerate(go_categories)}
    go_features = np.zeros((len(go_ids), len(go_categories)))
    for _, row in gene2go.iterrows():
        go_idx = go_ids[str(row['GO_ID'])]
        cat_idx = go_category_to_idx[row['Category']]
        go_features[go_idx, cat_idx] = 1
    
    # Create PubMed features (using publication year if available)
    pubmed_features = np.zeros((len(pubmed_ids), 1))
    if 'Year' in gene2pubmed.columns:
        for _, row in gene2pubmed.iterrows():
            pubmed_idx = pubmed_ids[str(row['PubMed_ID'])]
            pubmed_features[pubmed_idx, 0] = row['Year']
    
    # Create edge indices
    gene_go_edges = []
    for _, row in gene2go.iterrows():
        gene_id = str(row['GeneID'])
        go_id = str(row['GO_ID'])
        if gene_id in gene_ids and go_id in go_ids:
            gene_go_edges.append([gene_ids[gene_id], go_ids[go_id]])
    
    gene_pubmed_edges = []
    for _, row in gene2pubmed.iterrows():
        gene_id = str(row['GeneID'])
        pubmed_id = str(row['PubMed_ID'])
        if gene_id in gene_ids and pubmed_id in pubmed_ids:
            gene_pubmed_edges.append([gene_ids[gene_id], pubmed_ids[pubmed_id]])
    
    # Create heterogeneous graph
    data = HeteroData()
    
    # Add node features
    data['gene'].x = torch.tensor(gene_features, dtype=torch.float)
    data['go'].x = torch.tensor(go_features, dtype=torch.float)
    data['pubmed'].x = torch.tensor(pubmed_features, dtype=torch.float)
    
    # Add edge indices
    data['gene', 'associated_with', 'go'].edge_index = torch.tensor(gene_go_edges, dtype=torch.long).t().contiguous()
    data['gene', 'cited_in', 'pubmed'].edge_index = torch.tensor(gene_pubmed_edges, dtype=torch.long).t().contiguous()
    
    # Assign binary driver gene labels using Cancer Gene Census
    gene_entrez_ids = gene_info['GeneID'].astype(str).values
    driver_labels = np.array([1 if gene_id in driver_entrez_ids else 0 for gene_id in gene_entrez_ids], dtype=np.int64)
    data['gene'].y = torch.tensor(driver_labels, dtype=torch.long)

    logger.info(f"Assigned binary driver gene labels: {driver_labels.sum()} driver(s), {len(driver_labels) - driver_labels.sum()} non-driver(s)")
    
    logger.info(f"Created heterogeneous graph with:")
    logger.info(f"- {data['gene'].x.size(0)} gene nodes")
    logger.info(f"- {data['go'].x.size(0)} GO term nodes")
    logger.info(f"- {data['pubmed'].x.size(0)} PubMed nodes")
    logger.info(f"- {data['gene', 'associated_with', 'go'].edge_index.size(1)} gene-GO edges")
    logger.info(f"- {data['gene', 'cited_in', 'pubmed'].edge_index.size(1)} gene-PubMed edges")
    
    return data

def main():
    data_dir = Path("data/raw/ncbi/brca1")
    census_path = Path("data/external/Census_allFri Jun 20 16_29_53 2025.csv")
    
    # Check if we have the full NCBI data or need to use BRCA1 data
    if (data_dir / "gene_info.gz").exists():
        gene_info, gene2go, gene2pubmed, driver_entrez_ids = load_cancer_driver_data(data_dir, census_path)
    else:
        logger.warning("Full NCBI data not found, using BRCA1 data only")
        # Fallback to original BRCA1 approach
        gene_info, gene2go, gene2pubmed = load_brca1_data(data_dir)
        census_df = pd.read_csv(census_path)
        driver_entrez_ids = set(map(str, census_df['Entrez GeneId'].dropna().astype(int)))
    
    graph_data = create_heterogeneous_graph(gene_info, gene2go, gene2pubmed, driver_entrez_ids)
    
    # Save processed data
    output_dir = Path("data/processed/cancer_drivers")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph_data, output_dir / "heterogeneous_graph.pt")
    logger.info(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    main() 