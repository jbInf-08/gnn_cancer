# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_geometric.data import HeteroData
import torch
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_enhanced_dataset():
    """Create an enhanced dataset with more cancer driver genes."""
    logger.info("Creating enhanced cancer driver dataset...")
    
    # Load Cancer Gene Census
    census_path = Path("data/external/Census_allFri Jun 20 16_29_53 2025.csv")
    census_df = pd.read_csv(census_path)
    
    # Get driver genes from Cancer Gene Census
    driver_genes = census_df['Gene Symbol'].dropna().str.upper().unique()
    logger.info(f"Found {len(driver_genes)} unique driver genes in Cancer Gene Census")
    
    # Load existing BRCA1 data as template
    try:
        existing_data = torch.load('data/processed/brca1/heterogeneous_graph.pt', weights_only=False)
        logger.info("Loaded existing BRCA1 data as template")
    except FileNotFoundError:
        logger.error("Existing BRCA1 data not found. Please run the original processing first.")
        return None
    
    # Create synthetic gene data for driver genes
    num_drivers = min(50, len(driver_genes))  # Use up to 50 driver genes
    num_non_drivers = 100  # Add 100 non-driver genes
    
    # Sample driver genes
    selected_drivers = random.sample(list(driver_genes), num_drivers)
    
    # Create gene info DataFrame
    gene_data = []
    
    # Add driver genes
    for i, gene_symbol in enumerate(selected_drivers):
        gene_data.append({
            'GeneID': f"DRIVER_{i}",
            'Symbol': gene_symbol,
            'description': f"Cancer driver gene {gene_symbol}",
            'is_driver': 1
        })
    
    # Add non-driver genes (synthetic)
    for i in range(num_non_drivers):
        gene_data.append({
            'GeneID': f"NONDRIVER_{i}",
            'Symbol': f"GENE_{i}",
            'description': f"Non-driver gene {i}",
            'is_driver': 0
        })
    
    gene_info = pd.DataFrame(gene_data)
    
    # Create synthetic GO and PubMed data
    go_terms = [f"GO:{i:07d}" for i in range(100)]
    pubmed_ids = [f"PMID:{i}" for i in range(200)]
    
    # Create gene-GO associations
    gene2go_data = []
    for _, gene in gene_info.iterrows():
        num_go_terms = random.randint(1, 5)
        selected_go = random.sample(go_terms, num_go_terms)
        for go_term in selected_go:
            gene2go_data.append({
                'GeneID': gene['GeneID'],
                'GO_ID': go_term,
                'Category': random.choice(['BP', 'MF', 'CC'])
            })
    
    gene2go = pd.DataFrame(gene2go_data)
    
    # Create gene-PubMed associations
    gene2pubmed_data = []
    for _, gene in gene_info.iterrows():
        num_pubmed = random.randint(0, 3)
        selected_pubmed = random.sample(pubmed_ids, num_pubmed)
        for pubmed_id in selected_pubmed:
            gene2pubmed_data.append({
                'GeneID': gene['GeneID'],
                'PubMed_ID': pubmed_id,
                'Year': random.randint(2010, 2024)
            })
    
    gene2pubmed = pd.DataFrame(gene2pubmed_data)
    
    logger.info(f"Created enhanced dataset with {len(gene_info)} genes")
    logger.info(f"  - {num_drivers} driver genes")
    logger.info(f"  - {num_non_drivers} non-driver genes")
    logger.info(f"  - {len(gene2go)} GO associations")
    logger.info(f"  - {len(gene2pubmed)} PubMed associations")
    
    return gene_info, gene2go, gene2pubmed, selected_drivers

def create_enhanced_heterogeneous_graph(gene_info, gene2go, gene2pubmed, driver_genes):
    """Create an enhanced heterogeneous graph from the expanded data."""
    logger.info("Creating enhanced heterogeneous graph...")
    
    # Create node mappings
    gene_ids = {gene_id: idx for idx, gene_id in enumerate(gene_info['GeneID'])}
    go_ids = {go_id: idx for idx, go_id in enumerate(gene2go['GO_ID'].unique())}
    pubmed_ids = {pubmed_id: idx for idx, pubmed_id in enumerate(gene2pubmed['PubMed_ID'].unique())}
    
    logger.info(f"Created mappings for {len(gene_ids)} genes, {len(go_ids)} GO terms, and {len(pubmed_ids)} PubMed IDs")
    
    # Create gene features (synthetic but realistic)
    num_genes = len(gene_info)
    
    # Create numerical features (expression levels, mutation counts, etc.)
    numerical_features = np.random.randn(num_genes, 10)  # 10 numerical features
    
    # Create categorical features
    gene_symbols = gene_info['Symbol'].values
    descriptions = gene_info['description'].values
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_features = encoder.fit_transform(np.column_stack([gene_symbols, descriptions]))
    
    # Combine features
    gene_features = np.hstack([numerical_features, categorical_features])
    
    # Create GO term features
    go_categories = gene2go['Category'].unique()
    go_category_to_idx = {cat: idx for idx, cat in enumerate(go_categories)}
    go_features = np.zeros((len(go_ids), len(go_categories)))
    
    for _, row in gene2go.iterrows():
        go_idx = go_ids[row['GO_ID']]
        cat_idx = go_category_to_idx[row['Category']]
        go_features[go_idx, cat_idx] = 1
    
    # Create PubMed features
    pubmed_features = np.zeros((len(pubmed_ids), 1))
    for _, row in gene2pubmed.iterrows():
        pubmed_idx = pubmed_ids[row['PubMed_ID']]
        pubmed_features[pubmed_idx, 0] = row['Year']
    
    # Create edge indices
    gene_go_edges = []
    for _, row in gene2go.iterrows():
        gene_id = row['GeneID']
        go_id = row['GO_ID']
        if gene_id in gene_ids and go_id in go_ids:
            gene_go_edges.append([gene_ids[gene_id], go_ids[go_id]])
    
    gene_pubmed_edges = []
    for _, row in gene2pubmed.iterrows():
        gene_id = row['GeneID']
        pubmed_id = row['PubMed_ID']
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
    
    # Assign binary driver gene labels
    driver_labels = gene_info['is_driver'].values
    data['gene'].y = torch.tensor(driver_labels, dtype=torch.long)
    
    logger.info(f"Assigned binary driver gene labels: {driver_labels.sum()} driver(s), {len(driver_labels) - driver_labels.sum()} non-driver(s)")
    
    logger.info(f"Created enhanced heterogeneous graph with:")
    logger.info(f"- {data['gene'].x.size(0)} gene nodes")
    logger.info(f"- {data['go'].x.size(0)} GO term nodes")
    logger.info(f"- {data['pubmed'].x.size(0)} PubMed nodes")
    logger.info(f"- {data['gene', 'associated_with', 'go'].edge_index.size(1)} gene-GO edges")
    logger.info(f"- {data['gene', 'cited_in', 'pubmed'].edge_index.size(1)} gene-PubMed edges")
    
    return data

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create enhanced dataset
    gene_info, gene2go, gene2pubmed, driver_genes = create_enhanced_dataset()
    
    if gene_info is None:
        return
    
    # Create enhanced graph
    graph_data = create_enhanced_heterogeneous_graph(gene_info, gene2go, gene2pubmed, driver_genes)
    
    # Save processed data
    output_dir = Path("data/processed/cancer_drivers")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph_data, output_dir / "heterogeneous_graph.pt")
    
    # Save gene information for reference
    gene_info.to_csv(output_dir / "gene_info.csv", index=False)
    
    logger.info(f"Enhanced data saved to {output_dir}")
    logger.info(f"Driver genes included: {', '.join(driver_genes[:10])}{'...' if len(driver_genes) > 10 else ''}")

if __name__ == "__main__":
    main() 