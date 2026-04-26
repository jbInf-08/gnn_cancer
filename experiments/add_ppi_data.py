#!/usr/bin/env python3
"""
Add Protein-Protein Interaction (PPI) network data to enhance the graph structure.
This matches the research paper's approach of integrating PPI networks.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import json
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPINetworkIntegrator:
    """Integrate PPI networks into the graph structure."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.ppi_cache_file = data_dir / "ppi_network.pt"
        
    def load_string_ppi_data(self, gene_list: List[str], confidence_threshold: float = 0.7) -> Dict:
        """
        Load PPI data from STRING database for given genes.
        
        Args:
            gene_list: List of gene names
            confidence_threshold: Minimum confidence score for interactions
            
        Returns:
            Dictionary containing PPI network data
        """
        logger.info(f"Loading PPI data for {len(gene_list)} genes from STRING database...")
        
        # Check cache first
        if self.ppi_cache_file.exists():
            logger.info("Loading PPI data from cache...")
            ppi_data = torch.load(self.ppi_cache_file)
            return ppi_data
        
        # For now, create synthetic PPI data based on gene co-occurrence
        # In a real implementation, you would query the STRING API
        logger.info("Creating synthetic PPI network (replace with real STRING API calls)...")
        
        ppi_network = self._create_synthetic_ppi_network(gene_list, confidence_threshold)
        
        # Cache the data
        torch.save(ppi_network, self.ppi_cache_file)
        logger.info(f"PPI network saved to {self.ppi_cache_file}")
        
        return ppi_network
    
    def _create_synthetic_ppi_network(self, gene_list: List[str], confidence_threshold: float) -> Dict:
        """
        Create synthetic PPI network for demonstration.
        In practice, this would be replaced with real STRING API calls.
        """
        num_genes = len(gene_list)
        
        # Create synthetic interactions based on gene similarity
        # Genes with similar names or functions are more likely to interact
        interactions = []
        confidence_scores = []
        
        for i in range(num_genes):
            for j in range(i + 1, num_genes):
                # Simple heuristic: genes with similar names are more likely to interact
                gene1, gene2 = gene_list[i], gene_list[j]
                
                # Calculate similarity score (0-1)
                similarity = self._calculate_gene_similarity(gene1, gene2)
                
                # Only include interactions above confidence threshold
                if similarity >= confidence_threshold:
                    interactions.append([i, j])
                    confidence_scores.append(similarity)
        
        # Convert to tensors
        edge_index = torch.tensor(interactions, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(confidence_scores, dtype=torch.float)
        
        # Create edge attributes: [interaction_type, confidence_score, coexpression]
        edge_attr = torch.zeros(len(interactions), 3)
        edge_attr[:, 0] = 0  # PPI interaction type
        edge_attr[:, 1] = edge_weights  # Confidence score
        edge_attr[:, 2] = torch.rand(len(interactions)) * 0.5  # Synthetic coexpression
        
        logger.info(f"Created PPI network with {len(interactions)} interactions")
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_weights': edge_weights,
            'num_interactions': len(interactions),
            'confidence_threshold': confidence_threshold
        }
    
    def _calculate_gene_similarity(self, gene1: str, gene2: str) -> float:
        """
        Calculate similarity between two genes based on name similarity.
        This is a simple heuristic - in practice, use functional similarity.
        """
        # Convert to lowercase for comparison
        g1, g2 = gene1.lower(), gene2.lower()
        
        # Check for exact match
        if g1 == g2:
            return 1.0
        
        # Check for substring matches
        if g1 in g2 or g2 in g1:
            return 0.8
        
        # Check for common prefixes
        min_len = min(len(g1), len(g2))
        common_prefix = 0
        for i in range(min_len):
            if g1[i] == g2[i]:
                common_prefix += 1
            else:
                break
        
        if min_len > 0:
            return common_prefix / min_len * 0.6
        
        return 0.1
    
    def integrate_ppi_with_existing_graph(self, data, ppi_network: Dict):
        """
        Integrate PPI network with existing graph data.
        
        Args:
            data: Existing PyTorch Geometric data object
            ppi_network: PPI network data
            
        Returns:
            Enhanced data object with PPI edges
        """
        logger.info("Integrating PPI network with existing graph...")
        
        # Get existing edges
        existing_edge_index = data.edge_index
        existing_edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Get PPI edges
        ppi_edge_index = ppi_network['edge_index']
        ppi_edge_attr = ppi_network['edge_attr']
        
        # Combine edges
        combined_edge_index = torch.cat([existing_edge_index, ppi_edge_index], dim=1)
        
        # Combine edge attributes
        if existing_edge_attr is not None:
            combined_edge_attr = torch.cat([existing_edge_attr, ppi_edge_attr], dim=0)
        else:
            # If no existing edge attributes, create them for existing edges
            num_existing_edges = existing_edge_index.size(1)
            existing_edge_attr = torch.zeros(num_existing_edges, 3)
            existing_edge_attr[:, 0] = 1  # Different interaction type
            existing_edge_attr[:, 1] = 1.0  # Full confidence
            existing_edge_attr[:, 2] = 0.0  # No coexpression
            combined_edge_attr = torch.cat([existing_edge_attr, ppi_edge_attr], dim=0)
        
        # Create enhanced data object
        enhanced_data = data.clone()
        enhanced_data.edge_index = combined_edge_index
        enhanced_data.edge_attr = combined_edge_attr
        
        logger.info(f"Enhanced graph: {existing_edge_index.size(1)} original edges + {ppi_edge_index.size(1)} PPI edges = {combined_edge_index.size(1)} total edges")
        
        return enhanced_data

def enhance_data_with_ppi(data, gene_names: List[str], data_dir: Path):
    """
    Main function to enhance data with PPI networks.
    
    Args:
        data: PyTorch Geometric data object
        gene_names: List of gene names
        data_dir: Directory to store PPI data
        
    Returns:
        Enhanced data object with PPI and pathway information
    """
    integrator = PPINetworkIntegrator(data_dir)
    
    # Load PPI data
    ppi_network = integrator.load_string_ppi_data(gene_names, confidence_threshold=0.7)
    
    # Integrate PPI with existing graph
    enhanced_data = integrator.integrate_ppi_with_existing_graph(data, ppi_network)
    
    logger.info("Data enhancement complete!")
    logger.info(f"Final graph: {enhanced_data.num_nodes} nodes, {enhanced_data.edge_index.size(1)} edges")
    
    return enhanced_data

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # This would be called from the main training script
    print("PPI Network Integration Module")
    print("Use enhance_data_with_ppi() function to integrate PPI networks into your graph data.") 