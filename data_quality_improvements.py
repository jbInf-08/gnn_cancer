"""
Data Quality Improvements to Match Paper Methodology
- Validate graph construction methodology
- Verify protein-protein interaction network quality
- Check pathway information integration
- Validate feature engineering approaches
- Ensure data matches paper specifications
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityValidator:
    """
    Comprehensive data quality validation and improvement
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = None
        self.validation_results = {}
        
    def load_data(self):
        """Load all relevant data files"""
        logger.info("Loading data for quality validation...")
        
        # Load enhanced data
        enhanced_file = self.data_path / "enhanced" / "real_only_torch_geometric_data.pt"
        if enhanced_file.exists():
            self.data = torch.load(enhanced_file, weights_only=False)
            logger.info(f"Loaded enhanced data: {enhanced_file}")
        else:
            raise FileNotFoundError(f"Enhanced data not found: {enhanced_file}")
        
        # Load additional data files
        self.load_additional_data()
        
    def load_additional_data(self):
        """Load additional data files for validation"""
        # Load STRING PPI data
        string_file = self.data_path / "external" / "string" / "protein_links.txt"
        if string_file.exists():
            self.string_data = pd.read_csv(string_file, sep=' ', compression='gzip' if string_file.suffix == '.gz' else None)
            logger.info(f"Loaded STRING data: {len(self.string_data)} interactions")
        
        # Load pathway data
        kegg_file = self.data_path / "external" / "pathways" / "KEGG_2021_Human.gmt"
        if kegg_file.exists():
            self.kegg_pathways = self.load_gmt_file(kegg_file)
            logger.info(f"Loaded KEGG pathways: {len(self.kegg_pathways)} pathways")
        
        reactome_file = self.data_path / "external" / "pathways" / "Reactome_2022.gmt"
        if reactome_file.exists():
            self.reactome_pathways = self.load_gmt_file(reactome_file)
            logger.info(f"Loaded Reactome pathways: {len(self.reactome_pathways)} pathways")
    
    def load_gmt_file(self, file_path: Path) -> Dict[str, List[str]]:
        """Load GMT format pathway files"""
        pathways = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    pathway_name = parts[0]
                    pathway_desc = parts[1]
                    genes = parts[2:]
                    pathways[pathway_name] = genes
        return pathways
    
    def validate_graph_construction(self):
        """Validate graph construction methodology"""
        logger.info("Validating graph construction...")
        
        results = {}
        
        # Check graph structure
        num_nodes = self.data.x.shape[0]
        num_edges = self.data.edge_index.shape[1]
        density = num_edges / (num_nodes * (num_nodes - 1))
        
        results['graph_stats'] = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': 2 * num_edges / num_nodes
        }
        
        # Check connectivity
        edge_set = set()
        for i in range(self.data.edge_index.shape[1]):
            edge_set.add((self.data.edge_index[0, i].item(), self.data.edge_index[1, i].item()))
        
        # Check for isolated nodes
        isolated_nodes = 0
        connected_components = []
        visited = set()
        
        for node in range(num_nodes):
            if node not in visited:
                component = self.find_connected_component(node, edge_set)
                connected_components.append(component)
                visited.update(component)
                if len(component) == 1:
                    isolated_nodes += 1
        
        results['connectivity'] = {
            'isolated_nodes': isolated_nodes,
            'num_components': len(connected_components),
            'largest_component_size': max(len(comp) for comp in connected_components),
            'component_sizes': [len(comp) for comp in connected_components]
        }
        
        # Check edge attributes
        if self.data.edge_attr is not None:
            edge_types = self.data.edge_attr[:, 0].unique()
            results['edge_types'] = {
                'num_types': len(edge_types),
                'type_distribution': {int(t): (self.data.edge_attr[:, 0] == t).sum().item() 
                                    for t in edge_types}
            }
        
        logger.info(f"Graph validation results: {results}")
        self.validation_results['graph_construction'] = results
        return results
    
    def find_connected_component(self, start_node: int, edge_set: set) -> set:
        """Find connected component using BFS"""
        component = set()
        queue = [start_node]
        component.add(start_node)
        
        while queue:
            node = queue.pop(0)
            for edge in edge_set:
                if edge[0] == node and edge[1] not in component:
                    component.add(edge[1])
                    queue.append(edge[1])
                elif edge[1] == node and edge[0] not in component:
                    component.add(edge[0])
                    queue.append(edge[0])
        
        return component
    
    def validate_ppi_network(self):
        """Validate protein-protein interaction network quality"""
        logger.info("Validating PPI network quality...")
        
        results = {}
        
        if hasattr(self, 'string_data'):
            # Check STRING data quality
            results['string_stats'] = {
                'total_interactions': len(self.string_data),
                'unique_proteins': len(set(self.string_data['protein1'].unique()) | 
                                     set(self.string_data['protein2'].unique())),
                'confidence_scores': {
                    'min': self.string_data['combined_score'].min(),
                    'max': self.string_data['combined_score'].max(),
                    'mean': self.string_data['combined_score'].mean(),
                    'std': self.string_data['combined_score'].std()
                }
            }
            
            # Check confidence score distribution
            high_confidence = self.string_data[self.string_data['combined_score'] >= 700]
            medium_confidence = self.string_data[(self.string_data['combined_score'] >= 400) & 
                                               (self.string_data['combined_score'] < 700)]
            
            results['confidence_distribution'] = {
                'high_confidence': len(high_confidence),
                'medium_confidence': len(medium_confidence),
                'high_confidence_ratio': len(high_confidence) / len(self.string_data)
            }
            
            # Check for breast cancer related interactions
            if 'breast' in str(self.string_data.columns).lower():
                breast_cancer_interactions = self.string_data[
                    self.string_data.apply(lambda x: 'breast' in str(x).lower(), axis=1)
                ]
                results['breast_cancer_interactions'] = len(breast_cancer_interactions)
        
        logger.info(f"PPI validation results: {results}")
        self.validation_results['ppi_network'] = results
        return results
    
    def validate_pathway_integration(self):
        """Validate pathway information integration"""
        logger.info("Validating pathway integration...")
        
        results = {}
        
        # Check KEGG pathways
        if hasattr(self, 'kegg_pathways'):
            breast_cancer_pathways = [name for name in self.kegg_pathways.keys() 
                                    if 'breast' in name.lower() or 'cancer' in name.lower()]
            
            results['kegg_pathways'] = {
                'total_pathways': len(self.kegg_pathways),
                'breast_cancer_pathways': len(breast_cancer_pathways),
                'avg_genes_per_pathway': np.mean([len(genes) for genes in self.kegg_pathways.values()]),
                'breast_cancer_pathway_names': breast_cancer_pathways
            }
        
        # Check Reactome pathways
        if hasattr(self, 'reactome_pathways'):
            breast_cancer_pathways = [name for name in self.reactome_pathways.keys() 
                                    if 'breast' in name.lower() or 'cancer' in name.lower()]
            
            results['reactome_pathways'] = {
                'total_pathways': len(self.reactome_pathways),
                'breast_cancer_pathways': len(breast_cancer_pathways),
                'avg_genes_per_pathway': np.mean([len(genes) for genes in self.reactome_pathways.values()]),
                'breast_cancer_pathway_names': breast_cancer_pathways
            }
        
        logger.info(f"Pathway validation results: {results}")
        self.validation_results['pathway_integration'] = results
        return results
    
    def validate_feature_engineering(self):
        """Validate feature engineering approaches"""
        logger.info("Validating feature engineering...")
        
        results = {}
        
        # Check node features
        node_features = self.data.x
        
        results['node_features'] = {
            'num_features': node_features.shape[1],
            'feature_stats': {
                'mean': node_features.mean().item(),
                'std': node_features.std().item(),
                'min': node_features.min().item(),
                'max': node_features.max().item()
            }
        }
        
        # Check for NaN values
        nan_count = torch.isnan(node_features).sum().item()
        results['data_quality'] = {
            'nan_values': nan_count,
            'nan_percentage': nan_count / node_features.numel() * 100
        }
        
        # Check feature distribution
        feature_means = node_features.mean(dim=0)
        feature_stds = node_features.std(dim=0)
        
        results['feature_distribution'] = {
            'mean_of_means': feature_means.mean().item(),
            'std_of_means': feature_means.std().item(),
            'mean_of_stds': feature_stds.mean().item(),
            'std_of_stds': feature_stds.std().item()
        }
        
        # Check for feature correlation
        if node_features.shape[1] <= 100:  # Only for reasonable number of features
            feature_corr = torch.corrcoef(node_features.T)
            high_corr_pairs = torch.where(torch.abs(feature_corr) > 0.8)
            high_corr_pairs = [(i.item(), j.item()) for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]) 
                              if i < j]  # Remove duplicates
            
            results['feature_correlation'] = {
                'high_correlation_pairs': len(high_corr_pairs),
                'max_correlation': feature_corr.max().item()
            }
        
        logger.info(f"Feature engineering validation results: {results}")
        self.validation_results['feature_engineering'] = results
        return results
    
    def validate_labels(self):
        """Validate label quality and distribution"""
        logger.info("Validating labels...")
        
        results = {}
        
        if hasattr(self.data, 'y') and self.data.y is not None:
            labels = self.data.y
            
            results['label_stats'] = {
                'num_samples': len(labels),
                'num_classes': len(labels.unique()),
                'class_distribution': {int(c): (labels == c).sum().item() 
                                     for c in labels.unique()}
            }
            
            # Check class balance
            class_counts = [(labels == c).sum().item() for c in labels.unique()]
            min_count = min(class_counts)
            max_count = max(class_counts)
            balance_ratio = min_count / max_count
            
            results['class_balance'] = {
                'balance_ratio': balance_ratio,
                'is_balanced': balance_ratio > 0.3,  # Threshold for balanced
                'minority_class_size': min_count,
                'majority_class_size': max_count
            }
        
        logger.info(f"Label validation results: {results}")
        self.validation_results['labels'] = results
        return results
    
    def improve_data_quality(self):
        """Apply data quality improvements"""
        logger.info("Applying data quality improvements...")
        
        improvements = {}
        
        # 1. Handle NaN values
        if torch.isnan(self.data.x).any():
            logger.info("Handling NaN values in node features...")
            self.data.x = torch.nan_to_num(self.data.x, nan=0.0)
            improvements['nan_handling'] = 'Applied nan_to_num'
        
        if self.data.edge_attr is not None and torch.isnan(self.data.edge_attr).any():
            logger.info("Handling NaN values in edge attributes...")
            self.data.edge_attr = torch.nan_to_num(self.data.edge_attr, nan=0.0)
            improvements['edge_nan_handling'] = 'Applied nan_to_num'
        
        # 2. Feature normalization
        logger.info("Applying feature normalization...")
        scaler = StandardScaler()
        self.data.x = torch.tensor(scaler.fit_transform(self.data.x.cpu().numpy()), 
                                 dtype=torch.float32, device=self.data.x.device)
        improvements['feature_normalization'] = 'Applied StandardScaler'
        
        # 3. Remove isolated nodes if any
        edge_set = set()
        for i in range(self.data.edge_index.shape[1]):
            edge_set.add((self.data.edge_index[0, i].item(), self.data.edge_index[1, i].item()))
        
        isolated_nodes = []
        for node in range(self.data.x.shape[0]):
            connected = False
            for edge in edge_set:
                if edge[0] == node or edge[1] == node:
                    connected = True
                    break
            if not connected:
                isolated_nodes.append(node)
        
        if isolated_nodes:
            logger.info(f"Found {len(isolated_nodes)} isolated nodes, removing...")
            # Create node mapping
            node_mapping = {}
            new_idx = 0
            for old_idx in range(self.data.x.shape[0]):
                if old_idx not in isolated_nodes:
                    node_mapping[old_idx] = new_idx
                    new_idx += 1
            
            # Update node features
            keep_indices = [i for i in range(self.data.x.shape[0]) if i not in isolated_nodes]
            self.data.x = self.data.x[keep_indices]
            
            # Update edge index
            new_edges = []
            for i in range(self.data.edge_index.shape[1]):
                src, dst = self.data.edge_index[:, i]
                if src.item() in node_mapping and dst.item() in node_mapping:
                    new_edges.append([node_mapping[src.item()], node_mapping[dst.item()]])
            
            self.data.edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.data.edge_index.device).T
            
            # Update edge attributes
            if self.data.edge_attr is not None:
                new_edge_attrs = []
                for i in range(self.data.edge_index.shape[1]):
                    src, dst = self.data.edge_index[:, i]
                    if src.item() in node_mapping and dst.item() in node_mapping:
                        new_edge_attrs.append(self.data.edge_attr[i])
                self.data.edge_attr = torch.stack(new_edge_attrs)
            
            # Update labels
            if hasattr(self.data, 'y') and self.data.y is not None:
                self.data.y = self.data.y[keep_indices]
            
            improvements['isolated_node_removal'] = f'Removed {len(isolated_nodes)} isolated nodes'
        
        # 4. Add self-loops if missing
        edge_set = set()
        for i in range(self.data.edge_index.shape[1]):
            edge_set.add((self.data.edge_index[0, i].item(), self.data.edge_index[1, i].item()))
        
        missing_self_loops = []
        for node in range(self.data.x.shape[0]):
            if (node, node) not in edge_set:
                missing_self_loops.append(node)
        
        if missing_self_loops:
            logger.info(f"Adding {len(missing_self_loops)} self-loops...")
            self_loop_edges = torch.tensor([[node, node] for node in missing_self_loops], 
                                         dtype=torch.long, device=self.data.edge_index.device).T
            
            self.data.edge_index = torch.cat([self.data.edge_index, self_loop_edges], dim=1)
            
            if self.data.edge_attr is not None:
                # Add edge attributes for self-loops (type 0)
                self_loop_attrs = torch.zeros(len(missing_self_loops), self.data.edge_attr.shape[1], 
                                            dtype=torch.float32, device=self.data.edge_attr.device)
                self.data.edge_attr = torch.cat([self.data.edge_attr, self_loop_attrs], dim=0)
            
            improvements['self_loops'] = f'Added {len(missing_self_loops)} self-loops'
        
        logger.info(f"Data quality improvements applied: {improvements}")
        self.validation_results['improvements'] = improvements
        return improvements
    
    def save_improved_data(self):
        """Save the improved data"""
        output_file = self.data_path / "enhanced" / "improved_torch_geometric_data.pt"
        torch.save(self.data, output_file)
        logger.info(f"Saved improved data to {output_file}")
        
        # Save validation results
        validation_file = self.data_path / "enhanced" / "data_validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        logger.info(f"Saved validation results to {validation_file}")
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature distribution
        if 'feature_engineering' in self.validation_results:
            feature_stats = self.validation_results['feature_engineering']['feature_stats']
            stats_names = list(feature_stats.keys())
            stats_values = list(feature_stats.values())
            
            axes[0, 0].bar(stats_names, stats_values, color='skyblue')
            axes[0, 0].set_title('Feature Statistics')
            axes[0, 0].set_ylabel('Value')
        
        # 2. Class distribution
        if 'labels' in self.validation_results:
            class_dist = self.validation_results['labels']['label_stats']['class_distribution']
            classes = list(class_dist.keys())
            counts = list(class_dist.values())
            
            axes[0, 1].pie(counts, labels=classes, autopct='%1.1f%%')
            axes[0, 1].set_title('Class Distribution')
        
        # 3. Graph connectivity
        if 'graph_construction' in self.validation_results:
            graph_stats = self.validation_results['graph_construction']['graph_stats']
            stat_names = ['Nodes', 'Edges', 'Density', 'Avg Degree']
            stat_values = [graph_stats['num_nodes'], graph_stats['num_edges'], 
                          graph_stats['density'], graph_stats['avg_degree']]
            
            axes[1, 0].bar(stat_names, stat_values, color='lightgreen')
            axes[1, 0].set_title('Graph Statistics')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Edge type distribution
        if 'graph_construction' in self.validation_results and 'edge_types' in self.validation_results['graph_construction']:
            edge_dist = self.validation_results['graph_construction']['edge_types']['type_distribution']
            edge_types = list(edge_dist.keys())
            edge_counts = list(edge_dist.values())
            
            axes[1, 1].bar(edge_types, edge_counts, color='lightcoral')
            axes[1, 1].set_title('Edge Type Distribution')
            axes[1, 1].set_xlabel('Edge Type')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.data_path / "enhanced" / "data_validation_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Validation report generated and saved")
    
    def run_comprehensive_validation(self):
        """Run comprehensive data validation and improvement"""
        logger.info("Starting comprehensive data validation...")
        
        # Load data
        self.load_data()
        
        # Run all validations
        self.validate_graph_construction()
        self.validate_ppi_network()
        self.validate_pathway_integration()
        self.validate_feature_engineering()
        self.validate_labels()
        
        # Apply improvements
        self.improve_data_quality()
        
        # Save improved data
        self.save_improved_data()
        
        # Generate report
        self.generate_validation_report()
        
        logger.info("Comprehensive data validation completed!")
        return self.validation_results

def main():
    """Main execution function"""
    validator = DataQualityValidator('data')
    results = validator.run_comprehensive_validation()
    
    print("\n" + "="*50)
    print("DATA QUALITY VALIDATION COMPLETED")
    print("="*50)
    print(f"Graph nodes: {results['graph_construction']['graph_stats']['num_nodes']}")
    print(f"Graph edges: {results['graph_construction']['graph_stats']['num_edges']}")
    print(f"Feature count: {results['feature_engineering']['node_features']['num_features']}")
    print(f"Class balance ratio: {results['labels']['class_balance']['balance_ratio']:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
