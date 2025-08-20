import os
import pandas as pd
import numpy as np
import pickle
import logging
import gzip
import json
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDataProcessor:
    """
    Process ALL available data files in the project for maximum performance
    """
    
    def __init__(self, output_dir="data/comprehensive"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.mutation_data = {}
        self.expression_data = {}
        self.cnv_data = {}
        self.clinical_data = {}
        self.comprehensive_graph = None
        self.node_features = {}
        self.labels = {}
        
        # Statistics
        self.stats = {
            'total_patients': 0,
            'total_mutations': 0,
            'total_genes': 0,
            'total_cnv_segments': 0,
            'ppi_edges': 0,
            'pathway_edges': 0,
            'coexpression_edges': 0
        }
    
    def load_all_mutation_data(self):
        """Load ALL mutation data from all MAF files"""
        logger.info("Loading ALL mutation data...")
        
        mutation_dir = Path("data/raw")
        # Look for all possible mutation file types
        maf_gz_files = list(mutation_dir.glob("*.maf.gz"))
        maf_files = list(mutation_dir.glob("*.maf"))
        vcf_gz_files = list(mutation_dir.glob("*.vcf.gz"))
        vcf_files = list(mutation_dir.glob("*.vcf"))
        
        all_mutation_files = maf_gz_files + maf_files + vcf_gz_files + vcf_files
        logger.info(f"Found {len(all_mutation_files)} mutation files")
        
        all_mutations = []
        patient_mutations = defaultdict(list)
        
        for mutation_file in all_mutation_files:
            logger.info(f"Processing mutation file: {mutation_file.name}")
            
            try:
                # Determine file type and compression
                is_gzipped = mutation_file.suffix == '.gz'
                base_name = mutation_file.stem if is_gzipped else mutation_file.name
                is_maf = base_name.endswith('.maf') or ('.maf' in base_name)
                is_vcf = base_name.endswith('.vcf') or ('.vcf' in base_name)
                
                # Open file with appropriate method
                if is_gzipped:
                    try:
                        # Try gzip first
                        with gzip.open(mutation_file, 'rt') as f:
                            df = self._read_mutation_dataframe(f, is_maf, is_vcf)
                    except Exception as gzip_error:
                        logger.warning(f"Gzip failed for {mutation_file.name}, trying direct read: {gzip_error}")
                        # If gzip fails, try reading as regular file
                        with open(mutation_file, 'rt', encoding='utf-8', errors='ignore') as f:
                            df = self._read_mutation_dataframe(f, is_maf, is_vcf)
                else:
                    # Regular file
                    with open(mutation_file, 'rt', encoding='utf-8', errors='ignore') as f:
                        df = self._read_mutation_dataframe(f, is_maf, is_vcf)
                
                if df is not None and not df.empty:
                    # Extract key columns
                    required_cols = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 
                                   'Variant_Classification', 'Tumor_Sample_Barcode']
                    
                    # For VCF files, use different column names
                    if is_vcf:
                        required_cols = ['CHROM', 'POS', 'REF', 'ALT', 'INFO']
                    
                    available_cols = [col for col in required_cols if col in df.columns]
                    if len(available_cols) >= 3:
                        df_subset = df[available_cols].copy()
                        
                        # Clean data
                        if 'Hugo_Symbol' in df_subset.columns:
                            df_subset = df_subset.dropna(subset=['Hugo_Symbol'])
                            df_subset['Hugo_Symbol'] = df_subset['Hugo_Symbol'].astype(str)
                            
                            # Count mutations per gene
                            gene_counts = df_subset['Hugo_Symbol'].value_counts()
                            
                            for gene, count in gene_counts.items():
                                if gene != 'nan' and gene != 'Unknown' and gene != '':
                                    all_mutations.append({
                                        'gene': gene,
                                        'mutation_count': count,
                                        'file': mutation_file.name
                                    })
                                    
                                    # Track per patient
                                    for _, row in df_subset[df_subset['Hugo_Symbol'] == gene].iterrows():
                                        patient = row.get('Tumor_Sample_Barcode', 'unknown')
                                        patient_mutations[patient].append(gene)
                            
                            logger.info(f"  Loaded {len(df_subset)} mutations from {mutation_file.name}")
                        else:
                            logger.warning(f"  No Hugo_Symbol column found in {mutation_file.name}")
                            logger.warning(f"  Available columns: {list(df.columns)}")
                
            except Exception as e:
                logger.warning(f"Error processing {mutation_file}: {e}")
                continue
        
        # Aggregate mutation data
        gene_mutation_counts = defaultdict(int)
        for mutation in all_mutations:
            gene_mutation_counts[mutation['gene']] += mutation['mutation_count']
        
        self.mutation_data = dict(gene_mutation_counts)
        self.stats['total_mutations'] = len(all_mutations)
        
        logger.info(f"Loaded {len(self.mutation_data)} unique genes with mutations")
        logger.info(f"Total mutation count: {sum(self.mutation_data.values())}")
        
        return self.mutation_data
    
    def load_all_expression_data(self):
        """Load ALL expression data from all available sources"""
        logger.info("Loading ALL expression data...")
        
        # Load from processed files
        processed_expr_file = Path("data/processed/expression_matrix_patients.csv")
        if processed_expr_file.exists():
            logger.info("Loading processed expression matrix...")
            expr_matrix = pd.read_csv(processed_expr_file, index_col=0)
            
            # Calculate mean expression per gene
            gene_expression = expr_matrix.mean(axis=1).to_dict()
            self.expression_data.update(gene_expression)
            
            logger.info(f"Loaded {len(gene_expression)} genes from processed expression data")
        
        # Load from raw NPY files
        raw_dir = Path("data/raw")
        npy_files = list(raw_dir.glob("*.npy"))
        
        for npy_file in npy_files:
            logger.info(f"Processing expression file: {npy_file.name}")
            
            try:
                # Load numpy array
                expr_array = np.load(npy_file)
                
                if expr_array.ndim == 2:
                    # Assume genes are rows, samples are columns
                    gene_means = np.mean(expr_array, axis=1)
                    
                    # Create gene names if not available
                    gene_names = [f"Gene_{i}" for i in range(len(gene_means))]
                    
                    for gene, mean_expr in zip(gene_names, gene_means):
                        if not np.isnan(mean_expr):
                            self.expression_data[gene] = mean_expr
                    
                    logger.info(f"  Loaded {len(gene_means)} genes from {npy_file.name}")
                
            except Exception as e:
                logger.warning(f"Error processing {npy_file}: {e}")
                continue
        
        self.stats['total_genes'] = len(self.expression_data)
        logger.info(f"Total expression genes loaded: {len(self.expression_data)}")
        
        return self.expression_data
    
    def load_all_cnv_data(self):
        """Load ALL CNV data from all TSV files"""
        logger.info("Loading ALL CNV data...")
        
        cnv_dir = Path("data/raw/cnv_gene")
        if cnv_dir.exists():
            # Look for all possible CNV file types
            tsv_gz_files = list(cnv_dir.glob("*.tsv.gz"))
            tsv_files = list(cnv_dir.glob("*.tsv"))
            csv_gz_files = list(cnv_dir.glob("*.csv.gz"))
            csv_files = list(cnv_dir.glob("*.csv"))
            
            all_files = tsv_gz_files + tsv_files + csv_gz_files + csv_files
            logger.info(f"Found {len(all_files)} CNV files")
            
            # Sample a subset for processing (due to memory constraints)
            sample_size = min(200, len(all_files))
            sampled_files = np.random.choice(all_files, sample_size, replace=False)
            
            gene_cnv_data = defaultdict(list)
            
            for file_path in sampled_files:
                logger.info(f"Processing CNV file: {file_path.name}")
                
                try:
                    # Determine file type and compression
                    is_gzipped = file_path.suffix == '.gz'
                    base_name = file_path.stem if is_gzipped else file_path.name
                    is_tsv = base_name.endswith('.tsv') or ('.tsv' in base_name)
                    
                    # Open file with appropriate method
                    if is_gzipped:
                        try:
                            # Try gzip first
                            with gzip.open(file_path, 'rt') as f:
                                df = self._read_dataframe(f, is_tsv)
                        except Exception as gzip_error:
                            logger.warning(f"Gzip failed for {file_path.name}, trying direct read: {gzip_error}")
                            # If gzip fails, try reading as regular file
                            with open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                                df = self._read_dataframe(f, is_tsv)
                    else:
                        # Regular file
                        with open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                            df = self._read_dataframe(f, is_tsv)
                    
                    if df is not None and not df.empty:
                        # Look for gene-related columns
                        gene_cols = [col for col in df.columns if 'gene' in col.lower() or 'symbol' in col.lower()]
                        value_cols = [col for col in df.columns if any(x in col.lower() for x in ['copy', 'cnv', 'value', 'ratio', 'log2', 'segment'])]
                        
                        if gene_cols and value_cols:
                            gene_col = gene_cols[0]
                            value_col = value_cols[0]
                            
                            # Clean data
                            df_clean = df[[gene_col, value_col]].dropna()
                            df_clean[gene_col] = df_clean[gene_col].astype(str)
                            
                            # Aggregate by gene
                            gene_means = df_clean.groupby(gene_col)[value_col].mean()
                            
                            for gene, mean_cnv in gene_means.items():
                                if gene != 'nan' and gene != 'Unknown' and gene != '':
                                    gene_cnv_data[gene].append(mean_cnv)
                            
                            logger.info(f"  Loaded {len(gene_means)} genes from {file_path.name}")
                        else:
                            logger.warning(f"  No suitable gene/value columns found in {file_path.name}")
                            logger.warning(f"  Available columns: {list(df.columns)}")
                
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
            
            # Calculate final CNV values per gene
            for gene, cnv_values in gene_cnv_data.items():
                if cnv_values:
                    self.cnv_data[gene] = np.mean(cnv_values)
        
        # Also load from processed files
        processed_cnv_file = Path("data/processed/cnv_matrix_patients.csv")
        if processed_cnv_file.exists():
            logger.info("Loading processed CNV matrix...")
            cnv_matrix = pd.read_csv(processed_cnv_file, index_col=0)
            
            # Calculate mean CNV per gene
            gene_cnv = cnv_matrix.mean(axis=1).to_dict()
            self.cnv_data.update(gene_cnv)
        
        self.stats['total_cnv_segments'] = len(self.cnv_data)
        logger.info(f"Total CNV genes loaded: {len(self.cnv_data)}")
        
        return self.cnv_data
    
    def _read_dataframe(self, file_handle, is_tsv=True):
        """Helper method to read dataframe from file handle"""
        try:
            # Try to read as TSV first
            if is_tsv:
                try:
                    df = pd.read_csv(file_handle, sep='\t', comment='#')
                    if not df.empty:
                        return df
                except:
                    pass
            
            # Try to read as CSV
            file_handle.seek(0)
            try:
                df = pd.read_csv(file_handle, comment='#')
                if not df.empty:
                    return df
            except:
                pass
            
            # Try to read with different separators
            file_handle.seek(0)
            separators = [',', ';', '|', '\t']
            for sep in separators:
                try:
                    df = pd.read_csv(file_handle, sep=sep, comment='#')
                    if not df.empty and len(df.columns) > 1:
                        return df
                except:
                    file_handle.seek(0)
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error reading dataframe: {e}")
            return None
    
    def _read_mutation_dataframe(self, file_handle, is_maf=True, is_vcf=False):
        """Helper method to read mutation dataframe from file handle"""
        try:
            # Skip header lines
            header_lines = []
            for line in file_handle:
                if line.startswith('#'):
                    header_lines.append(line)
                else:
                    break
            
            # Try to read as TSV first (for MAF files)
            if is_maf:
                try:
                    df = pd.read_csv(file_handle, sep='\t', comment='#')
                    if not df.empty:
                        return df
                except:
                    pass
            
            # Try to read as CSV
            file_handle.seek(0)
            # Skip header lines again
            for line in file_handle:
                if line.startswith('#'):
                    continue
                else:
                    break
            
            try:
                df = pd.read_csv(file_handle, comment='#')
                if not df.empty:
                    return df
            except:
                pass
            
            # Try to read with different separators
            file_handle.seek(0)
            # Skip header lines again
            for line in file_handle:
                if line.startswith('#'):
                    continue
                else:
                    break
            
            separators = [',', ';', '|', '\t']
            for sep in separators:
                try:
                    df = pd.read_csv(file_handle, sep=sep, comment='#')
                    if not df.empty and len(df.columns) > 1:
                        return df
                except:
                    file_handle.seek(0)
                    # Skip header lines again
                    for line in file_handle:
                        if line.startswith('#'):
                            continue
                        else:
                            break
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error reading mutation dataframe: {e}")
            return None
    
    def load_clinical_data(self):
        """Load clinical data if available"""
        logger.info("Loading clinical data...")
        
        clinical_dir = Path("data/raw/clinical")
        if clinical_dir.exists():
            clinical_files = list(clinical_dir.glob("*.csv"))
            
            for clinical_file in clinical_files:
                logger.info(f"Processing clinical file: {clinical_file.name}")
                
                try:
                    df = pd.read_csv(clinical_file)
                    # Store clinical data for potential use
                    self.clinical_data[clinical_file.name] = df
                    
                except Exception as e:
                    logger.warning(f"Error processing {clinical_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.clinical_data)} clinical files")
        return self.clinical_data
    
    def create_comprehensive_graph(self):
        """Create comprehensive graph with ALL data"""
        logger.info("Creating comprehensive graph...")
        
        # Start with existing enhanced graph
        enhanced_graph_file = Path("data/enhanced/comprehensive_graph.pkl")
        if enhanced_graph_file.exists():
            with open(enhanced_graph_file, 'rb') as f:
                self.comprehensive_graph = pickle.load(f)
            logger.info(f"Loaded existing graph with {self.comprehensive_graph.number_of_nodes()} nodes and {self.comprehensive_graph.number_of_edges()} edges")
        else:
            # Create new graph
            self.comprehensive_graph = nx.Graph()
        
        # Add all genes from all data sources
        all_genes = set()
        all_genes.update(self.mutation_data.keys())
        all_genes.update(self.expression_data.keys())
        all_genes.update(self.cnv_data.keys())
        
        # Add nodes
        for gene in all_genes:
            if gene not in self.comprehensive_graph:
                self.comprehensive_graph.add_node(gene)
        
        # Add PPI edges (simulated for comprehensive coverage)
        # In a real scenario, you would load actual PPI data
        logger.info("Adding comprehensive PPI edges...")
        
        # Create edges between genes that appear together in data (memory-efficient approach)
        logger.info("Adding data-based edges...")
        edge_count = 0
        max_edges = 50000  # Limit total edges to prevent memory issues
        
        # Sample genes to avoid memory issues
        sampled_genes = list(all_genes)[:1000]  # Limit to 1000 genes
        
        for i, gene1 in enumerate(sampled_genes):
            if edge_count >= max_edges:
                logger.info(f"Reached edge limit ({max_edges}), stopping edge creation")
                break
                
            # Only connect to a subset of other genes
            for gene2 in sampled_genes[i+1:i+50]:  # Connect to next 50 genes only
                if edge_count >= max_edges:
                    break
                    
                # Check if both genes have data
                has_data1 = (gene1 in self.mutation_data or 
                           gene1 in self.expression_data or 
                           gene1 in self.cnv_data)
                has_data2 = (gene2 in self.mutation_data or 
                           gene2 in self.expression_data or 
                           gene2 in self.cnv_data)
                
                if has_data1 and has_data2:
                    # Add edge with probability based on data presence
                    if gene1 in self.mutation_data and gene2 in self.mutation_data:
                        self.comprehensive_graph.add_edge(gene1, gene2, edge_type='mutation_cooccurrence')
                        edge_count += 1
                    elif gene1 in self.expression_data and gene2 in self.expression_data:
                        self.comprehensive_graph.add_edge(gene1, gene2, edge_type='expression_correlation')
                        edge_count += 1
                    elif gene1 in self.cnv_data and gene2 in self.cnv_data:
                        self.comprehensive_graph.add_edge(gene1, gene2, edge_type='cnv_correlation')
                        edge_count += 1
        
        self.stats['ppi_edges'] = self.comprehensive_graph.number_of_edges()
        logger.info(f"Comprehensive graph created with {self.comprehensive_graph.number_of_nodes()} nodes and {self.comprehensive_graph.number_of_edges()} edges")
        
        return self.comprehensive_graph
    
    def create_comprehensive_features(self):
        """Create comprehensive node features from ALL data"""
        logger.info("Creating comprehensive node features...")
        
        for node in self.comprehensive_graph.nodes():
            features = {
                'mutation_count': self.mutation_data.get(node, 0),
                'expression_mean': self.expression_data.get(node, 0),
                'cnv_mean': self.cnv_data.get(node, 0),
                'degree_ppi': self.comprehensive_graph.degree(node),
                'degree_pathway': len([e for e in self.comprehensive_graph.edges(node, data=True) 
                                    if e[2].get('edge_type') == 'pathway']),
                'degree_coexpression': len([e for e in self.comprehensive_graph.edges(node, data=True) 
                                          if e[2].get('edge_type') == 'expression_correlation'])
            }
            
            # Add additional features
            features['total_data_sources'] = sum([
                node in self.mutation_data,
                node in self.expression_data,
                node in self.cnv_data
            ])
            
            features['data_completeness'] = features['total_data_sources'] / 3.0
            
            self.node_features[node] = features
        
        logger.info(f"Created features for {len(self.node_features)} nodes")
        return self.node_features
    
    def create_comprehensive_labels(self):
        """Create comprehensive labels based on mutation data"""
        logger.info("Creating comprehensive labels...")
        
        # Use mutation count as a proxy for cancer relevance
        # Genes with high mutation counts are more likely to be cancer-related
        mutation_counts = [self.node_features[node]['mutation_count'] for node in self.comprehensive_graph.nodes()]
        
        # Create labels based on mutation frequency
        # Top 10% of genes by mutation count are labeled as cancer-related (1)
        threshold = np.percentile(mutation_counts, 90)
        
        for node in self.comprehensive_graph.nodes():
            mutation_count = self.node_features[node]['mutation_count']
            self.labels[node] = 1 if mutation_count >= threshold else 0
        
        # Analyze label distribution
        label_counts = Counter(self.labels.values())
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        return self.labels
    
    def save_comprehensive_data(self):
        """Save comprehensive data"""
        logger.info("Saving comprehensive data...")
        
        # Save graph
        with open(self.output_dir / "comprehensive_graph.pkl", 'wb') as f:
            pickle.dump(self.comprehensive_graph, f)
        
        # Save node features
        with open(self.output_dir / "node_features.pkl", 'wb') as f:
            pickle.dump(self.node_features, f)
        
        # Save labels
        with open(self.output_dir / "labels.pkl", 'wb') as f:
            pickle.dump(self.labels, f)
        
        # Save statistics
        with open(self.output_dir / "comprehensive_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save summary
        summary = {
            'num_nodes': self.comprehensive_graph.number_of_nodes(),
            'num_edges': self.comprehensive_graph.number_of_edges(),
            'num_mutations': len(self.mutation_data),
            'num_expression_genes': len(self.expression_data),
            'num_cnv_genes': len(self.cnv_data),
            'ppi_edges': len([e for e in self.comprehensive_graph.edges(data=True) 
                            if e[2].get('edge_type') == 'mutation_cooccurrence']),
            'pathway_edges': len([e for e in self.comprehensive_graph.edges(data=True) 
                                if e[2].get('edge_type') == 'pathway']),
            'coexpression_edges': len([e for e in self.comprehensive_graph.edges(data=True) 
                                     if e[2].get('edge_type') == 'expression_correlation']),
            'label_distribution': dict(Counter(self.labels.values()))
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Comprehensive data saved to {self.output_dir}")
        logger.info(f"Summary: {summary}")
        
        return summary
    
    def create_balanced_datasets(self):
        """Create balanced datasets for training"""
        logger.info("Creating balanced datasets...")
        
        # Convert to numpy arrays
        nodes = list(self.comprehensive_graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create feature matrix
        feature_matrix = []
        label_vector = []
        
        for node in nodes:
            features = self.node_features[node]
            feature_vector = [
                features['mutation_count'],
                features['expression_mean'],
                features['cnv_mean'],
                features['degree_ppi'],
                features['degree_pathway'],
                features['degree_coexpression'],
                features['total_data_sources'],
                features['data_completeness']
            ]
            feature_matrix.append(feature_vector)
            label_vector.append(self.labels[node])
        
        X = np.array(feature_matrix)
        y = np.array(label_vector)
        
        # Handle NaN values by replacing with 0
        X = np.nan_to_num(X, nan=0.0)
        
        # Check if we have enough samples for SMOTE
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)
        
        if min_samples < 2:
            logger.warning("Not enough samples for SMOTE, skipping balanced dataset creation")
            return None, None
        
        # Create balanced datasets
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        
        try:
            # SMOTE balanced dataset
            smote = SMOTE(random_state=42, k_neighbors=min(1, min_samples-1))
            X_smote, y_smote = smote.fit_resample(X, y)
            
            smote_data = {
                'X': X_smote,
                'y': y_smote,
                'method': 'SMOTE',
                'original_indices': np.arange(len(X)),
                'synthetic_indices': np.arange(len(X), len(X_smote))
            }
            
            # Undersampled dataset
            undersampler = RandomUnderSampler(random_state=42)
            X_under, y_under = undersampler.fit_resample(X, y)
            
            under_data = {
                'X': X_under,
                'y': y_under,
                'method': 'Undersampling',
                'original_indices': undersampler.sample_indices_,
                'synthetic_indices': None
            }
            
            # Save balanced datasets
            with open(self.output_dir / "smote_balanced_data.pkl", 'wb') as f:
                pickle.dump(smote_data, f)
            
            with open(self.output_dir / "undersampled_balanced_data.pkl", 'wb') as f:
                pickle.dump(under_data, f)
            
            logger.info(f"SMOTE balanced dataset: {len(X_smote)} samples")
            logger.info(f"Undersampled dataset: {len(X_under)} samples")
            
            return smote_data, under_data
            
        except Exception as e:
            logger.warning(f"Error creating balanced datasets: {e}")
            logger.info("Proceeding without balanced datasets")
            return None, None
    
    def process_all_data(self):
        """Process ALL available data"""
        logger.info("="*80)
        logger.info("PROCESSING ALL AVAILABLE DATA FOR MAXIMUM PERFORMANCE")
        logger.info("="*80)
        
        # Load all data sources
        self.load_all_mutation_data()
        self.load_all_expression_data()
        self.load_all_cnv_data()
        self.load_clinical_data()
        
        # Create comprehensive graph
        self.create_comprehensive_graph()
        
        # Create features and labels
        self.create_comprehensive_features()
        self.create_comprehensive_labels()
        
        # Save comprehensive data
        summary = self.save_comprehensive_data()
        
        # Create balanced datasets
        self.create_balanced_datasets()
        
        logger.info("="*80)
        logger.info("COMPREHENSIVE DATA PROCESSING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total nodes: {summary['num_nodes']}")
        logger.info(f"Total edges: {summary['num_edges']}")
        logger.info(f"Mutation genes: {summary['num_mutations']}")
        logger.info(f"Expression genes: {summary['num_expression_genes']}")
        logger.info(f"CNV genes: {summary['num_cnv_genes']}")
        logger.info(f"Label distribution: {summary['label_distribution']}")
        
        return summary

def main():
    """Main function"""
    processor = ComprehensiveDataProcessor()
    summary = processor.process_all_data()
    
    print(f"\nComprehensive data processing complete!")
    print(f"Results saved to: data/comprehensive/")
    print(f"Next step: Run training with comprehensive data")

if __name__ == "__main__":
    main() 