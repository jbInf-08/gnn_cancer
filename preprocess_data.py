# preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import gzip
import json
from pathlib import Path
import logging
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, data_dir='data/raw', processed_dir='data/processed'):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def is_gzipped(self, filepath):
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'

    def load_mutation_data(self, source='tcga'):
        """Load mutation data from various sources."""
        if source.lower() == 'tcga':
            # Load TCGA MAF files
            maf_files = list(self.data_dir.glob('**/*.maf'))
            dfs = []
            for maf_file in maf_files:
                maf_path = str(maf_file)
                try:
                    if maf_path.endswith('.gz') or self.is_gzipped(maf_file):
                        with gzip.open(maf_file, 'rt') as f:
                            df = pd.read_csv(f, sep='\t', comment='#')
                    else:
                        df = pd.read_csv(maf_file, sep='\t', comment='#')
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {maf_file}: {e}")
            return pd.concat(dfs, ignore_index=True)
        elif source.lower() == 'cosmic':
            # Load COSMIC mutation data
            cosmic_file = self.data_dir / 'cosmic_mutations.csv'
            return pd.read_csv(cosmic_file)
        else:
            raise ValueError(f"Unsupported mutation data source: {source}")
    
    def load_expression_data(self, source='gtex'):
        """Load gene expression data from various sources."""
        if source.lower() == 'gtex':
            # Load GTEx expression data
            gtex_file = self.data_dir / 'gtex_expression.gct.gz'
            with gzip.open(gtex_file, 'rt') as f:
                # Skip first two lines (header)
                next(f)
                next(f)
                df = pd.read_csv(f, sep='\t', index_col=0)
            return df
        elif source.lower() == 'tcga':
            # Look for *.FPKM.txt, expression.csv, and .tsv files
            expr_files = list(self.data_dir.glob('**/*.FPKM.txt'))
            expr_files += list(self.data_dir.glob('**/expression.csv'))
            expr_files += list(self.data_dir.glob('**/*.tsv'))
            # Prefer uncompressed BRCA_expression.tsv if it exists
            brca_expr_tsv = self.data_dir / 'BRCA_expression.tsv'
            if brca_expr_tsv.exists():
                # Use whitespace delimiter, skip comment lines, use gene_name as index
                df = pd.read_csv(brca_expr_tsv, sep=r'\s+', comment='#', engine='python', index_col='gene_name')
                return df
            if not expr_files:
                raise FileNotFoundError("No expression files (*.FPKM.txt, expression.csv, or *.tsv) found in data/raw or its subdirectories.")
            dfs = []
            for expr_file in expr_files:
                df = pd.read_csv(expr_file, sep='\t' if expr_file.suffix == '.tsv' else None, engine='python', index_col=0)
                dfs.append(df)
            if len(dfs) == 1:
                return dfs[0]
            return pd.concat(dfs, axis=1)
        else:
            raise ValueError(f"Unsupported expression data source: {source}")
    
    def load_cnv_data(self, source='tcga'):
        """Load copy number variation data."""
        if source.lower() == 'tcga':
            cnv_files = list(self.data_dir.glob('**/*.cnv.txt'))
            cnv_files += list(self.data_dir.glob('**/cnv.csv'))
            if not cnv_files:
                raise FileNotFoundError("No CNV files (*.cnv.txt or cnv.csv) found in data/raw or its subdirectories.")
            dfs = []
            for cnv_file in cnv_files:
                df = pd.read_csv(cnv_file, sep=None, engine='python', index_col=0)
                dfs.append(df)
            if len(dfs) == 1:
                return dfs[0]
            return pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(f"Unsupported CNV data source: {source}")
    
    def preprocess_mutation_data(self, mut_df, min_frequency=0.01):
        """Preprocess mutation data with advanced filtering."""
        # Filter relevant columns
        cols = ['Hugo_Symbol', 'Variant_Classification', 'Variant_Type', 
                'Tumor_Sample_Barcode', 'HGVSp_Short']
        mut_df = mut_df[cols]
        
        # One-hot encode mutation types
        mut_types = pd.get_dummies(mut_df['Variant_Type'], prefix='mut')
        mut_df = pd.concat([mut_df, mut_types], axis=1)
        
        # Filter genes with mutation frequency > threshold
        gene_counts = mut_df['Hugo_Symbol'].value_counts()
        min_count = len(mut_df['Tumor_Sample_Barcode'].unique()) * min_frequency
        frequent_genes = gene_counts[gene_counts >= min_count].index
        mut_df = mut_df[mut_df['Hugo_Symbol'].isin(frequent_genes)]
        
        # Create patient-gene matrix
        patients = mut_df['Tumor_Sample_Barcode'].unique()
        genes = mut_df['Hugo_Symbol'].unique()
        
        # Initialize matrix with zeros
        mut_matrix = pd.DataFrame(0, index=patients, columns=genes)
        
        # Fill matrix with mutation status
        for _, row in tqdm(mut_df.iterrows(), desc="Creating mutation matrix"):
            mut_matrix.loc[row['Tumor_Sample_Barcode'], row['Hugo_Symbol']] = 1
        
        return mut_matrix
    
    def normalize_expression_data(self, expr_df, method='minmax'):
        """Normalize gene expression data with different methods."""
        # Remove non-gene rows (e.g., N_unmapped, N_multimapping)
        expr_df = expr_df[~expr_df.index.str.startswith('N_')]
        # Transpose to get genes as columns
        expr_df = expr_df.T
        # Debug: print all index and column names containing 'N_'
        print('Index with N_:', [idx for idx in expr_df.index if 'N_' in str(idx)])
        print('Columns with N_:', [col for col in expr_df.columns if 'N_' in str(col)])
        # Remove non-gene columns (e.g., N_unmapped, N_multimapping)
        expr_df = expr_df.loc[:, ~expr_df.columns.str.startswith('N_')]
        # Convert all values to numeric, coercing errors to NaN
        expr_df = expr_df.apply(pd.to_numeric, errors='coerce')
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        expr_values = imputer.fit_transform(expr_df.values)
        # Normalize using specified method
        if method.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif method.lower() == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        normalized_values = scaler.fit_transform(expr_values)
        normalized_df = pd.DataFrame(
            normalized_values,
            index=expr_df.index, 
            columns=expr_df.columns
        )
        return normalized_df
    
    def preprocess_cnv_data(self, cnv_df):
        """Preprocess copy number variation data."""
        # Handle missing values
        cnv_df = cnv_df.fillna(0)
        
        # Transpose to get genes as columns
        cnv_df = cnv_df.T
        
        return cnv_df
    
    def merge_datasets(self, mut_df, expr_df, cnv_df=None):
        """Merge multiple data types with advanced alignment."""
        # Get common patients and genes
        common_patients = list(set(mut_df.index).intersection(expr_df.index))
        common_genes = list(set(mut_df.columns).intersection(expr_df.columns))
        print('common_patients:', common_patients[:5], '... total:', len(common_patients))
        print('common_genes:', common_genes[:5], '... total:', len(common_genes))

        # Create aligned matrices
        aligned_mut = mut_df.loc[common_patients, common_genes]
        aligned_expr = expr_df.loc[common_patients, common_genes]

        # Align CNV data if available
        if cnv_df is not None:
            cnv_genes = list(set(common_genes).intersection(cnv_df.columns))
            if cnv_genes:
                aligned_cnv = cnv_df.loc[common_patients, cnv_genes]
                return aligned_mut, aligned_expr, aligned_cnv

        return aligned_mut, aligned_expr, None
    
    def process_all_data(self, cancer_type='BRCA'):
        """Process all data types for a specific cancer type."""
        try:
            # Load data
            mut_df = self.load_mutation_data('tcga')
            expr_df = self.load_expression_data('tcga')
            cnv_df = self.load_cnv_data('tcga')
            
            # Preprocess individual datasets
            processed_mut = self.preprocess_mutation_data(mut_df)
            normalized_expr = self.normalize_expression_data(expr_df)
            processed_cnv = self.preprocess_cnv_data(cnv_df)
            
            # Debug: print shapes and indices/columns
            print('processed_mut shape:', processed_mut.shape)
            print('processed_mut index (patients):', list(processed_mut.index)[:5])
            print('processed_mut columns (genes):', list(processed_mut.columns)[:5])
            print('normalized_expr shape:', normalized_expr.shape)
            print('normalized_expr index (patients):', list(normalized_expr.index)[:5])
            print('normalized_expr columns (genes):', list(normalized_expr.columns)[:5])
            print('processed_cnv shape:', processed_cnv.shape)
            print('processed_cnv index (patients):', list(processed_cnv.index)[:5])
            print('processed_cnv columns (genes):', list(processed_cnv.columns)[:5])
            
            # Merge datasets
            aligned_mut, aligned_expr, aligned_cnv = self.merge_datasets(
                processed_mut, normalized_expr, processed_cnv
            )
            
            # Save processed data
            aligned_mut.to_csv(self.processed_dir / 'processed_mutations.csv')
            aligned_expr.to_csv(self.processed_dir / 'processed_expression.csv')
            if aligned_cnv is not None:
                aligned_cnv.to_csv(self.processed_dir / 'processed_cnv.csv')
            
            self.logger.info(f"Processed data saved. Mutation matrix shape: {aligned_mut.shape}")
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

def main():
    preprocessor = DataPreprocessor()
    preprocessor.process_all_data()

if __name__ == "__main__":
    main()