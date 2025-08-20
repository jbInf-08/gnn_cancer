"""
Enhanced Data Collection for Comprehensive Multi-omics Integration
Implements the reference paper's approach for scaling to 150+ samples
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import re
from io import StringIO
import glob
import gzip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """
    Comprehensive data collector for multi-omics cancer data
    Implements reference paper's approach for 150+ samples
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # API endpoints
        self.gdc_api = "https://api.gdc.cancer.gov"
        self.cptac_api = "https://proteomics.cancer.gov/program/cptac"
        
        # Data storage
        self.mutation_data = {}
        self.expression_data = {}
        self.cnv_data = {}
        self.protein_data = {}
        self.clinical_data = {}
        
    def collect_tcga_brca_data(self, max_samples: int = 200) -> Dict:
        """
        Collect comprehensive TCGA BRCA data
        Target: 150+ samples with full multi-omics profile
        """
        logger.info(f"Collecting TCGA BRCA data for up to {max_samples} samples")
        
        # TCGA BRCA project ID
        project_id = "TCGA-BRCA"
        
        # Collect mutation data (SNVs)
        logger.info("Collecting mutation data...")
        # Recursively find all .maf and .maf.gz files in data/raw and data/raw/tcga/BRCA
        maf_files = glob.glob("data/raw/**/*.maf", recursive=True)
        maf_files += glob.glob("data/raw/**/*.maf.gz", recursive=True)
        logger.info(f"Found {len(maf_files)} mutation files: {maf_files[:3]}{'...' if len(maf_files) > 3 else ''}")
        mutation_dfs = []
        for maf_file in maf_files:
            try:
                try:
                    df = pd.read_csv(maf_file, sep='\t', comment='#', low_memory=False)
                except UnicodeDecodeError:
                    logger.info(f"File {maf_file} appears to be gzipped, retrying with gzip...")
                    with gzip.open(maf_file, 'rt') as f:
                        df = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
                # Only keep relevant columns if present
                keep_cols = [
                    'Hugo_Symbol', 'Tumor_Sample_Barcode', 'Chromosome', 'Start_Position',
                    'Variant_Classification', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2'
                ]
                df = df[[col for col in keep_cols if col in df.columns]]
                mutation_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to process mutation file {maf_file}: {e}")
        if mutation_dfs:
            mutations = pd.concat(mutation_dfs, ignore_index=True)
        else:
            mutations = pd.DataFrame()
        logger.info(f"Loaded mutation data shape: {mutations.shape}")
        
        # Collect expression data
        logger.info("Collecting gene expression data...")
        expression_files = self._query_gdc_files(
            project_id=project_id,
            data_category="Transcriptome Profiling",
            data_type="Gene Expression Quantification",
            max_samples=max_samples
        )
        
        # Collect CNV data
        logger.info("Collecting CNV data...")
        cnv_files = self._query_gdc_files(
            project_id=project_id,
            data_category="Copy Number Variation",
            data_type="Copy Number Segment",
            max_samples=max_samples
        )
        
        # Collect clinical data
        logger.info("Collecting clinical data...")
        clinical_files = self._query_gdc_files(
            project_id=project_id,
            data_category="Clinical",
            data_type="Clinical Supplement",
            max_samples=max_samples
        )
        
        # Download and process data
        data = {
            'mutations': mutations,
            'expression': self._download_and_process_expression(expression_files),
            'cnv': self._download_and_process_cnv(cnv_files),
            'clinical': self._download_and_process_clinical(clinical_files)
        }
        
        logger.info(f"Collected data for {len(data['clinical'])} samples")
        return data
    
    def collect_cptac_brca_data(self) -> Dict:
        """
        Collect CPTAC BRCA data for protein abundance
        """
        logger.info("Collecting CPTAC BRCA protein data...")
        
        # CPTAC BRCA protein data
        cptac_data = {
            'protein_abundance': {},
            'phosphoproteomics': {},
            'clinical': {}
        }
        
        # Download CPTAC data files
        cptac_urls = {
            'protein': 'https://proteomics.cancer.gov/sites/default/files/2020-08/CPTAC_BRCA_Proteome_CDAP.r2.precursor_area.tsv',
            'phospho': 'https://proteomics.cancer.gov/sites/default/files/2020-08/CPTAC_BRCA_Phosphoproteome_CDAP.r2.precursor_area.tsv',
            'clinical': 'https://proteomics.cancer.gov/sites/default/files/2020-08/CPTAC_BRCA_Clinical_CDAP.r2.tsv'
        }
        
        for data_type, url in cptac_urls.items():
            try:
                logger.info(f"Downloading CPTAC {data_type} data...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save raw data
                output_file = self.data_dir / f"cptac_brca_{data_type}.tsv"
                with open(output_file, 'w') as f:
                    f.write(response.text)
                
                # Process data
                df = pd.read_csv(output_file, sep='\t')
                cptac_data[f'{data_type}_abundance'] = df
                
            except Exception as e:
                logger.error(f"Failed to download CPTAC {data_type} data: {e}")
        
        return cptac_data
    
    def _query_gdc_files(self, project_id: str, data_category: str, 
                        data_type: str, max_samples: int) -> List[str]:
        """
        Query GDC API for file IDs
        """
        query = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "=",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": project_id
                        }
                    },
                    {
                        "op": "=",
                        "content": {
                            "field": "files.data_category",
                            "value": data_category
                        }
                    },
                    {
                        "op": "=",
                        "content": {
                            "field": "files.data_type",
                            "value": data_type
                        }
                    }
                ]
            },
            "format": "json",
            "size": max_samples
        }
        
        try:
            response = requests.post(f"{self.gdc_api}/files", json=query)
            response.raise_for_status()
            data = response.json()
            
            file_ids = [file['file_id'] for file in data['data']['hits']]
            logger.info(f"Found {len(file_ids)} files for {data_category}/{data_type}")
            return file_ids
            
        except Exception as e:
            logger.error(f"Failed to query GDC API: {e}")
            return []
    
    def _find_maf_files(self):
        # Find all .maf and .maf.gz files in relevant directories
        maf_files = glob.glob('data/raw/*.maf') + glob.glob('data/raw/*.maf.gz') \
            + glob.glob('data/raw/tcga/BRCA/mutation/*.maf') + glob.glob('data/raw/tcga/BRCA/mutation/*.maf.gz')
        return maf_files

    def _download_and_process_expression(self, file_ids: List[str]) -> pd.DataFrame:
        """
        Download and robustly process gene expression data.
        For each file, extract only the 'fpkm_unstranded' column (or another quantification column),
        use the file_id as the sample column name, and merge into a single DataFrame with genes as rows
        and sample_ids as columns. Skips special rows like 'N_unmapped', 'N_multimapping', etc.
        """
        all_dfs = []
        gene_col_candidates = ['gene', 'gene_name', 'gene_id', 'Ensembl_ID', 'id']
        quant_col = 'fpkm_unstranded'  # Change this to 'tpm_unstranded' or another if desired
        skip_prefixes = ('N_unmapped', 'N_multimapping', 'N_noFeature', 'N_ambiguous')
        for file_id in file_ids[:30]:  # Limit for processing
            try:
                response = requests.get(f"{self.gdc_api}/data/{file_id}")
                response.raise_for_status()
                text = response.text
                # Skip comment lines (starting with '#')
                lines = text.split('\n')
                data_lines = [line for line in lines if line.strip() and not line.startswith('#')]
                if not data_lines:
                    logger.error(f"No data lines found in expression file {file_id}")
                    continue
                df = pd.read_csv(StringIO('\n'.join(data_lines)), sep='\t', engine='python')
                # Find gene column
                gene_col = None
                for col in df.columns:
                    if col.lower() in [c.lower() for c in gene_col_candidates]:
                        gene_col = col
                        break
                if gene_col is None:
                    gene_col = df.columns[0]
                    logger.warning(f"Falling back to first column '{gene_col}' as gene column in expression file {file_id}")
                # Skip special rows
                df = df[~df[gene_col].astype(str).str.startswith(skip_prefixes)]
                # Extract quantification column
                if quant_col not in df.columns:
                    logger.warning(f"Quantification column '{quant_col}' not found in file {file_id}, skipping file.")
                    continue
                df = df[[gene_col, quant_col]].set_index(gene_col)
                df.columns = [file_id]
                df = df.apply(pd.to_numeric, errors='coerce')
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to process expression file {file_id}: {e}")
        if not all_dfs:
            return pd.DataFrame()
        merged = pd.concat(all_dfs, axis=1, join='outer')
        return merged
    
    def _download_and_process_cnv(self, file_ids: List[str]) -> pd.DataFrame:
        """
        Download and process CNV data
        """
        cnv_data = []
        
        for file_id in file_ids[:20]:  # Limit for processing
            try:
                response = requests.get(f"{self.gdc_api}/data/{file_id}")
                response.raise_for_status()
                
                lines = response.text.split('\n')
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 6:
                            cnv_data.append({
                                'sample_id': parts[0],
                                'chromosome': parts[1],
                                'start': int(parts[2]),
                                'end': int(parts[3]),
                                'num_probes': int(parts[4]),
                                'segment_mean': float(parts[5])
                            })
                
            except Exception as e:
                logger.error(f"Failed to process CNV file {file_id}: {e}")
        
        return pd.DataFrame(cnv_data)
    
    def _download_and_process_clinical(self, file_ids: List[str]) -> pd.DataFrame:
        """
        Download and process clinical data
        """
        clinical_data = []
        
        for file_id in file_ids[:10]:  # Limit for processing
            try:
                response = requests.get(f"{self.gdc_api}/data/{file_id}")
                response.raise_for_status()
                
                # Parse clinical data
                lines = response.text.split('\n')
                headers = lines[0].split('\t')
                
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) == len(headers):
                            clinical_data.append(dict(zip(headers, parts)))
                
            except Exception as e:
                logger.error(f"Failed to process clinical file {file_id}: {e}")
        
        return pd.DataFrame(clinical_data)
    
    def create_comprehensive_dataset(self) -> Dict:
        """
        Create comprehensive dataset combining all sources
        """
        logger.info("Creating comprehensive multi-omics dataset...")
        
        # Collect data from all sources
        tcga_data = self.collect_tcga_brca_data()
        cptac_data = self.collect_cptac_brca_data()
        
        # Combine and integrate data
        integrated_data = self._integrate_multi_omics_data(tcga_data, cptac_data)
        
        # Save integrated dataset
        output_file = self.data_dir / "comprehensive_brca_data.pkl"
        integrated_data.to_pickle(output_file)
        
        logger.info(f"Comprehensive dataset saved to {output_file}")
        return integrated_data
    
    def _integrate_multi_omics_data(self, tcga_data: Dict, cptac_data: Dict) -> pd.DataFrame:
        """
        Integrate multi-omics data from different sources
        """
        # Start with clinical data as base
        integrated = tcga_data['clinical'].copy()
        
        # Add mutation features
        if not tcga_data['mutations'].empty:
            mutation_features = self._create_mutation_features(tcga_data['mutations'])
            integrated = integrated.merge(mutation_features, on='sample_id', how='left')
        
        # Add expression features
        if not tcga_data['expression'].empty:
            expression_features = self._create_expression_features(tcga_data['expression'])
            integrated = integrated.merge(expression_features, on='sample_id', how='left')
        
        # Add CNV features
        if not tcga_data['cnv'].empty:
            cnv_features = self._create_cnv_features(tcga_data['cnv'])
            integrated = integrated.merge(cnv_features, on='sample_id', how='left')
        
        # Add protein features
        if 'protein_abundance' in cptac_data and not cptac_data['protein_abundance'].empty:
            protein_features = self._create_protein_features(cptac_data['protein_abundance'])
            integrated = integrated.merge(protein_features, on='sample_id', how='left')
        
        return integrated
    
    def _create_mutation_features(self, mutations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mutation-based features
        """
        # Count mutations per gene per sample
        mutation_counts = mutations_df.groupby(['sample_id', 'gene']).size().reset_index(name='mutation_count')
        
        # Pivot to create gene-specific features
        mutation_features = mutation_counts.pivot(index='sample_id', columns='gene', values='mutation_count').fillna(0)
        
        # Add summary features
        mutation_features['total_mutations'] = mutation_features.sum(axis=1)
        mutation_features['mutated_genes'] = (mutation_features > 0).sum(axis=1)
        
        return mutation_features.reset_index()
    
    def _create_expression_features(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create expression-based features
        """
        # For now, return basic expression features
        # In practice, this would process the full expression matrix
        return expression_df
    
    def _create_cnv_features(self, cnv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create CNV-based features
        """
        # Calculate CNV burden per sample
        cnv_features = cnv_df.groupby('sample_id').agg({
            'segment_mean': ['mean', 'std', 'count'],
            'num_probes': 'sum'
        }).reset_index()
        
        cnv_features.columns = ['sample_id', 'cnv_mean', 'cnv_std', 'cnv_segments', 'total_probes']
        
        return cnv_features
    
    def _create_protein_features(self, protein_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create protein abundance features
        """
        # Process protein abundance data
        # This would involve normalizing and creating protein-specific features
        return protein_df

def main():
    """
    Main function to run comprehensive data collection
    """
    collector = EnhancedDataCollector()
    
    # Create comprehensive dataset
    comprehensive_data = collector.create_comprehensive_dataset()
    
    print(f"Comprehensive dataset created with {len(comprehensive_data)} samples")
    print(f"Features: {comprehensive_data.shape[1]}")
    print(f"Sample columns: {list(comprehensive_data.columns[:10])}")

if __name__ == "__main__":
    main() 