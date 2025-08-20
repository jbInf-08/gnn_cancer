import os
import pandas as pd
import numpy as np
import gzip
import logging
from pathlib import Path
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_expression_matrix(expression_dir="data/raw/expression_patients"):
    """Build expression matrix from patient-level files."""
    logger.info("Building expression matrix from patient-level data...")
    
    expression_files = list(Path(expression_dir).glob("*_expression.tsv.gz"))
    logger.info(f"Found {len(expression_files)} expression files")
    
    if not expression_files:
        raise FileNotFoundError(f"No expression files found in {expression_dir}")
    
    # Load first file to get gene structure
    sample_data = {}
    gene_ids = set()
    
    for i, file_path in enumerate(expression_files):
        if i % 5 == 0:
            logger.info(f"Processing expression file {i+1}/{len(expression_files)}")
        
        patient_id = file_path.stem.replace('_expression', '')
        
        try:
            # Always try plain text first since files have .gz extension but are plain text
            df = pd.read_csv(file_path, sep='\t', comment='#', compression=None)
            
            # Check if we have the expected columns
            if 'gene_id' in df.columns and 'tpm_unstranded' in df.columns:
                # Use gene_id and tpm_unstranded
                gene_expr = df.set_index('gene_id')['tpm_unstranded']
                sample_data[patient_id] = gene_expr
                gene_ids.update(gene_expr.index)
            elif 'gene_id' in df.columns and 'expected_count' in df.columns:
                # Use gene_id and expected_count
                gene_expr = df.set_index('gene_id')['expected_count']
                sample_data[patient_id] = gene_expr
                gene_ids.update(gene_expr.index)
            else:
                logger.warning(f"Unexpected columns in {file_path}: {df.columns.tolist()}")
                continue
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if not sample_data:
        raise ValueError("No valid expression data found")
    
    # Create matrix
    gene_ids = sorted(list(gene_ids))
    expression_matrix = pd.DataFrame(index=gene_ids, columns=list(sample_data.keys()))
    
    for patient_id, gene_expr in sample_data.items():
        expression_matrix[patient_id] = gene_expr.reindex(gene_ids, fill_value=0)
    
    logger.info(f"Expression matrix shape: {expression_matrix.shape}")
    logger.info(f"Expression matrix info:")
    logger.info(f"  - Samples: {expression_matrix.shape[1]}")
    logger.info(f"  - Genes: {expression_matrix.shape[0]}")
    logger.info(f"  - Non-zero values: {(expression_matrix != 0).sum().sum()}")
    
    return expression_matrix

def build_cnv_matrix(cnv_dir="data/raw/cnv_patients"):
    """Build CNV matrix from patient-level files."""
    logger.info("Building CNV matrix from patient-level data...")
    
    cnv_files = list(Path(cnv_dir).glob("*_cnv.tsv.gz"))
    logger.info(f"Found {len(cnv_files)} CNV files")
    
    if not cnv_files:
        raise FileNotFoundError(f"No CNV files found in {cnv_dir}")
    
    # Load first file to understand structure
    sample_data = {}
    all_segments = set()
    
    for i, file_path in enumerate(cnv_files):
        if i % 5 == 0:
            logger.info(f"Processing CNV file {i+1}/{len(cnv_files)}")
        
        patient_id = file_path.stem.replace('_cnv', '')
        
        try:
            # Always try plain text first since files have .gz extension but are plain text
            df = pd.read_csv(file_path, sep='\t', comment='#', compression=None)
            
            # Check if we have the expected columns
            if 'Chromosome' in df.columns and 'Start' in df.columns and 'End' in df.columns and 'Segment_Mean' in df.columns:
                # Create segment identifier
                df['segment_id'] = df['Chromosome'].astype(str) + ':' + df['Start'].astype(str) + '-' + df['End'].astype(str)
                segment_cnv = df.set_index('segment_id')['Segment_Mean']
                sample_data[patient_id] = segment_cnv
                all_segments.update(segment_cnv.index)
            else:
                logger.warning(f"Unexpected columns in {file_path}: {df.columns.tolist()}")
                continue
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if not sample_data:
        raise ValueError("No valid CNV data found")
    
    # Create matrix
    all_segments = sorted(list(all_segments))
    cnv_matrix = pd.DataFrame(index=all_segments, columns=list(sample_data.keys()))
    
    for patient_id, segment_cnv in sample_data.items():
        cnv_matrix[patient_id] = segment_cnv.reindex(all_segments, fill_value=0)
    
    logger.info(f"CNV matrix shape: {cnv_matrix.shape}")
    logger.info(f"CNV matrix info:")
    logger.info(f"  - Samples: {cnv_matrix.shape[1]}")
    logger.info(f"  - Segments: {cnv_matrix.shape[0]}")
    logger.info(f"  - Non-zero values: {(cnv_matrix != 0).sum().sum()}")
    
    return cnv_matrix

def save_matrices(expression_matrix, cnv_matrix, output_dir="data/processed"):
    """Save the matrices to files."""
    logger.info("Saving matrices...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save expression matrix
    expression_file = output_path / "expression_matrix_patients.csv"
    expression_matrix.to_csv(expression_file)
    logger.info(f"Saved expression matrix to {expression_file}")
    
    # Save CNV matrix
    cnv_file = output_path / "cnv_matrix_patients.csv"
    cnv_matrix.to_csv(cnv_file)
    logger.info(f"Saved CNV matrix to {cnv_file}")
    
    # Save metadata
    metadata = {
        'expression_samples': list(expression_matrix.columns),
        'cnv_samples': list(cnv_matrix.columns),
        'expression_genes': list(expression_matrix.index),
        'cnv_segments': list(cnv_matrix.index),
        'common_samples': list(set(expression_matrix.columns) & set(cnv_matrix.columns))
    }
    
    metadata_file = output_path / "patient_matrices_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {metadata_file}")
    
    return metadata

def main():
    logger.info("Starting patient-level matrix building...")
    
    try:
        # Build expression matrix
        expression_matrix = build_expression_matrix()
        
        # Build CNV matrix
        cnv_matrix = build_cnv_matrix()
        
        # Save matrices
        metadata = save_matrices(expression_matrix, cnv_matrix)
        
        # Report results
        logger.info("Matrix building complete!")
        logger.info(f"Expression matrix: {expression_matrix.shape}")
        logger.info(f"CNV matrix: {cnv_matrix.shape}")
        logger.info(f"Common samples: {len(metadata['common_samples'])}")
        
        if metadata['common_samples']:
            logger.info("Sample common samples:")
            for sample in metadata['common_samples'][:5]:
                logger.info(f"  - {sample}")
        
    except Exception as e:
        logger.error(f"Error in matrix building: {e}")
        raise

if __name__ == "__main__":
    main() 