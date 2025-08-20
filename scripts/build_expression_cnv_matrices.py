import os
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_uuid_to_barcode_mapping():
    """Load the UUID to TCGA barcode mapping."""
    mapping_file = "uuid_to_barcode.csv"
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file {mapping_file} not found.")
    
    mapping = pd.read_csv(mapping_file)
    # Create UUID to barcode mapping
    uuid_to_barcode = dict(zip(mapping['uuid'], mapping['barcode']))
    
    logger.info(f"Loaded {len(uuid_to_barcode)} UUID to barcode mappings")
    return uuid_to_barcode

def build_expression_matrix(expression_dir="data/raw/expression", uuid_to_barcode=None):
    """Build expression matrix from per-sample files."""
    logger.info("Building expression matrix...")
    
    expression_files = list(Path(expression_dir).glob("*.tsv.gz"))
    logger.info(f"Found {len(expression_files)} expression files")
    
    if not expression_files:
        raise FileNotFoundError(f"No expression files found in {expression_dir}")
    
    # Load first file to get gene structure
    sample_data = {}
    gene_ids = set()
    
    for i, file_path in enumerate(expression_files):
        if i % 50 == 0:
            logger.info(f"Processing expression file {i+1}/{len(expression_files)}")
        
        try:
            # Extract UUID from filename
            uuid = file_path.stem.replace('.tsv', '')
            
            # Read the file as plain text (files have .gz extension but are not compressed)
            with open(file_path, 'rt') as f:
                df = pd.read_csv(f, sep='\t', comment='#')
            
            # Extract gene_id and tpm values
            if 'gene_id' in df.columns and 'tpm_unstranded' in df.columns:
                gene_expr = dict(zip(df['gene_id'], df['tpm_unstranded']))
                sample_data[uuid] = gene_expr
                gene_ids.update(gene_expr.keys())
            else:
                logger.warning(f"Unexpected columns in {file_path}: {list(df.columns)}")
                
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue
    
    # Build matrix
    gene_ids = sorted(list(gene_ids))
    logger.info(f"Building matrix with {len(gene_ids)} genes and {len(sample_data)} samples")
    
    # Create DataFrame
    expression_matrix = pd.DataFrame.from_dict(sample_data, orient='index', columns=gene_ids)
    logger.info(f"Expression matrix shape: {expression_matrix.shape}")
    
    # Convert UUIDs to TCGA barcodes if mapping is available
    if uuid_to_barcode:
        logger.info("Converting expression sample IDs from UUIDs to TCGA barcodes...")
        # Map UUIDs to barcodes, keep original if not found
        new_index = []
        for uuid in expression_matrix.index:
            barcode = uuid_to_barcode.get(uuid, uuid)
            new_index.append(barcode)
        expression_matrix.index = new_index
        logger.info(f"Expression matrix after conversion: {expression_matrix.shape}")
    
    return expression_matrix

def build_cnv_matrix(cnv_dir="data/raw/cnv", uuid_to_barcode=None):
    logger.info("Building segment-level CNV matrix...")
    cnv_files = list(Path(cnv_dir).glob("*.tsv.gz"))
    logger.info(f"Found {len(cnv_files)} CNV files")
    if not cnv_files:
        raise FileNotFoundError(f"No CNV files found in {cnv_dir}")
    sample_data = {}
    segment_ids = set()
    for i, file_path in enumerate(cnv_files):
        if i % 50 == 0:
            logger.info(f"Processing CNV file {i+1}/{len(cnv_files)}")
        uuid = file_path.stem.replace('.tsv', '')
        df = None
        # Always try gzip first, then always try plain text if gzip fails
        try:
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, sep='\t', comment='#')
        except Exception:
            logger.info(f"File {file_path} is not gzipped or failed to read as gzip, trying as plain text...")
            try:
                with open(file_path, 'rt') as f:
                    df = pd.read_csv(f, sep='\t', comment='#')
            except Exception as e:
                logger.warning(f"Failed to read CNV file {file_path} as plain text: {e}")
                continue
        if df is None or df.empty:
            logger.warning(f"CNV file {file_path} is empty or could not be read.")
            continue
        # Build segment IDs and collect Segment_Mean
        for _, row in df.iterrows():
            seg_id = f"chr{row['Chromosome']}:{row['Start']}-{row['End']}"
            segment_ids.add(seg_id)
            if uuid not in sample_data:
                sample_data[uuid] = {}
            sample_data[uuid][seg_id] = row['Segment_Mean']
    segment_ids = sorted(segment_ids)
    cnv_matrix = pd.DataFrame.from_dict(sample_data, orient='index', columns=segment_ids)
    logger.info(f"Segment-level CNV matrix shape: {cnv_matrix.shape}")
    
    # Convert UUIDs to TCGA barcodes if mapping is available
    if uuid_to_barcode:
        logger.info("Converting CNV sample IDs from UUIDs to TCGA barcodes...")
        # Map UUIDs to barcodes, keep original if not found
        new_index = []
        for uuid in cnv_matrix.index:
            barcode = uuid_to_barcode.get(uuid, uuid)
            new_index.append(barcode)
        cnv_matrix.index = new_index
        logger.info(f"CNV matrix after conversion: {cnv_matrix.shape}")
    
    return cnv_matrix

def align_expression_cnv_data(expression_matrix, cnv_matrix):
    """Align expression and CNV data to common samples."""
    logger.info("Aligning expression and CNV data...")
    
    # Debug: Show sample IDs from each matrix
    logger.info(f"Expression sample IDs (first 5): {list(expression_matrix.index[:5])}")
    logger.info(f"CNV sample IDs (first 5): {list(cnv_matrix.index[:5])}")
    
    # Find common samples
    common_samples = set(expression_matrix.index) & set(cnv_matrix.index)
    logger.info(f"Common samples (expression + CNV): {len(common_samples)}")
    
    if len(common_samples) == 0:
        logger.error("No common samples found between expression and CNV data")
        return None, None
    
    # Align matrices to common samples
    aligned_expression = expression_matrix.loc[list(common_samples)]
    aligned_cnv = cnv_matrix.loc[list(common_samples)]
    
    logger.info(f"Aligned matrices - Expression: {aligned_expression.shape}, CNV: {aligned_cnv.shape}")
    
    return aligned_expression, aligned_cnv

def save_aligned_data(expression_matrix, cnv_matrix, output_dir="data/processed"):
    """Save aligned expression and CNV data."""
    os.makedirs(output_dir, exist_ok=True)
    
    expression_matrix.to_csv(f"{output_dir}/aligned_expression_cnv_expression.csv")
    cnv_matrix.to_csv(f"{output_dir}/aligned_expression_cnv_cnv.csv")
    
    # Also save a summary
    summary = {
        'expression_samples': len(expression_matrix),
        'expression_genes': len(expression_matrix.columns),
        'cnv_samples': len(cnv_matrix),
        'cnv_segments': len(cnv_matrix.columns),
        'common_samples': len(expression_matrix.index)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/expression_cnv_summary.csv", index=False)
    
    logger.info(f"Saved aligned data to {output_dir}/")
    logger.info(f"Summary: {summary}")

def main():
    logger.info("Starting expression and CNV matrix construction and alignment...")
    
    # Load UUID to barcode mapping
    try:
        uuid_to_barcode = load_uuid_to_barcode_mapping()
        logger.info(f"Loaded {len(uuid_to_barcode)} UUID to barcode mappings")
    except FileNotFoundError:
        logger.warning("UUID to barcode mapping not found. Using UUIDs as sample IDs.")
        uuid_to_barcode = None
    
    # Build matrices
    expression_matrix = build_expression_matrix(uuid_to_barcode=uuid_to_barcode)
    cnv_matrix = build_cnv_matrix(uuid_to_barcode=uuid_to_barcode)
    
    # Align expression and CNV data
    aligned_expression, aligned_cnv = align_expression_cnv_data(expression_matrix, cnv_matrix)
    
    if aligned_expression is not None and aligned_cnv is not None:
        # Save aligned data
        save_aligned_data(aligned_expression, aligned_cnv)
        logger.info("Expression and CNV matrix construction and alignment complete!")
    else:
        logger.error("Failed to align expression and CNV data")

if __name__ == "__main__":
    main() 