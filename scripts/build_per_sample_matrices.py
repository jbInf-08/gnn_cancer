import os
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import logging
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_uuid_to_barcode_mapping():
    """Load the UUID to TCGA barcode mapping."""
    mapping_file = "uuid_to_barcode.csv"
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file {mapping_file} not found. Run build_uuid_to_barcode_mapping.py first.")
    
    mapping = pd.read_csv(mapping_file)
    # Create UUID to barcode mapping
    uuid_to_barcode = dict(zip(mapping['uuid'], mapping['barcode']))
    
    logger.info(f"Loaded {len(uuid_to_barcode)} UUID to barcode mappings")
    return uuid_to_barcode

def build_expression_matrix(expression_dir="data/raw/expression", uuid_to_barcode=None, save_path=None):
    """Build expression matrix from per-sample files. Optionally save to disk and skip if file exists."""
    if save_path is not None and os.path.exists(save_path):
        logger.info(f"Expression matrix file already exists at {save_path}, skipping computation.")
        return pd.read_csv(save_path, index_col=0)
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
    if save_path is not None:
        expression_matrix.to_csv(save_path)
        logger.info(f"Saved expression matrix to {save_path}")
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

def build_cnv_matrix_chunked_sparse(cnv_dir="data/raw/cnv", uuid_to_barcode=None, chunk_size=100):
    logger.info("Building segment-level CNV matrix in chunks with sparse and float32...")
    cnv_files = list(Path(cnv_dir).glob("*.tsv.gz"))
    logger.info(f"Found {len(cnv_files)} CNV files")
    if not cnv_files:
        raise FileNotFoundError(f"No CNV files found in {cnv_dir}")

    segment_ids = set()
    # First pass: collect all segment IDs
    for file_path in cnv_files:
        try:
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, sep='\t', comment='#')
        except Exception:
            try:
                with open(file_path, 'rt') as f:
                    df = pd.read_csv(f, sep='\t', comment='#')
            except Exception:
                continue
        for _, row in df.iterrows():
            seg_id = f"chr{row['Chromosome']}:{row['Start']}-{row['End']}"
            segment_ids.add(seg_id)
    segment_ids = sorted(segment_ids)
    logger.info(f"Total unique segments: {len(segment_ids)}")

    # Second pass: process in chunks
    for i in range(0, len(cnv_files), chunk_size):
        chunk_files = cnv_files[i:i+chunk_size]
        out_path = f"data/processed/cnv_matrix_chunk_{i//chunk_size}.csv"
        if os.path.exists(out_path):
            logger.info(f"Chunk {i//chunk_size} already exists at {out_path}, skipping.")
            continue
        sample_data = {}
        for file_path in chunk_files:
            uuid = file_path.stem.replace('.tsv', '')
            try:
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f, sep='\t', comment='#')
            except Exception:
                try:
                    with open(file_path, 'rt') as f:
                        df = pd.read_csv(f, sep='\t', comment='#')
                except Exception:
                    continue
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                seg_id = f"chr{row['Chromosome']}:{row['Start']}-{row['End']}"
                if uuid not in sample_data:
                    sample_data[uuid] = {}
                sample_data[uuid][seg_id] = np.float32(row['Segment_Mean'])
        # Create sparse DataFrame
        cnv_matrix_chunk = pd.DataFrame.from_dict(sample_data, orient='index', columns=segment_ids, dtype=np.float32)
        cnv_matrix_chunk = cnv_matrix_chunk.astype(pd.SparseDtype("float32", np.nan))
        # Convert UUIDs to TCGA barcodes if mapping is available
        if uuid_to_barcode:
            new_index = [uuid_to_barcode.get(uuid, uuid) for uuid in cnv_matrix_chunk.index]
            cnv_matrix_chunk.index = new_index
        # Save each chunk to disk
        cnv_matrix_chunk.to_csv(out_path)
        logger.info(f"Saved sparse chunk {i//chunk_size} to {out_path}")
    logger.info("All chunks processed and saved as sparse DataFrames.")

def combine_cnv_chunks(processed_dir="data/processed", output_file="data/processed/cnv_matrix_combined.csv"):
    """Combine all CNV matrix chunks into a single sparse DataFrame using memory-efficient approach."""
    if os.path.exists(output_file):
        logger.info(f"Combined CNV matrix already exists at {output_file}, skipping.")
        return
    
    logger.info(f"Combining CNV matrix chunks from {processed_dir} using memory-efficient approach...")
    chunk_files = sorted(glob.glob(f"{processed_dir}/cnv_matrix_chunk_*.csv"))
    if not chunk_files:
        logger.warning(f"No chunk files found in {processed_dir}")
        return
    
    # First pass: collect all sample IDs and segment IDs to determine final matrix dimensions
    logger.info("Collecting sample and segment IDs from all chunks...")
    all_sample_ids = set()
    all_segment_ids = set()
    
    for chunk_file in chunk_files:
        logger.info(f"Scanning {chunk_file} for IDs...")
        # Read just the header to get column names (segment IDs)
        chunk_df = pd.read_csv(chunk_file, index_col=0, nrows=0)
        all_segment_ids.update(chunk_df.columns)
        # Read just the index to get sample IDs
        chunk_df = pd.read_csv(chunk_file, index_col=0, usecols=[0])
        all_sample_ids.update(chunk_df.index)
    
    logger.info(f"Total unique samples: {len(all_sample_ids)}")
    logger.info(f"Total unique segments: {len(all_segment_ids)}")
    
    # Sort IDs for consistent ordering
    all_sample_ids = sorted(list(all_sample_ids))
    all_segment_ids = sorted(list(all_segment_ids))
    
    # Create a mapping for efficient lookup
    sample_id_to_idx = {sample_id: idx for idx, sample_id in enumerate(all_sample_ids)}
    segment_id_to_idx = {segment_id: idx for idx, segment_id in enumerate(all_segment_ids)}
    
    # Initialize sparse matrix using scipy.sparse for maximum memory efficiency
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # Pre-allocate sparse matrix with estimated non-zero elements
    # Estimate: assume 10% of matrix is non-zero (typical for CNV data)
    estimated_nnz = int(len(all_sample_ids) * len(all_segment_ids) * 0.1)
    logger.info(f"Estimated non-zero elements: {estimated_nnz}")
    
    # Use incremental building with lists
    rows, cols, data = [], [], []
    
    # Process each chunk and add to sparse matrix
    for chunk_file in chunk_files:
        logger.info(f"Processing {chunk_file}...")
        
        # Load chunk in chunks to avoid memory issues
        chunk_reader = pd.read_csv(chunk_file, index_col=0, chunksize=50)
        
        for chunk_df in chunk_reader:
            # Convert to sparse format immediately
            chunk_sparse = chunk_df.astype(pd.SparseDtype("float32", np.nan))
            
            # Get non-zero values and their positions
            for sample_idx, sample_id in enumerate(chunk_df.index):
                if sample_id in sample_id_to_idx:
                    global_sample_idx = sample_id_to_idx[sample_id]
                    
                    for col_idx, segment_id in enumerate(chunk_df.columns):
                        if segment_id in segment_id_to_idx:
                            global_segment_idx = segment_id_to_idx[segment_id]
                            value = chunk_sparse.iloc[sample_idx, col_idx]
                            
                            # Only add non-zero, non-NaN values
                            if pd.notna(value) and value != 0:
                                rows.append(global_sample_idx)
                                cols.append(global_segment_idx)
                                data.append(float(value))
    
    logger.info(f"Collected {len(data)} non-zero elements")
    
    # Create sparse matrix
    logger.info("Creating final sparse matrix...")
    sparse_matrix = csr_matrix((data, (rows, cols)), 
                              shape=(len(all_sample_ids), len(all_segment_ids)),
                              dtype=np.float32)
    
    # Convert to pandas DataFrame with sparse dtype
    logger.info("Converting to pandas DataFrame...")
    combined_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, 
                                                   index=all_sample_ids,
                                                   columns=all_segment_ids)
    
    # Convert to sparse dtype for consistency
    combined_df = combined_df.astype(pd.SparseDtype("float32", 0.0))
    
    logger.info(f"Combined CNV matrix shape: {combined_df.shape}")
    logger.info(f"Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save using sparse format
    combined_df.to_csv(output_file)
    logger.info(f"Saved combined CNV matrix to {output_file}")

def filter_genes_by_mutation_frequency(mutation_matrix, threshold=0.01):
    """Remove genes (columns) with mutation frequency < threshold."""
    freq = (mutation_matrix > 0).sum(axis=0) / mutation_matrix.shape[0]
    keep_genes = freq[freq >= threshold].index
    filtered = mutation_matrix[keep_genes]
    logger.info(f"Filtered genes by mutation frequency <{threshold}: {mutation_matrix.shape[1]} -> {filtered.shape[1]}")
    return filtered

def impute_missing(df):
    """Impute missing values: median for continuous, mode for categorical."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            mode = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            df[col] = df[col].fillna(mode)
    return df

def minmax_normalize(df):
    """Min-max scale all numeric columns."""
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def balance_classes(aligned_data, label_col='mutation', method='oversample'):
    """Balance classes by oversampling positives or downsampling negatives."""
    # Assume 'mutation' is a binary matrix: samples x genes
    # For each gene, balance positives/negatives
    mut = aligned_data['mutation']
    pos_idx = mut.index[(mut.sum(axis=1) > 0)]
    neg_idx = mut.index[(mut.sum(axis=1) == 0)]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    logger.info(f"Class balance before: {n_pos} positives, {n_neg} negatives")
    if n_pos == 0 or n_neg == 0:
        logger.warning("No positives or negatives to balance!")
        return aligned_data
    if method == 'oversample':
        # Oversample positives
        pos_samples = resample(mut.loc[pos_idx], replace=True, n_samples=n_neg, random_state=42)
        new_mut = pd.concat([mut.loc[neg_idx], pos_samples])
    else:
        # Downsample negatives
        neg_samples = resample(mut.loc[neg_idx], replace=False, n_samples=n_pos, random_state=42)
        new_mut = pd.concat([mut.loc[pos_idx], neg_samples])
    # Shuffle
    new_mut = new_mut.sample(frac=1, random_state=42)
    # Align other omics
    new_expr = aligned_data['expression'].loc[new_mut.index]
    new_cnv = aligned_data['cnv'].loc[new_mut.index]
    logger.info(f"Class balance after: {sum(new_mut.sum(axis=1)>0)} positives, {sum(new_mut.sum(axis=1)==0)} negatives")
    return {'expression': new_expr, 'cnv': new_cnv, 'mutation': new_mut}

def load_mutation_matrix(mutation_file="data/raw/BRCA_mutation.maf.gz", save_path=None):
    """Load and process mutation matrix. Optionally save to disk and skip if file exists."""
    if save_path is not None and os.path.exists(save_path):
        logger.info(f"Mutation matrix file already exists at {save_path}, skipping computation.")
        return pd.read_csv(save_path, index_col=0)
    logger.info("Loading mutation matrix...")
    
    if not os.path.exists(mutation_file):
        raise FileNotFoundError(f"Mutation file {mutation_file} not found")
    
    # Read mutation file
    try:
        with gzip.open(mutation_file, 'rt') as f:
            mutations = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
    except UnicodeDecodeError:
        # Try as uncompressed
        mutations = pd.read_csv(mutation_file, sep='\t', comment='#', low_memory=False)
    
    # Create sample-by-gene mutation matrix
    if 'Hugo_Symbol' in mutations.columns and 'Tumor_Sample_Barcode' in mutations.columns:
        # Count mutations per gene per sample
        mutation_matrix = mutations.groupby(['Hugo_Symbol', 'Tumor_Sample_Barcode']).size().unstack(fill_value=0)
        # Transpose to get samples as rows and genes as columns
        mutation_matrix = mutation_matrix.T
        # Filter genes by mutation frequency <1%
        mutation_matrix = filter_genes_by_mutation_frequency(mutation_matrix, threshold=0.01)
        logger.info(f"Mutation matrix shape after filtering: {mutation_matrix.shape}")
        if save_path is not None:
            mutation_matrix.to_csv(save_path)
            logger.info(f"Saved mutation matrix to {save_path}")
        return mutation_matrix
    else:
        logger.warning("Mutation file does not contain expected columns")
        return pd.DataFrame()

def align_omics_data(expression_matrix, cnv_matrix, mutation_matrix):
    """Align all omics data to common samples and genes."""
    logger.info("Aligning omics data...")
    
    # Debug: Show sample IDs from each matrix
    logger.info(f"Expression sample IDs (first 5): {list(expression_matrix.index[:5])}")
    logger.info(f"CNV sample IDs (first 5): {list(cnv_matrix.index[:5])}")
    logger.info(f"Mutation sample IDs (first 5): {list(mutation_matrix.index[:5])}")
    
    # Find common samples across all omics
    common_samples = set(expression_matrix.index) & set(cnv_matrix.index) & set(mutation_matrix.index)
    logger.info(f"Common samples across all omics: {len(common_samples)}")
    
    if len(common_samples) == 0:
        logger.warning("No common samples found across all omics data")
        # Try with just expression and CNV
        common_samples = set(expression_matrix.index) & set(cnv_matrix.index)
        logger.info(f"Common samples (expression + CNV): {len(common_samples)}")
    
    # Get common genes (for expression and CNV)
    expr_genes = set(expression_matrix.columns)
    cnv_genes = set(cnv_matrix.columns)
    common_genes = expr_genes.intersection(cnv_genes)
    logger.info(f"Common genes (expression + CNV): {len(common_genes)}")
    
    # Align matrices
    aligned_expression = expression_matrix.loc[list(common_samples), list(common_genes)]
    aligned_cnv = cnv_matrix.loc[list(common_samples), list(common_genes)]
    
    if not mutation_matrix.empty:
        # For mutation matrix, we need to align to common samples and genes
        mutation_genes = set(mutation_matrix.columns)
        common_mutation_genes = common_genes.intersection(mutation_genes)
        aligned_mutation = mutation_matrix.loc[list(common_samples), list(common_mutation_genes)]
        # Fill missing genes with zeros
        missing_genes = common_genes - mutation_genes
        if missing_genes:
            missing_df = pd.DataFrame(0, index=aligned_mutation.index, columns=list(missing_genes))
            aligned_mutation = pd.concat([aligned_mutation, missing_df], axis=1)
            aligned_mutation = aligned_mutation[list(common_genes)]  # Reorder columns
    else:
        aligned_mutation = pd.DataFrame(index=common_samples, columns=common_genes, data=0)
    
    # Impute missing values
    aligned_expression = impute_missing(aligned_expression)
    aligned_cnv = impute_missing(aligned_cnv)
    aligned_mutation = impute_missing(aligned_mutation)
    # Normalize features
    aligned_expression = minmax_normalize(aligned_expression)
    aligned_cnv = minmax_normalize(aligned_cnv)
    # One-hot encoding hooks (if mutation types/clinical categories present)
    # TODO: Add one-hot encoding for mutation types/clinical categories if available
    # Return
    logger.info(f"Aligned matrices - Expression: {aligned_expression.shape}, CNV: {aligned_cnv.shape}, Mutation: {aligned_mutation.shape}")
    return {'expression': aligned_expression, 'cnv': aligned_cnv, 'mutation': aligned_mutation}

def save_aligned_data(expression_matrix, cnv_matrix, mutation_matrix, output_dir="data/processed"):
    """Save aligned omics data."""
    os.makedirs(output_dir, exist_ok=True)
    expr_path = f"{output_dir}/aligned_expression.csv"
    cnv_path = f"{output_dir}/aligned_cnv.csv"
    mut_path = f"{output_dir}/aligned_mutation.csv"
    if os.path.exists(expr_path):
        logger.info(f"Aligned expression file already exists at {expr_path}, skipping.")
    else:
        expression_matrix.to_csv(expr_path)
        logger.info(f"Saved aligned expression to {expr_path}")
    if os.path.exists(cnv_path):
        logger.info(f"Aligned CNV file already exists at {cnv_path}, skipping.")
    else:
        cnv_matrix.to_csv(cnv_path)
        logger.info(f"Saved aligned CNV to {cnv_path}")
    if os.path.exists(mut_path):
        logger.info(f"Aligned mutation file already exists at {mut_path}, skipping.")
    else:
        mutation_matrix.to_csv(mut_path)
        logger.info(f"Saved aligned mutation to {mut_path}")
    logger.info(f"Checked/created aligned data in {output_dir}/")

def load_cnv_matrix_in_chunks(file_path, chunk_size=1000):
    """Load large CNV matrix in chunks to avoid memory issues."""
    logger.info(f"Loading CNV matrix in chunks from {file_path}...")
    
    # First, get the structure
    sample_ids = pd.read_csv(file_path, index_col=0, nrows=0).index
    segment_ids = pd.read_csv(file_path, index_col=0, nrows=0).columns
    
    logger.info(f"Matrix structure: {len(sample_ids)} samples × {len(segment_ids)} segments")
    
    # Load in chunks and combine
    chunk_list = []
    for i in range(0, len(sample_ids), chunk_size):
        end_idx = min(i + chunk_size, len(sample_ids))
        logger.info(f"Loading chunk {i//chunk_size + 1}: samples {i} to {end_idx-1}")
        
        chunk_df = pd.read_csv(file_path, index_col=0, 
                              skiprows=range(1, i+1), nrows=end_idx-i)
        chunk_df = chunk_df.astype(pd.SparseDtype("float32", 0.0))
        chunk_list.append(chunk_df)
    
    # Combine chunks
    logger.info("Combining chunks...")
    combined_df = pd.concat(chunk_list, axis=0)
    logger.info(f"Final matrix shape: {combined_df.shape}")
    
    return combined_df

def main():
    logger.info("Starting matrix construction and alignment...")

    # Load UUID to barcode mapping
    try:
        uuid_to_barcode = load_uuid_to_barcode_mapping()
        logger.info(f"Loaded {len(uuid_to_barcode)} UUID to barcode mappings")
    except FileNotFoundError:
        logger.warning("UUID to barcode mapping not found. Using UUIDs as sample IDs.")
        uuid_to_barcode = None

    # Build expression matrix with checkpointing
    expression_matrix = build_expression_matrix(
        uuid_to_barcode=uuid_to_barcode,
        save_path="data/processed/expression_matrix.csv"
    )

    # Build CNV matrix in chunks (sparse, float32)
    logger.info("Building CNV matrix in chunks (sparse, float32)...")
    build_cnv_matrix_chunked_sparse(uuid_to_barcode=uuid_to_barcode)
    logger.info("Combining CNV matrix chunks into a single matrix...")
    combine_cnv_chunks()
    logger.info("Loading combined CNV matrix...")
    
    # Load the combined CNV matrix efficiently
    try:
        # Try to load as sparse first
        cnv_matrix = pd.read_csv("data/processed/cnv_matrix_combined.csv", index_col=0)
        cnv_matrix = cnv_matrix.astype(pd.SparseDtype("float32", 0.0))
        logger.info(f"Loaded CNV matrix as sparse: {cnv_matrix.shape}")
    except MemoryError:
        logger.warning("Memory error loading full CNV matrix, loading in chunks...")
        # Fallback: load in chunks and process incrementally
        cnv_matrix = load_cnv_matrix_in_chunks("data/processed/cnv_matrix_combined.csv")

    # Load mutation matrix with checkpointing
    logger.info("Loading mutation matrix...")
    mutation_matrix = load_mutation_matrix(
        save_path="data/processed/mutation_matrix.csv"
    )
    logger.info(f"Mutation matrix shape: {mutation_matrix.shape}")

    # Align omics data
    logger.info("Aligning omics data...")
    aligned_data = align_omics_data(expression_matrix, cnv_matrix, mutation_matrix)
    # Balance classes after alignment
    aligned_data = balance_classes(aligned_data, method='oversample')

    # Save aligned data
    save_aligned_data(aligned_data['expression'], aligned_data['cnv'], aligned_data['mutation'])

    logger.info("Matrix construction and alignment complete!")

if __name__ == "__main__":
    main() 