#!/usr/bin/env python3
"""
Delete ALL synthetic data and replace with real data
This script removes synthetic data files and ensures only real data is used
"""

import os
import shutil
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_synthetic_data():
    """Delete all synthetic data files"""
    logger.info("Starting deletion of synthetic data...")
    
    # Files to delete (synthetic data)
    synthetic_files = [
        "generate_comprehensive_data.py",
        "generate_sample_data.py", 
        "data/processed/BRCA_comprehensive_data.pt",
        "data/processed/BRCA_data.pt",
        "data/processed/BRCA_sample_dataset.pt",
        "data/processed/expression_data.csv",
        "data/processed/mutation_data.csv",
        "data/processed/labels.npy",
        "data/processed/graph.pkl",
        "data/enhanced/torch_geometric_data.pt",
        "data/enhanced/patient_graph.pkl",
        "data/enhanced/feature_scaler.pkl",
        "data/enhanced/dataset_summary.pkl"
    ]
    
    # Directories to clean
    synthetic_dirs = [
        "data/processed/brca1",
        "data/processed/cancer_drivers", 
        "data/processed/uci"
    ]
    
    deleted_files = []
    deleted_dirs = []
    
    # Delete synthetic files
    for file_path in synthetic_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                logger.info(f"Deleted synthetic file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
    
    # Delete synthetic directories
    for dir_path in synthetic_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                deleted_dirs.append(dir_path)
                logger.info(f"Deleted synthetic directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to delete {dir_path}: {e}")
    
    # Clean up corrupted files
    corrupted_files = [
        "data/raw/tcga/BRCA/clinical/TCGA-BRCA.survival.tsv",
        "data/processed/aligned_mutation.csv",
        "data/processed/aligned_expression.csv", 
        "data/processed/aligned_cnv.csv"
    ]
    
    for file_path in corrupted_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                logger.info(f"Deleted corrupted file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
    
    # Create summary
    summary = {
        "deleted_files": deleted_files,
        "deleted_directories": deleted_dirs,
        "total_files_deleted": len(deleted_files),
        "total_dirs_deleted": len(deleted_dirs)
    }
    
    # Save deletion summary
    with open("synthetic_data_deletion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Deletion complete! Deleted {len(deleted_files)} files and {len(deleted_dirs)} directories")
    logger.info("Summary saved to synthetic_data_deletion_summary.json")
    
    return summary

def verify_real_data():
    """Verify that real data files exist"""
    logger.info("Verifying real data files...")
    
    real_data_files = [
        "data/processed/expression_matrix_patients.csv",
        "data/processed/cnv_matrix_patients.csv", 
        "data/processed/patient_matrices_metadata.pkl"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in real_data_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            logger.info(f"✓ Real data file exists: {file_path}")
        else:
            missing_files.append(file_path)
            logger.warning(f"✗ Missing real data file: {file_path}")
    
    # Check file sizes
    for file_path in existing_files:
        try:
            size = os.path.getsize(file_path)
            if size > 1000:  # More than 1KB
                logger.info(f"  - {file_path}: {size:,} bytes")
            else:
                logger.warning(f"  - {file_path}: {size:,} bytes (suspiciously small)")
        except Exception as e:
            logger.error(f"  - {file_path}: Error checking size - {e}")
    
    summary = {
        "existing_real_data": existing_files,
        "missing_real_data": missing_files,
        "total_existing": len(existing_files),
        "total_missing": len(missing_files)
    }
    
    logger.info(f"Real data verification complete: {len(existing_files)} existing, {len(missing_files)} missing")
    
    return summary

def main():
    """Main function"""
    logger.info("=== SYNTHETIC DATA DELETION AND REAL DATA VERIFICATION ===")
    
    # Step 1: Delete synthetic data
    deletion_summary = delete_synthetic_data()
    
    # Step 2: Verify real data
    verification_summary = verify_real_data()
    
    # Final summary
    logger.info("\n=== FINAL SUMMARY ===")
    logger.info(f"Synthetic files deleted: {deletion_summary['total_files_deleted']}")
    logger.info(f"Synthetic directories deleted: {deletion_summary['total_dirs_deleted']}")
    logger.info(f"Real data files verified: {verification_summary['total_existing']}")
    logger.info(f"Missing real data files: {verification_summary['total_missing']}")
    
    if verification_summary['total_missing'] > 0:
        logger.warning("Some real data files are missing. Run the data processing scripts first.")
    else:
        logger.info("All real data files are present and ready for integration!")

if __name__ == "__main__":
    main() 