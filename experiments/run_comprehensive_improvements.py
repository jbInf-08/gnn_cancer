#!/usr/bin/env python3
"""
Run Comprehensive Improvements for Cancer Genomics GNN Project

This script executes all improvements to match/improve results:

1. Increase Sample Size: Download and process more patients (up to 358 available)
2. Use Real Labels: Integrate clinical or mutation status labels for supervised learning
3. Graph Construction: Build gene/protein graph using PPI/pathway data (STRING, KEGG, Reactome)
4. Edge Features: Add biological edge types (PPI, pathway, co-expression) for richer graph structure
5. Model Tuning: Use same hyperparameters and architectures as the paper for fair comparison
"""

import subprocess
import sys
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the result"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Run all comprehensive improvements"""
    logger.info("🚀 Starting Comprehensive Improvements for Cancer Genomics GNN Project")
    
    # Step 1: Increase Sample Size - Download more patients
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Increasing Sample Size (up to 358 patients)")
    logger.info("="*80)
    
    success = run_command(
        "python download_gdc_data.py",
        "Download additional patient data (up to 358 patients)"
    )
    
    if not success:
        logger.error("Failed to download additional patient data")
        return False
    
    # Step 2: Build per-sample matrices with increased data
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Building Per-Sample Matrices with Increased Data")
    logger.info("="*80)
    
    success = run_command(
        "python scripts/build_per_sample_matrices.py",
        "Build per-sample expression and CNV matrices"
    )
    
    if not success:
        logger.error("Failed to build per-sample matrices")
        return False
    
    # Step 3: Use Real Data Only with Comprehensive Networks
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Using Real Data Only with Comprehensive Networks")
    logger.info("="*80)
    
    success = run_command(
        "python scripts/use_real_data_only.py",
        "Integrate real data with comprehensive PPI/pathway networks"
    )
    
    if not success:
        logger.error("Failed to integrate real data")
        return False
    
    # Step 4: Train Enhanced GNN with Paper Hyperparameters
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Training Enhanced GNN with Paper Hyperparameters")
    logger.info("="*80)
    
    success = run_command(
        "python train_enhanced_gnn.py",
        "Train enhanced GNN with paper hyperparameters"
    )
    
    if not success:
        logger.error("Failed to train enhanced GNN")
        return False
    
    # Step 5: Generate Comprehensive Report
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Generating Comprehensive Report")
    logger.info("="*80)
    
    success = run_command(
        "python generate_report.py",
        "Generate comprehensive improvement report"
    )
    
    if not success:
        logger.warning("Failed to generate report (optional step)")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("🎉 COMPREHENSIVE IMPROVEMENTS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    logger.info("✅ Increased sample size to maximum available patients")
    logger.info("✅ Integrated real clinical and mutation labels")
    logger.info("✅ Built comprehensive PPI/pathway networks (STRING, KEGG, Reactome)")
    logger.info("✅ Added biological edge features (PPI, pathway, co-expression)")
    logger.info("✅ Implemented paper hyperparameters (8-head attention, ELU, etc.)")
    logger.info("✅ Trained enhanced GNN model with fair comparison setup")
    
    logger.info("\n📊 Results available in:")
    logger.info("   - data/enhanced/ (enhanced datasets)")
    logger.info("   - models/best_enhanced_model.pt (trained model)")
    logger.info("   - results/ (evaluation results)")
    logger.info("   - wandb/ (training logs)")
    
    logger.info("\n🔬 Next steps:")
    logger.info("   1. Analyze results in wandb dashboard")
    logger.info("   2. Compare with baseline models")
    logger.info("   3. Run ablation studies")
    logger.info("   4. Generate publication-ready figures")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 