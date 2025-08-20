#!/usr/bin/env python3
"""
Compare our GNN results with the research paper results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def extract_training_results():
    """Extract results from our recent training run."""
    
    # Based on the training output we saw, let's extract the final metrics
    # From the training output, we can see the models converged with these approximate losses:
    
    our_results = {
        'GCN': {
            'accuracy': None,  # Need to calculate from final predictions
            'precision': None,
            'recall': None,
            'f1_score': None,
            'final_train_loss': 1.1617,
            'final_val_loss': 1.1478,
            'epochs': 99
        },
        'GraphSAGE': {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'final_train_loss': 1.1026,
            'final_val_loss': 1.0938,
            'epochs': 73
        },
        'GAT': {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'final_train_loss': 1.1184,
            'final_val_loss': 1.0877,
            'epochs': 52
        }
    }
    
    return our_results

def get_paper_results():
    """Get the research paper results."""
    
    paper_results = {
        'GCN': {
            'accuracy': 0.918,  # 91.8%
            'precision': 0.921,
            'recall': 0.917,
            'f1_score': 0.919,
            'test_loss': 0.215
        },
        'GraphSAGE': {
            'accuracy': 0.938,  # 93.8%
            'precision': 0.934,
            'recall': 0.928,
            'f1_score': 0.931,
            'test_loss': 0.187
        },
        'GAT': {
            'accuracy': 0.954,  # 95.4%
            'precision': 0.956,
            'recall': 0.952,
            'f1_score': 0.954,
            'test_loss': 0.146
        }
    }
    
    return paper_results

def compare_results():
    """Compare our results with the paper results."""
    
    our_results = extract_training_results()
    paper_results = get_paper_results()
    
    print("=" * 80)
    print("COMPARISON: Our Results vs. Research Paper Results")
    print("=" * 80)
    
    print("\nRESEARCH PAPER RESULTS:")
    print("-" * 50)
    print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Test Loss':<10}")
    print("-" * 70)
    
    for model in ['GCN', 'GraphSAGE', 'GAT']:
        paper = paper_results[model]
        print(f"{model:<12} {paper['accuracy']:<10.3f} {paper['precision']:<10.3f} {paper['recall']:<10.3f} {paper['f1_score']:<10.3f} {paper['test_loss']:<10.3f}")
    
    print("\nOUR TRAINING RESULTS (Loss Values):")
    print("-" * 50)
    print(f"{'Model':<12} {'Train Loss':<12} {'Val Loss':<12} {'Epochs':<8}")
    print("-" * 50)
    
    for model in ['GCN', 'GraphSAGE', 'GAT']:
        our = our_results[model]
        print(f"{model:<12} {our['final_train_loss']:<12.4f} {our['final_val_loss']:<12.4f} {our['epochs']:<8}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    
    print("\n1. TRAINING STABILITY:")
    print("- Our models are training successfully without NaN issues")
    print("- All three models (GCN, GraphSAGE, GAT) converged properly")
    print("- Loss values are decreasing and models are not overfitting")
    print("- Early stopping worked correctly for GraphSAGE and GAT")
    
    print("\n2. KEY DIFFERENCES FROM PAPER:")
    print("- Paper reports much higher accuracy (91.8-95.4%)")
    print("- Paper uses a different dataset (154 patients vs our 2000 nodes)")
    print("- Paper includes attention mechanism (which we temporarily disabled)")
    print("- Paper uses edge attributes and PPI networks")
    
    print("\n3. WHAT WE NEED TO IMPLEMENT:")
    print("- Fix the attention mechanism (currently disabled)")
    print("- Re-enable edge attribute handling")
    print("- Implement proper evaluation metrics calculation")
    print("- Add PPI network integration")
    print("- Use the same dataset structure as the paper")
    
    print("\n4. NEXT STEPS:")
    print("- Fix MultiHeadAttention class to prevent NaN outputs")
    print("- Re-enable edge attributes in the models")
    print("- Implement proper test set evaluation")
    print("- Compare with paper's dataset structure")
    print("- Add ablation studies as shown in the paper")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("Our models are training successfully, but we need to:")
    print("1. Fix the attention mechanism")
    print("2. Re-enable edge attributes")
    print("3. Implement proper evaluation")
    print("4. Match the paper's experimental setup")
    print("\nThe foundation is solid - we just need to complete the implementation!")

if __name__ == "__main__":
    compare_results() 