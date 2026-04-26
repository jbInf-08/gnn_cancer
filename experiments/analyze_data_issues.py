import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from collections import Counter
import json

def analyze_class_distribution():
    """Analyze the class distribution in our dataset"""
    print("="*80)
    print("ANALYZING CLASS DISTRIBUTION")
    print("="*80)
    
    # Load enhanced data
    enhanced_dir = Path("data/enhanced")
    
    try:
        # Load labels
        with open(enhanced_dir / "labels.pkl", 'rb') as f:
            labels = pickle.load(f)
        
        # Convert to list and analyze
        label_list = list(labels.values())
        class_counts = Counter(label_list)
        
        print(f"\nTotal samples: {len(label_list)}")
        print(f"Class distribution: {dict(class_counts)}")
        
        # Calculate percentages
        total = len(label_list)
        for class_label, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")
        
        # Check for severe imbalance
        min_class_count = min(class_counts.values())
        max_class_count = max(class_counts.values())
        imbalance_ratio = max_class_count / min_class_count
        
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            print("   This explains the poor F1 scores and identical model performance.")
        elif imbalance_ratio > 5:
            print("⚠️  MODERATE CLASS IMBALANCE DETECTED!")
        else:
            print("✅ Class distribution looks balanced.")
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # Class distribution pie chart
        plt.subplot(1, 2, 1)
        plt.pie(class_counts.values(), labels=[f'Class {k}' for k in class_counts.keys()], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution')
        
        # Class distribution bar chart
        plt.subplot(1, 2, 2)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
        plt.title('Class Counts')
        plt.xlabel('Class')
        plt.ylabel('Count')
        for i, count in enumerate(counts):
            plt.text(classes[i], count + max(counts)*0.01, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig('results/class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_counts, imbalance_ratio
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        return None, None

def analyze_data_quality():
    """Analyze data quality and preprocessing"""
    print("\n" + "="*80)
    print("ANALYZING DATA QUALITY")
    print("="*80)
    
    enhanced_dir = Path("data/enhanced")
    
    try:
        # Load graph
        with open(enhanced_dir / "comprehensive_graph.pkl", 'rb') as f:
            graph = pickle.load(f)
        
        # Load node features
        with open(enhanced_dir / "node_features.pkl", 'rb') as f:
            node_features = pickle.load(f)
        
        # Load labels
        with open(enhanced_dir / "labels.pkl", 'rb') as f:
            labels = pickle.load(f)
        
        print(f"\nGraph Analysis:")
        print(f"  - Number of nodes: {len(graph.nodes())}")
        print(f"  - Number of edges: {len(graph.edges())}")
        print(f"  - Graph density: {len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1) / 2):.6f}")
        
        print(f"\nFeature Analysis:")
        print(f"  - Number of nodes with features: {len(node_features)}")
        print(f"  - Number of nodes with labels: {len(labels)}")
        
        # Check for missing data
        nodes_with_features = set(node_features.keys())
        nodes_with_labels = set(labels.keys())
        nodes_in_graph = set(graph.nodes())
        
        print(f"\nData Completeness:")
        print(f"  - Nodes in graph: {len(nodes_in_graph)}")
        print(f"  - Nodes with features: {len(nodes_with_features)}")
        print(f"  - Nodes with labels: {len(nodes_with_labels)}")
        
        # Check overlap
        complete_nodes = nodes_in_graph & nodes_with_features & nodes_with_labels
        print(f"  - Complete nodes (all data): {len(complete_nodes)}")
        
        if len(complete_nodes) < len(nodes_in_graph):
            print("⚠️  INCOMPLETE DATA DETECTED!")
            print("   Some nodes are missing features or labels.")
        
        # Analyze feature statistics
        if node_features:
            sample_features = list(node_features.values())[0]
            print(f"\nFeature Statistics:")
            print(f"  - Number of features per node: {len(sample_features)}")
            print(f"  - Feature names: {list(sample_features.keys())}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        return False

def check_paper_methodology():
    """Check if our implementation matches paper methodology"""
    print("\n" + "="*80)
    print("CHECKING PAPER METHODOLOGY COMPLIANCE")
    print("="*80)
    
    print("\nPaper Methodology Requirements:")
    print("1. Dataset: 154 patients from CPTAC and GDC")
    print("2. Graph Construction:")
    print("   - Nodes: Individual genes")
    print("   - Edges: PPI (STRING > 0.7), Pathway co-occurrence, Co-expression")
    print("   - Graph size: ~2,000 nodes, ~18,000 edges")
    
    print("\n3. Model Architectures:")
    print("   - GCN: 3 layers, 64 hidden units, ReLU, dropout 0.5")
    print("   - GAT: 3 layers, 8 attention heads, ELU, dropout 0.5")
    print("   - GraphSAGE: 3 layers, mean aggregation, ReLU, dropout 0.5")
    
    print("\n4. Training:")
    print("   - Optimizer: Adam (lr=0.001, weight_decay=5e-4)")
    print("   - Loss: Binary Cross-Entropy")
    print("   - Split: 70/15/15 train/val/test")
    print("   - Early stopping: patience=10")
    
    print("\n5. Evaluation:")
    print("   - Metrics: Accuracy, Precision, Recall, F1")
    print("   - Cross-validation: 5-fold")
    
    # Check our implementation
    print("\nOur Implementation Check:")
    
    # Load our training script to check parameters
    try:
        with open('enhanced_training.py', 'r') as f:
            training_code = f.read()
        
        checks = {
            "Adam optimizer": "torch.optim.Adam" in training_code,
            "Learning rate 0.001": "lr=0.001" in training_code or "learning_rate=0.001" in training_code,
            "Weight decay 5e-4": "weight_decay=5e-4" in training_code,
            "CrossEntropyLoss": "CrossEntropyLoss" in training_code,
            "Early stopping": "patience" in training_code,
            "5-fold validation": "n_splits=5" in training_code or "StratifiedKFold" in training_code
        }
        
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check}")
        
    except FileNotFoundError:
        print("   ❌ Could not check training script")

def suggest_fixes():
    """Suggest specific fixes for the identified issues"""
    print("\n" + "="*80)
    print("SUGGESTED FIXES")
    print("="*80)
    
    print("\n1. FIX CLASS IMBALANCE:")
    print("   - Use class weights in loss function")
    print("   - Implement SMOTE or other oversampling techniques")
    print("   - Use balanced accuracy instead of regular accuracy")
    print("   - Consider focal loss for imbalanced datasets")
    
    print("\n2. IMPROVE DATA PREPROCESSING:")
    print("   - Verify graph construction matches paper exactly")
    print("   - Check feature engineering process")
    print("   - Ensure proper normalization")
    print("   - Validate data quality and completeness")
    
    print("\n3. ENHANCE MODEL TRAINING:")
    print("   - Implement proper cross-validation")
    print("   - Add class weights to loss function")
    print("   - Use balanced sampling in data loaders")
    print("   - Implement proper early stopping")
    
    print("\n4. IMPROVE EVALUATION:")
    print("   - Use balanced accuracy metric")
    print("   - Implement proper confusion matrix analysis")
    print("   - Add ROC-AUC and PR-AUC analysis")
    print("   - Use stratified sampling for splits")
    
    print("\n5. VERIFY IMPLEMENTATION:")
    print("   - Check model architectures match paper exactly")
    print("   - Verify hyperparameter settings")
    print("   - Ensure proper initialization")
    print("   - Validate training procedure")

def main():
    """Main analysis function"""
    print("Analyzing data issues and class distribution...")
    
    # Analyze class distribution
    class_counts, imbalance_ratio = analyze_class_distribution()
    
    # Analyze data quality
    data_quality_ok = analyze_data_quality()
    
    # Check paper methodology
    check_paper_methodology()
    
    # Suggest fixes
    suggest_fixes()
    
    # Save analysis results
    analysis_results = {
        "class_distribution": dict(class_counts) if class_counts else None,
        "imbalance_ratio": imbalance_ratio,
        "data_quality_ok": data_quality_ok,
        "issues_found": {
            "class_imbalance": imbalance_ratio > 10 if imbalance_ratio else True,
            "incomplete_data": not data_quality_ok,
            "methodology_mismatch": True  # Need to verify
        }
    }
    
    with open('results/data_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis results saved to results/data_analysis_results.json")
    print(f"Class distribution plot saved to results/class_distribution_analysis.png")

if __name__ == "__main__":
    main() 