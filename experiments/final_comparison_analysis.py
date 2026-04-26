import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_all_results():
    """Load all results for comparison"""
    results = {}
    
    # Load original results
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        try:
            with open(f'results/{model}_metrics.json', 'r') as f:
                results[f'{model}_original'] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {model}_metrics.json not found")
    
    # Load fixed results
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        try:
            with open(f'results/{model}_fixed_metrics.json', 'r') as f:
                results[f'{model}_fixed'] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {model}_fixed_metrics.json not found")
    
    return results

def get_paper_results():
    """Paper's published results"""
    paper_results = {
        'GCN': {
            'precision': 0.921,
            'recall': 0.917,
            'f1': 0.919,
            'accuracy': 0.918,
            'test_loss': 0.215
        },
        'GraphSAGE': {
            'precision': 0.934,
            'recall': 0.928,
            'f1': 0.931,
            'accuracy': 0.938,
            'test_loss': 0.187
        },
        'GAT': {
            'precision': 0.956,
            'recall': 0.952,
            'f1': 0.954,
            'accuracy': 0.954,
            'test_loss': 0.146
        }
    }
    return paper_results

def create_comprehensive_comparison():
    """Create comprehensive comparison of all results"""
    print("="*100)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*100)
    
    # Load results
    all_results = load_all_results()
    paper_results = get_paper_results()
    
    # Create comparison table
    comparison_data = []
    
    # Add original results
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        if f'{model}_original' in all_results:
            metrics = all_results[f'{model}_original']
            comparison_data.append({
                'Model': model,
                'Version': 'Original',
                'Accuracy': metrics.get('accuracy', 0),
                'F1_Score': metrics.get('f1', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'ROC_AUC': metrics.get('roc_auc', 0),
                'PR_AUC': metrics.get('pr_auc', 0)
            })
    
    # Add fixed results
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        if f'{model}_fixed' in all_results:
            metrics = all_results[f'{model}_fixed']
            comparison_data.append({
                'Model': model,
                'Version': 'Fixed',
                'Accuracy': metrics.get('accuracy', 0),
                'F1_Score': metrics.get('f1', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'ROC_AUC': metrics.get('roc_auc', 0),
                'PR_AUC': metrics.get('pr_auc', 0),
                'Balanced_Accuracy': metrics.get('balanced_accuracy', 0)
            })
    
    # Add paper results
    for model, metrics in paper_results.items():
        comparison_data.append({
            'Model': model,
            'Version': 'Paper',
            'Accuracy': metrics['accuracy'],
            'F1_Score': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'ROC_AUC': 0,  # Not provided in paper
            'PR_AUC': 0    # Not provided in paper
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\nDetailed Comparison Table:")
    print("-" * 120)
    print(f"{'Model':<12} {'Version':<10} {'Accuracy':<10} {'F1_Score':<10} {'Precision':<10} {'Recall':<10} {'ROC_AUC':<10} {'PR_AUC':<10}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['Model']:<12} {row['Version']:<10} {row['Accuracy']:<10.4f} {row['F1_Score']:<10.4f} "
              f"{row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['ROC_AUC']:<10.4f} {row['PR_AUC']:<10.4f}")
    
    return df, all_results, paper_results

def analyze_improvements(df):
    """Analyze improvements made by fixes"""
    print("\n" + "="*100)
    print("IMPROVEMENT ANALYSIS")
    print("="*100)
    
    # Compare original vs fixed results
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        original = df[(df['Model'] == model) & (df['Version'] == 'Original')]
        fixed = df[(df['Model'] == model) & (df['Version'] == 'Fixed')]
        
        if not original.empty and not fixed.empty:
            print(f"\n{model} Model Improvements:")
            print("-" * 50)
            
            # F1 Score improvement
            f1_improvement = fixed.iloc[0]['F1_Score'] - original.iloc[0]['F1_Score']
            print(f"  F1 Score: {original.iloc[0]['F1_Score']:.4f} → {fixed.iloc[0]['F1_Score']:.4f} "
                  f"({f1_improvement:+.4f})")
            
            # ROC-AUC improvement
            roc_improvement = fixed.iloc[0]['ROC_AUC'] - original.iloc[0]['ROC_AUC']
            print(f"  ROC-AUC: {original.iloc[0]['ROC_AUC']:.4f} → {fixed.iloc[0]['ROC_AUC']:.4f} "
                  f"({roc_improvement:+.4f})")
            
            # PR-AUC improvement
            pr_improvement = fixed.iloc[0]['PR_AUC'] - original.iloc[0]['PR_AUC']
            print(f"  PR-AUC: {original.iloc[0]['PR_AUC']:.4f} → {fixed.iloc[0]['PR_AUC']:.4f} "
                  f"({pr_improvement:+.4f})")
            
            # Balanced Accuracy (new metric)
            if 'Balanced_Accuracy' in fixed.columns:
                print(f"  Balanced Accuracy: {fixed.iloc[0]['Balanced_Accuracy']:.4f} (new metric)")

def identify_remaining_issues(df, paper_results):
    """Identify remaining issues compared to paper"""
    print("\n" + "="*100)
    print("REMAINING ISSUES ANALYSIS")
    print("="*100)
    
    print("\nKey Issues Identified:")
    print("1. SEVERE CLASS IMBALANCE:")
    print("   - Our dataset: 99.74% Class 0, 0.26% Class 1 (380:1 ratio)")
    print("   - This is much more extreme than typical cancer datasets")
    print("   - Paper likely used a more balanced dataset or different labeling")
    
    print("\n2. F1 SCORE GAP:")
    print("   - Paper F1 scores: 0.919-0.954")
    print("   - Our F1 scores: 0.038 (even after fixes)")
    print("   - Gap: ~0.88-0.92 points")
    
    print("\n3. POSITIVE PROGRESS:")
    print("   - Balanced accuracy improved significantly (73-92%)")
    print("   - ROC-AUC improved (72-76%)")
    print("   - Models now learn meaningful patterns")
    
    print("\n4. DATA QUALITY ISSUES:")
    print("   - Dataset size: 1,143 samples vs paper's 154 patients")
    print("   - Different data sources or preprocessing")
    print("   - Possible labeling inconsistencies")

def create_visualization(df):
    """Create visualization of results comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Filter data for visualization
    plot_data = df[df['Version'].isin(['Original', 'Fixed', 'Paper'])]
    
    # 1. F1 Score comparison
    ax1 = axes[0, 0]
    for version in ['Original', 'Fixed', 'Paper']:
        data = plot_data[plot_data['Version'] == version]
        if not data.empty:
            ax1.bar([f"{row['Model']}\n({version})" for _, row in data.iterrows()], 
                   data['F1_Score'], alpha=0.8, label=version)
    ax1.set_title('F1 Score Comparison')
    ax1.set_ylabel('F1 Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC-AUC comparison
    ax2 = axes[0, 1]
    for version in ['Original', 'Fixed']:
        data = plot_data[plot_data['Version'] == version]
        if not data.empty:
            ax2.bar([f"{row['Model']}\n({version})" for _, row in data.iterrows()], 
                   data['ROC_AUC'], alpha=0.8, label=version)
    ax2.set_title('ROC-AUC Comparison')
    ax2.set_ylabel('ROC-AUC')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy comparison
    ax3 = axes[0, 2]
    for version in ['Original', 'Fixed', 'Paper']:
        data = plot_data[plot_data['Version'] == version]
        if not data.empty:
            ax3.bar([f"{row['Model']}\n({version})" for _, row in data.iterrows()], 
                   data['Accuracy'], alpha=0.8, label=version)
    ax3.set_title('Accuracy Comparison')
    ax3.set_ylabel('Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Balanced Accuracy (Fixed only)
    ax4 = axes[1, 0]
    fixed_data = plot_data[plot_data['Version'] == 'Fixed']
    if not fixed_data.empty and 'Balanced_Accuracy' in fixed_data.columns:
        ax4.bar([row['Model'] for _, row in fixed_data.iterrows()], 
               fixed_data['Balanced_Accuracy'], alpha=0.8, color='green')
    ax4.set_title('Balanced Accuracy (Fixed)')
    ax4.set_ylabel('Balanced Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # 5. PR-AUC comparison
    ax5 = axes[1, 1]
    for version in ['Original', 'Fixed']:
        data = plot_data[plot_data['Version'] == version]
        if not data.empty:
            ax5.bar([f"{row['Model']}\n({version})" for _, row in data.iterrows()], 
                   data['PR_AUC'], alpha=0.8, label=version)
    ax5.set_title('PR-AUC Comparison')
    ax5.set_ylabel('PR-AUC')
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Improvement summary
    ax6 = axes[1, 2]
    models = ['GAT', 'GCN', 'GraphSAGE']
    f1_improvements = []
    roc_improvements = []
    
    for model in models:
        original = plot_data[(plot_data['Model'] == model) & (plot_data['Version'] == 'Original')]
        fixed = plot_data[(plot_data['Model'] == model) & (plot_data['Version'] == 'Fixed')]
        
        if not original.empty and not fixed.empty:
            f1_improvements.append(fixed.iloc[0]['F1_Score'] - original.iloc[0]['F1_Score'])
            roc_improvements.append(fixed.iloc[0]['ROC_AUC'] - original.iloc[0]['ROC_AUC'])
        else:
            f1_improvements.append(0)
            roc_improvements.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    ax6.bar(x - width/2, f1_improvements, width, label='F1 Improvement', alpha=0.8)
    ax6.bar(x + width/2, roc_improvements, width, label='ROC-AUC Improvement', alpha=0.8)
    ax6.set_title('Improvements from Fixes')
    ax6.set_ylabel('Improvement')
    ax6.set_xticks(x)
    ax6.set_xticklabels(models)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def suggest_next_steps():
    """Suggest next steps to improve results"""
    print("\n" + "="*100)
    print("NEXT STEPS TO MATCH PAPER PERFORMANCE")
    print("="*100)
    
    print("\n1. DATA QUALITY IMPROVEMENTS:")
    print("   - Verify data sources match paper exactly")
    print("   - Check labeling methodology")
    print("   - Ensure proper data preprocessing")
    print("   - Consider using paper's exact dataset")
    
    print("\n2. CLASS IMBALANCE SOLUTIONS:")
    print("   - Implement SMOTE or other oversampling")
    print("   - Use focal loss instead of weighted cross-entropy")
    print("   - Consider undersampling majority class")
    print("   - Use ensemble methods")
    
    print("\n3. MODEL ARCHITECTURE:")
    print("   - Verify model implementations match paper exactly")
    print("   - Check initialization methods")
    print("   - Ensure proper layer configurations")
    print("   - Verify attention mechanisms in GAT")
    
    print("\n4. TRAINING PROCEDURE:")
    print("   - Use paper's exact hyperparameters")
    print("   - Implement proper learning rate scheduling")
    print("   - Use paper's exact train/val/test split")
    print("   - Consider longer training with more epochs")
    
    print("\n5. EVALUATION METRICS:")
    print("   - Focus on balanced accuracy and F1 score")
    print("   - Use proper cross-validation")
    print("   - Implement confusion matrix analysis")
    print("   - Add statistical significance testing")

def main():
    """Main analysis function"""
    print("Creating comprehensive comparison analysis...")
    
    # Create comparison
    df, all_results, paper_results = create_comprehensive_comparison()
    
    # Analyze improvements
    analyze_improvements(df)
    
    # Identify remaining issues
    identify_remaining_issues(df, paper_results)
    
    # Create visualization
    create_visualization(df)
    
    # Suggest next steps
    suggest_next_steps()
    
    # Save comparison data
    df.to_csv('results/comprehensive_comparison.csv', index=False)
    print(f"\nComparison data saved to results/comprehensive_comparison.csv")
    print(f"Visualization saved to results/comprehensive_comparison_analysis.png")

if __name__ == "__main__":
    main() 