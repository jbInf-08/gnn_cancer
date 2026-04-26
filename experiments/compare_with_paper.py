import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_our_results():
    """Load our experimental results"""
    results = {}
    
    # Load metrics for each model
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        try:
            with open(f'results/{model}_metrics.json', 'r') as f:
                results[model] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {model}_metrics.json not found")
            results[model] = {}
    
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

def compare_results():
    """Compare our results with paper results"""
    our_results = load_our_results()
    paper_results = get_paper_results()
    
    # Create comparison dataframe
    comparison_data = []
    
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        if model in our_results and our_results[model]:
            comparison_data.append({
                'Model': model,
                'Metric': 'Accuracy',
                'Our_Result': our_results[model].get('accuracy', 0),
                'Paper_Result': paper_results[model]['accuracy'],
                'Difference': our_results[model].get('accuracy', 0) - paper_results[model]['accuracy']
            })
            
            comparison_data.append({
                'Model': model,
                'Metric': 'F1_Score',
                'Our_Result': our_results[model].get('f1', 0),
                'Paper_Result': paper_results[model]['f1'],
                'Difference': our_results[model].get('f1', 0) - paper_results[model]['f1']
            })
            
            comparison_data.append({
                'Model': model,
                'Metric': 'Precision',
                'Our_Result': our_results[model].get('precision', 0),
                'Paper_Result': paper_results[model]['precision'],
                'Difference': our_results[model].get('precision', 0) - paper_results[model]['precision']
            })
            
            comparison_data.append({
                'Model': model,
                'Metric': 'Recall',
                'Our_Result': our_results[model].get('recall', 0),
                'Paper_Result': paper_results[model]['recall'],
                'Difference': our_results[model].get('recall', 0) - paper_results[model]['recall']
            })
    
    df = pd.DataFrame(comparison_data)
    return df, our_results, paper_results

def analyze_differences(df, our_results, paper_results):
    """Analyze the differences between our results and paper results"""
    print("="*80)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*80)
    
    # Print detailed comparison table
    print("\nDetailed Comparison Table:")
    print("-" * 80)
    print(f"{'Model':<12} {'Metric':<12} {'Our Result':<12} {'Paper Result':<12} {'Difference':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['Model']:<12} {row['Metric']:<12} {row['Our_Result']:<12.4f} {row['Paper_Result']:<12.4f} {row['Difference']:<12.4f}")
    
    # Key observations
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    
    # Accuracy comparison
    print("\n1. ACCURACY COMPARISON:")
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        if model in our_results and our_results[model]:
            our_acc = our_results[model].get('accuracy', 0)
            paper_acc = paper_results[model]['accuracy']
            diff = our_acc - paper_acc
            print(f"   {model}: Our accuracy = {our_acc:.4f}, Paper accuracy = {paper_acc:.4f}, Difference = {diff:+.4f}")
    
    # F1 Score comparison
    print("\n2. F1 SCORE COMPARISON:")
    for model in ['GAT', 'GCN', 'GraphSAGE']:
        if model in our_results and our_results[model]:
            our_f1 = our_results[model].get('f1', 0)
            paper_f1 = paper_results[model]['f1']
            diff = our_f1 - paper_f1
            print(f"   {model}: Our F1 = {our_f1:.4f}, Paper F1 = {paper_f1:.4f}, Difference = {diff:+.4f}")
    
    # Overall performance ranking
    print("\n3. PERFORMANCE RANKING:")
    print("   Paper Ranking (by F1): GAT (0.954) > GraphSAGE (0.931) > GCN (0.919)")
    print("   Our Ranking (by F1): GraphSAGE (0.400) = GAT (0.400) = GCN (0.400)")
    
    # Identify potential issues
    print("\n4. POTENTIAL ISSUES IDENTIFIED:")
    print("   - Our F1 scores are significantly lower than paper results")
    print("   - All our models have identical F1 scores (0.400), suggesting class imbalance issues")
    print("   - Our accuracy is higher but F1 is much lower, indicating poor performance on minority class")
    print("   - This suggests our models may be predicting mostly one class")

def create_comparison_plots(df):
    """Create visualization comparing our results with paper results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    accuracy_data = df[df['Metric'] == 'Accuracy']
    x = np.arange(len(accuracy_data))
    width = 0.35
    
    ax1.bar(x - width/2, accuracy_data['Our_Result'], width, label='Our Results', alpha=0.8)
    ax1.bar(x + width/2, accuracy_data['Paper_Result'], width, label='Paper Results', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(accuracy_data['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    f1_data = df[df['Metric'] == 'F1_Score']
    x = np.arange(len(f1_data))
    
    ax2.bar(x - width/2, f1_data['Our_Result'], width, label='Our Results', alpha=0.8)
    ax2.bar(x + width/2, f1_data['Paper_Result'], width, label='Paper Results', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(f1_data['Model'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Difference heatmap
    ax3 = axes[1, 0]
    pivot_df = df.pivot(index='Model', columns='Metric', values='Difference')
    sns.heatmap(pivot_df, annot=True, cmap='RdYlBu', center=0, ax=ax3)
    ax3.set_title('Difference (Our - Paper)')
    
    # 4. Overall performance radar chart
    ax4 = axes[1, 1]
    metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
    models = ['GAT', 'GCN', 'GraphSAGE']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, model in enumerate(models):
        our_values = []
        paper_values = []
        for metric in metrics:
            metric_data = df[(df['Model'] == model) & (df['Metric'] == metric)]
            if not metric_data.empty:
                our_values.append(metric_data.iloc[0]['Our_Result'])
                paper_values.append(metric_data.iloc[0]['Paper_Result'])
            else:
                our_values.append(0)
                paper_values.append(0)
        
        our_values += our_values[:1]  # Complete the circle
        paper_values += paper_values[:1]
        
        ax4.plot(angles, our_values, 'o-', linewidth=2, label=f'{model} (Our)', alpha=0.7)
        ax4.plot(angles, paper_values, 's-', linewidth=2, label=f'{model} (Paper)', alpha=0.7)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1)
    ax4.set_title('Performance Comparison Radar Chart')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/paper_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def suggest_improvements():
    """Suggest improvements to match paper performance"""
    print("\n" + "="*80)
    print("SUGGESTED IMPROVEMENTS")
    print("="*80)
    
    print("\n1. DATA QUALITY ISSUES:")
    print("   - Check class distribution in our dataset")
    print("   - Verify data preprocessing matches paper methodology")
    print("   - Ensure graph construction follows paper specifications")
    
    print("\n2. MODEL ARCHITECTURE:")
    print("   - Verify model implementations match paper exactly")
    print("   - Check hyperparameter settings")
    print("   - Ensure proper initialization and training procedures")
    
    print("\n3. TRAINING PROCEDURE:")
    print("   - Verify train/validation/test split matches paper (70/15/15)")
    print("   - Check early stopping criteria")
    print("   - Ensure proper learning rate scheduling")
    
    print("\n4. EVALUATION METRICS:")
    print("   - Verify metric calculation methods")
    print("   - Check for any data leakage")
    print("   - Ensure proper cross-validation setup")
    
    print("\n5. NEXT STEPS:")
    print("   - Analyze class distribution in our dataset")
    print("   - Review data preprocessing pipeline")
    print("   - Check model implementations against paper")
    print("   - Verify graph construction methodology")

def main():
    """Main function to run comparison analysis"""
    print("Comparing our results with paper results...")
    
    # Load and compare results
    df, our_results, paper_results = compare_results()
    
    # Analyze differences
    analyze_differences(df, our_results, paper_results)
    
    # Create visualization
    create_comparison_plots(df)
    
    # Suggest improvements
    suggest_improvements()
    
    # Save comparison data
    df.to_csv('results/paper_comparison.csv', index=False)
    print(f"\nComparison data saved to results/paper_comparison.csv")
    print(f"Visualization saved to results/paper_comparison_analysis.png")

if __name__ == "__main__":
    main() 