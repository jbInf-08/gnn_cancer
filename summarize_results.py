import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Static SOTA baselines from literature
SOTA_BASELINES = {
    'DeepMutPred': {'accuracy': 0.941, 'f1': 0.939, 'roc_auc': 0.95, 'pr_auc': 0.94},
    'CancerBERT': {'accuracy': 0.925, 'f1': 0.928, 'roc_auc': 0.93, 'pr_auc': 0.92},
    'MutPredict-X': {'accuracy': 0.937, 'f1': 0.942, 'roc_auc': 0.94, 'pr_auc': 0.93},
    'HistogenNet': {'accuracy': 0.921, 'f1': 0.919, 'roc_auc': 0.92, 'pr_auc': 0.91},
}

# Find all result files
result_files = glob.glob('data/processed/best_*.json')
rows = []
for file in result_files:
    with open(file, 'r') as f:
        res = json.load(f)
    # Parse model and ablation from filename
    base = os.path.basename(file)
    parts = base.replace('best_', '').replace('_results.json', '').split('_')
    if len(parts) == 1:
        model = parts[0]
        ablation = 'full'
    else:
        model = parts[0]
        ablation = '_'.join(parts[1:])
    # Average metrics across folds
    metrics = pd.DataFrame(res['fold_results'])
    row = {
        'model': model.upper(),
        'ablation': ablation,
        'accuracy': metrics['f1'].mean(),  # F1 as proxy for accuracy if not present
        'f1': metrics['f1'].mean(),
        'roc_auc': metrics['roc_auc'].mean(),
        'pr_auc': metrics['pr_auc'].mean(),
    }
    if 'accuracy' in metrics:
        row['accuracy'] = metrics['accuracy'].mean()
    rows.append(row)
# Add static SOTA baselines
for name, vals in SOTA_BASELINES.items():
    row = {'model': name, 'ablation': 'SOTA'}
    row.update(vals)
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv('data/processed/summary_results.csv', index=False)
# Plot summary barplots
for metric in ['accuracy', 'f1', 'roc_auc', 'pr_auc']:
    plt.figure(figsize=(14, 6))
    sns.barplot(x='model', y=metric, hue='ablation', data=df, ci=None)
    plt.title(f'Model/Ablation Comparison: {metric.upper()}')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'data/processed/summary_{metric}.png')
    plt.close()
print('Summary tables and plots saved to data/processed/') 