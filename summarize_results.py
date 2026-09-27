import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# External baselines for comparison.
#
# Every entry MUST be traceable: a real published result on THIS task and
# THIS data, with a citation, or a baseline reproduced in this repository.
# Do not add numbers that cannot be sourced. Entries without a 'citation'
# key are rejected below rather than silently reported.
#
# Shape:
#   'Name': {'accuracy': ..., 'f1': ..., 'roc_auc': ..., 'pr_auc': ...,
#            'citation': 'Author et al. (Year), Journal, doi:...'}
EXTERNAL_BASELINES = {}

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
        'accuracy': float('nan'),  # filled below only if the results file records accuracy
        'f1': metrics['f1'].mean(),
        'roc_auc': metrics['roc_auc'].mean(),
        'pr_auc': metrics['pr_auc'].mean(),
    }
    if 'accuracy' in metrics:
        row['accuracy'] = metrics['accuracy'].mean()
    rows.append(row)
# Add external baselines — only entries that carry a citation
for name, vals in EXTERNAL_BASELINES.items():
    if not vals.get('citation'):
        raise ValueError(f"External baseline {name!r} has no citation; refusing to report it")
    row = {'model': name, 'ablation': 'SOTA'}
    row.update({k: v for k, v in vals.items() if k != 'citation'})
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv('data/processed/summary_results.csv', index=False)
# Plot summary barplots
for metric in ['accuracy', 'f1', 'roc_auc', 'pr_auc']:
    if df[metric].isna().all():
        print(f'No recorded {metric} values; skipping its plot.')
        continue
    plt.figure(figsize=(14, 6))
    sns.barplot(x='model', y=metric, hue='ablation', data=df, ci=None)
    plt.title(f'Model/Ablation Comparison: {metric.upper()}')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'data/processed/summary_{metric}.png')
    plt.close()
print('Summary tables and plots saved to data/processed/') 