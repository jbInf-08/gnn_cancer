import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load TCGA data
# Replace these with your actual TCGA data files
pre_data = pd.read_csv('data/raw/tcga/pre_processing_data.csv')['column_name'].values
post_data = pd.read_csv('data/raw/tcga/post_processing_data.csv')['column_name'].values

# Calculate statistical measures
pre_mean = np.mean(pre_data)
pre_median = np.median(pre_data)
pre_std = np.std(pre_data)

post_mean = np.mean(post_data)
post_median = np.median(post_data)
post_std = np.std(post_data)

# Create figure for pre-processing
plt.figure(figsize=(10, 6))
plt.hist(pre_data, bins=30, alpha=0.7, color='blue', label='Pre-processing')
plt.axvline(pre_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {pre_mean:.2f}')
plt.axvline(pre_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {pre_median:.2f}')
plt.title('Pre-processing Data Distribution (TCGA)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('pre_processing.png')
plt.close()

# Create figure for post-processing
plt.figure(figsize=(10, 6))
plt.hist(post_data, bins=30, alpha=0.7, color='green', label='Post-processing')
plt.axvline(post_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {post_mean:.2f}')
plt.axvline(post_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {post_median:.2f}')
plt.title('Post-processing Data Distribution (TCGA)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('post_processing.png')
plt.close()

# Create figure comparing pre and post processing
plt.figure(figsize=(10, 6))
plt.hist(pre_data, bins=30, alpha=0.5, color='blue', label='Pre-processing')
plt.hist(post_data, bins=30, alpha=0.5, color='green', label='Post-processing')
plt.axvline(pre_mean, color='red', linestyle='dashed', linewidth=1, label=f'Pre Mean: {pre_mean:.2f}')
plt.axvline(post_mean, color='orange', linestyle='dashed', linewidth=1, label=f'Post Mean: {post_mean:.2f}')
plt.title('Comparison of Pre and Post Processing (TCGA)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('pre_post_comparison.png')
plt.close() 