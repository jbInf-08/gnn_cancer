import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BalancedDatasetCreator:
    """Create balanced datasets using various techniques"""
    
    def __init__(self, data_dir="data/enhanced"):
        self.data_dir = Path(data_dir)
        self.original_data = None
        self.balanced_datasets = {}
        
    def load_original_data(self):
        """Load original imbalanced data"""
        print("Loading original data...")
        
        # Load graph
        with open(self.data_dir / "comprehensive_graph.pkl", 'rb') as f:
            graph = pickle.load(f)
        
        # Load node features
        with open(self.data_dir / "node_features.pkl", 'rb') as f:
            node_features = pickle.load(f)
        
        # Load labels
        with open(self.data_dir / "labels.pkl", 'rb') as f:
            labels = pickle.load(f)
        
        # Convert to numpy arrays
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create feature matrix
        feature_matrix = []
        label_vector = []
        
        for node in nodes:
            features = node_features[node]
            feature_vector = [
                features['mutation_count'],
                features['expression_mean'],
                features['cnv_mean'],
                features['degree_ppi'],
                features['degree_pathway'],
                features['degree_coexpression']
            ]
            feature_matrix.append(feature_vector)
            label_vector.append(labels[node])
        
        X = np.array(feature_matrix)
        y = np.array(label_vector)
        
        # Store original data
        self.original_data = {
            'X': X,
            'y': y,
            'nodes': nodes,
            'node_to_idx': node_to_idx,
            'graph': graph
        }
        
        # Analyze class distribution
        class_counts = Counter(y)
        print(f"Original class distribution: {dict(class_counts)}")
        print(f"Imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}")
        
        return self.original_data
    
    def create_manual_oversampling_dataset(self, target_ratio=1.0):
        """Create balanced dataset using manual oversampling for extreme imbalance"""
        print(f"\nCreating manual oversampling dataset (target_ratio={target_ratio})...")
        
        X = self.original_data['X']
        y = self.original_data['y']
        
        # Get class indices
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        
        # Calculate target number of samples for each class
        n_class_0 = len(class_0_indices)
        n_class_1 = len(class_1_indices)
        
        if target_ratio == 1.0:
            # Perfect balance
            target_samples = max(n_class_0, n_class_1)
        else:
            # Custom ratio
            target_samples = int(n_class_0 * target_ratio)
        
        # Oversample minority class (class 1)
        if n_class_1 < target_samples:
            # Repeat minority class samples
            repeat_times = target_samples // n_class_1
            remainder = target_samples % n_class_1
            
            oversampled_indices = np.tile(class_1_indices, repeat_times)
            if remainder > 0:
                oversampled_indices = np.concatenate([oversampled_indices, class_1_indices[:remainder]])
            
            # Combine with majority class
            balanced_indices = np.concatenate([class_0_indices, oversampled_indices])
        else:
            # Undersample majority class
            balanced_indices = np.concatenate([class_0_indices[:target_samples], class_1_indices])
        
        # Shuffle the indices
        np.random.shuffle(balanced_indices)
        
        # Create balanced dataset
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        # Analyze new distribution
        class_counts = Counter(y_balanced)
        print(f"Manual oversampling class distribution: {dict(class_counts)}")
        print(f"New imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}")
        
        self.balanced_datasets['manual_oversampling'] = {
            'X': X_balanced,
            'y': y_balanced,
            'method': 'Manual Oversampling',
            'original_indices': balanced_indices,
            'synthetic_indices': None
        }
        
        return self.balanced_datasets['manual_oversampling']
    
    def create_smote_balanced_dataset(self, k_neighbors=1):
        """Create balanced dataset using SMOTE with adjusted parameters"""
        print(f"\nCreating SMOTE balanced dataset (k_neighbors={k_neighbors})...")
        
        X = self.original_data['X']
        y = self.original_data['y']
        
        # Check if SMOTE can be applied
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        if min_class_count <= k_neighbors:
            print(f"Warning: Minority class has only {min_class_count} samples, using manual oversampling instead")
            return self.create_manual_oversampling_dataset()
        
        try:
            # Apply SMOTE with adjusted parameters
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Analyze new distribution
            class_counts = Counter(y_balanced)
            print(f"SMOTE balanced class distribution: {dict(class_counts)}")
            print(f"New imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}")
            
            self.balanced_datasets['smote'] = {
                'X': X_balanced,
                'y': y_balanced,
                'method': 'SMOTE',
                'original_indices': np.arange(len(X)),
                'synthetic_indices': np.arange(len(X), len(X_balanced))
            }
            
            return self.balanced_datasets['smote']
            
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Falling back to manual oversampling")
            return self.create_manual_oversampling_dataset()
    
    def create_adasyn_balanced_dataset(self, k_neighbors=1):
        """Create balanced dataset using ADASYN with adjusted parameters"""
        print(f"\nCreating ADASYN balanced dataset (k_neighbors={k_neighbors})...")
        
        X = self.original_data['X']
        y = self.original_data['y']
        
        # Check if ADASYN can be applied
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        if min_class_count <= k_neighbors:
            print(f"Warning: Minority class has only {min_class_count} samples, using manual oversampling instead")
            return self.create_manual_oversampling_dataset()
        
        try:
            # Apply ADASYN with adjusted parameters
            adasyn = ADASYN(k_neighbors=k_neighbors, random_state=42)
            X_balanced, y_balanced = adasyn.fit_resample(X, y)
            
            # Analyze new distribution
            class_counts = Counter(y_balanced)
            print(f"ADASYN balanced class distribution: {dict(class_counts)}")
            print(f"New imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}")
            
            self.balanced_datasets['adasyn'] = {
                'X': X_balanced,
                'y': y_balanced,
                'method': 'ADASYN',
                'original_indices': np.arange(len(X)),
                'synthetic_indices': np.arange(len(X), len(X_balanced))
            }
            
            return self.balanced_datasets['adasyn']
            
        except Exception as e:
            print(f"ADASYN failed: {e}")
            print("Falling back to manual oversampling")
            return self.create_manual_oversampling_dataset()
    
    def create_undersampled_dataset(self, sampling_strategy='auto'):
        """Create balanced dataset using undersampling"""
        print(f"\nCreating undersampled dataset (sampling_strategy={sampling_strategy})...")
        
        X = self.original_data['X']
        y = self.original_data['y']
        
        # Apply undersampling
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        
        # Analyze new distribution
        class_counts = Counter(y_balanced)
        print(f"Undersampled class distribution: {dict(class_counts)}")
        print(f"New imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}")
        
        self.balanced_datasets['undersampled'] = {
            'X': X_balanced,
            'y': y_balanced,
            'method': 'Undersampling',
            'original_indices': undersampler.sample_indices_,
            'synthetic_indices': None
        }
        
        return self.balanced_datasets['undersampled']
    
    def create_hybrid_balanced_dataset(self):
        """Create balanced dataset using hybrid approach (SMOTE + ENN)"""
        print("\nCreating hybrid balanced dataset (SMOTE + ENN)...")
        
        X = self.original_data['X']
        y = self.original_data['y']
        
        # Check if hybrid approach can be applied
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        if min_class_count <= 1:
            print(f"Warning: Minority class has only {min_class_count} samples, using manual oversampling instead")
            return self.create_manual_oversampling_dataset()
        
        try:
            # Apply SMOTE + ENN
            smote_enn = SMOTEENN(random_state=42)
            X_balanced, y_balanced = smote_enn.fit_resample(X, y)
            
            # Analyze new distribution
            class_counts = Counter(y_balanced)
            print(f"Hybrid balanced class distribution: {dict(class_counts)}")
            print(f"New imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}")
            
            self.balanced_datasets['hybrid'] = {
                'X': X_balanced,
                'y': y_balanced,
                'method': 'SMOTE + ENN',
                'original_indices': np.arange(len(X)),
                'synthetic_indices': np.arange(len(X), len(X_balanced))
            }
            
            return self.balanced_datasets['hybrid']
            
        except Exception as e:
            print(f"Hybrid approach failed: {e}")
            print("Falling back to manual oversampling")
            return self.create_manual_oversampling_dataset()
    
    def create_weighted_dataset(self):
        """Create dataset with computed class weights"""
        print("\nComputing class weights for weighted loss...")
        
        X = self.original_data['X']
        y = self.original_data['y']
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        # Convert to tensor
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        print(f"Computed class weights: {class_weights}")
        
        self.balanced_datasets['weighted'] = {
            'X': X,
            'y': y,
            'method': 'Class Weights',
            'class_weights': class_weights_tensor,
            'original_indices': np.arange(len(X)),
            'synthetic_indices': None
        }
        
        return self.balanced_datasets['weighted']
    
    def create_moderate_balance_dataset(self, target_ratio=0.1):
        """Create moderately balanced dataset (10:1 ratio instead of 380:1)"""
        print(f"\nCreating moderately balanced dataset (target_ratio={target_ratio})...")
        
        X = self.original_data['X']
        y = self.original_data['y']
        
        # Get class indices
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        
        # Calculate target number of samples for minority class
        target_minority = int(len(class_0_indices) * target_ratio)
        
        # Oversample minority class to reach target ratio
        if len(class_1_indices) < target_minority:
            repeat_times = target_minority // len(class_1_indices)
            remainder = target_minority % len(class_1_indices)
            
            oversampled_indices = np.tile(class_1_indices, repeat_times)
            if remainder > 0:
                oversampled_indices = np.concatenate([oversampled_indices, class_1_indices[:remainder]])
        else:
            oversampled_indices = class_1_indices[:target_minority]
        
        # Combine with majority class
        balanced_indices = np.concatenate([class_0_indices, oversampled_indices])
        
        # Shuffle the indices
        np.random.shuffle(balanced_indices)
        
        # Create balanced dataset
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        # Analyze new distribution
        class_counts = Counter(y_balanced)
        print(f"Moderate balance class distribution: {dict(class_counts)}")
        print(f"New imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}")
        
        self.balanced_datasets['moderate_balance'] = {
            'X': X_balanced,
            'y': y_balanced,
            'method': 'Moderate Balance (10:1)',
            'original_indices': balanced_indices,
            'synthetic_indices': None
        }
        
        return self.balanced_datasets['moderate_balance']
    
    def visualize_balance_comparison(self):
        """Visualize the balance comparison across all methods"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original data
        original_counts = Counter(self.original_data['y'])
        axes[0, 0].pie(original_counts.values(), labels=[f'Class {k}' for k in original_counts.keys()], 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Original (Imbalanced)')
        
        # Manual oversampling
        if 'manual_oversampling' in self.balanced_datasets:
            manual_counts = Counter(self.balanced_datasets['manual_oversampling']['y'])
            axes[0, 1].pie(manual_counts.values(), labels=[f'Class {k}' for k in manual_counts.keys()], 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Manual Oversampling')
        
        # Moderate balance
        if 'moderate_balance' in self.balanced_datasets:
            moderate_counts = Counter(self.balanced_datasets['moderate_balance']['y'])
            axes[0, 2].pie(moderate_counts.values(), labels=[f'Class {k}' for k in moderate_counts.keys()], 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('Moderate Balance (10:1)')
        
        # Undersampled
        if 'undersampled' in self.balanced_datasets:
            under_counts = Counter(self.balanced_datasets['undersampled']['y'])
            axes[1, 0].pie(under_counts.values(), labels=[f'Class {k}' for k in under_counts.keys()], 
                          autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Undersampled')
        
        # SMOTE (if available)
        if 'smote' in self.balanced_datasets:
            smote_counts = Counter(self.balanced_datasets['smote']['y'])
            axes[1, 1].pie(smote_counts.values(), labels=[f'Class {k}' for k in smote_counts.keys()], 
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('SMOTE')
        
        # Bar chart comparison
        methods = ['Original']
        ratios = [max(original_counts.values()) / min(original_counts.values())]
        
        for method, data in self.balanced_datasets.items():
            if method != 'weighted':
                counts = Counter(data['y'])
                ratio = max(counts.values()) / min(counts.values())
                methods.append(data['method'])
                ratios.append(ratio)
        
        axes[1, 2].bar(methods, ratios, color=['red'] + ['green'] * (len(methods) - 1))
        axes[1, 2].set_title('Imbalance Ratio Comparison')
        axes[1, 2].set_ylabel('Imbalance Ratio')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/balance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_balanced_datasets(self):
        """Save all balanced datasets"""
        print("\nSaving balanced datasets...")
        
        for method, data in self.balanced_datasets.items():
            if method != 'weighted':
                # Save as pickle
                with open(f'data/enhanced/{method}_balanced_data.pkl', 'wb') as f:
                    pickle.dump(data, f)
                
                # Save as CSV for inspection
                df = pd.DataFrame(data['X'], columns=[
                    'mutation_count', 'expression_mean', 'cnv_mean',
                    'degree_ppi', 'degree_pathway', 'degree_coexpression'
                ])
                df['label'] = data['y']
                df.to_csv(f'data/enhanced/{method}_balanced_data.csv', index=False)
                
                print(f"Saved {method} balanced dataset: {len(data['X'])} samples")
        
        # Save class weights separately
        if 'weighted' in self.balanced_datasets:
            with open('data/enhanced/class_weights.pkl', 'wb') as f:
                pickle.dump(self.balanced_datasets['weighted']['class_weights'], f)
            print("Saved class weights")

def main():
    """Main function to create balanced datasets"""
    print("="*80)
    print("CREATING BALANCED DATASETS TO FIX CLASS IMBALANCE")
    print("="*80)
    
    # Initialize dataset creator
    creator = BalancedDatasetCreator()
    
    # Load original data
    creator.load_original_data()
    
    # Create various balanced datasets
    creator.create_manual_oversampling_dataset()
    creator.create_moderate_balance_dataset()
    creator.create_undersampled_dataset()
    creator.create_weighted_dataset()
    
    # Try SMOTE and ADASYN with adjusted parameters
    creator.create_smote_balanced_dataset(k_neighbors=1)
    creator.create_adasyn_balanced_dataset(k_neighbors=1)
    creator.create_hybrid_balanced_dataset()
    
    # Visualize comparison
    creator.visualize_balance_comparison()
    
    # Save datasets
    creator.save_balanced_datasets()
    
    print("\n" + "="*80)
    print("BALANCED DATASETS CREATED SUCCESSFULLY!")
    print("="*80)
    print("\nAvailable balanced datasets:")
    for method, data in creator.balanced_datasets.items():
        print(f"  - {data['method']}: {len(data['X'])} samples")
    
    print("\nNext steps:")
    print("1. Use these balanced datasets in training")
    print("2. Implement focal loss for additional improvement")
    print("3. Test different balancing methods")
    print("4. Compare results with paper performance")

if __name__ == "__main__":
    main() 