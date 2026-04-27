"""
Strategies for class imbalance in graph / tabular oncology workflows.

Do not assume fixed cohort sizes (e.g. 967k nodes / 19 positives); those were legacy
documentation literals, not measured values from this file. Measure class counts from
your loaded tensor or manifest before choosing weights, focal loss, or resampling.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImbalancedDataStrategy:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """Load the data"""
        try:
            data_file = Path(self.data_path) / "enhanced" / "real_only_torch_geometric_data.pt"
            if data_file.exists():
                self.data = torch.load(data_file, weights_only=False)
                logger.info(f"Loaded data from {data_file}")
            else:
                raise FileNotFoundError(f"Data file not found: {data_file}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_imbalance(self):
        """Analyze the class imbalance"""
        logger.info("Analyzing class imbalance...")
        
        if hasattr(self.data, 'y') and self.data.y is not None:
            y_np = self.data.y.numpy()
            unique_labels, counts = np.unique(y_np, return_counts=True)
            
            logger.info(f"Class distribution:")
            for label, count in zip(unique_labels, counts):
                percentage = (count / len(y_np)) * 100
                logger.info(f"  Label {label}: {count} samples ({percentage:.4f}%)")
            
            # Calculate imbalance ratio
            if len(unique_labels) == 2:
                minority_class = min(counts)
                majority_class = max(counts)
                imbalance_ratio = majority_class / minority_class
                logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
                
                if imbalance_ratio > 1000:
                    logger.warning("EXTREME class imbalance detected!")
                    logger.warning("This requires specialized handling techniques.")
                
                return {
                    'unique_labels': unique_labels,
                    'counts': counts,
                    'imbalance_ratio': imbalance_ratio,
                    'minority_class': minority_class,
                    'majority_class': majority_class
                }
        
        return None
    
    def create_balanced_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Create balanced train/validation/test splits"""
        logger.info("Creating balanced data splits...")
        
        if not hasattr(self.data, 'y') or self.data.y is None:
            logger.error("No labels found in data")
            return None
        
        y_np = self.data.y.numpy()
        num_nodes = self.data.x.shape[0]
        
        # Find positive and negative samples
        positive_indices = np.where(y_np == 1)[0]
        negative_indices = np.where(y_np == 0)[0]
        
        logger.info(f"Positive samples: {len(positive_indices)}")
        logger.info(f"Negative samples: {len(negative_indices)}")
        
        # For extreme imbalance, we need to be very careful with splitting
        # Ensure positive samples are represented in all splits
        
        # Split positive samples
        np.random.seed(random_state)
        np.random.shuffle(positive_indices)
        
        # Calculate split sizes for positive samples
        n_positive = len(positive_indices)
        n_test_positive = max(1, int(n_positive * test_size))
        n_val_positive = max(1, int(n_positive * val_size))
        n_train_positive = n_positive - n_test_positive - n_val_positive
        
        # Split positive samples
        test_positive = positive_indices[:n_test_positive]
        val_positive = positive_indices[n_test_positive:n_test_positive + n_val_positive]
        train_positive = positive_indices[n_test_positive + n_val_positive:]
        
        # For negative samples, we can use more standard splitting
        # But ensure we don't overwhelm the positive samples
        max_negative_per_split = 1000  # Limit negative samples per split
        
        np.random.shuffle(negative_indices)
        n_negative = min(len(negative_indices), max_negative_per_split * 3)
        negative_indices = negative_indices[:n_negative]
        
        n_test_negative = int(n_negative * test_size)
        n_val_negative = int(n_negative * val_size)
        n_train_negative = n_negative - n_test_negative - n_val_negative
        
        test_negative = negative_indices[:n_test_negative]
        val_negative = negative_indices[n_test_negative:n_test_negative + n_val_negative]
        train_negative = negative_indices[n_test_negative + n_val_negative:]
        
        # Combine splits
        train_indices = np.concatenate([train_positive, train_negative])
        val_indices = np.concatenate([val_positive, val_negative])
        test_indices = np.concatenate([test_positive, test_negative])
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        logger.info(f"Balanced splits created:")
        logger.info(f"  Train: {train_mask.sum().item()} samples")
        logger.info(f"  Validation: {val_mask.sum().item()} samples")
        logger.info(f"  Test: {test_mask.sum().item()} samples")
        
        # Log class distribution in each split
        for split_name, mask in [("Train", train_mask), ("Validation", val_mask), ("Test", test_mask)]:
            split_y = self.data.y[mask]
            unique, counts = torch.unique(split_y, return_counts=True)
            logger.info(f"  {split_name} class distribution:")
            for label, count in zip(unique, counts):
                percentage = (count / len(split_y)) * 100
                logger.info(f"    Label {label}: {count} samples ({percentage:.2f}%)")
        
        return train_mask, val_mask, test_mask
    
    def compute_class_weights(self, train_mask):
        """Compute class weights for imbalanced data"""
        logger.info("Computing class weights...")
        
        train_y = self.data.y[train_mask].numpy()
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_y),
            y=train_y
        )
        
        # Convert to tensor
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        logger.info(f"Class weights: {class_weights_tensor}")
        
        return class_weights_tensor
    
    def create_focal_loss(self, alpha=1.0, gamma=2.0):
        """Create focal loss for handling class imbalance"""
        logger.info("Creating focal loss function...")
        
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    def create_weighted_sampler(self, train_mask):
        """Create weighted sampler for training"""
        logger.info("Creating weighted sampler...")
        
        train_y = self.data.y[train_mask].numpy()
        
        # Compute sample weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_y),
            y=train_y
        )
        
        # Assign weights to samples
        sample_weights = class_weights[train_y]
        
        # Create weighted sampler
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        logger.info("Weighted sampler created")
        return sampler
    
    def create_balanced_metrics(self):
        """Create metrics suitable for imbalanced data"""
        logger.info("Creating balanced evaluation metrics...")
        
        def balanced_accuracy(y_true, y_pred):
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_true, y_pred)
        
        def f1_macro(y_true, y_pred):
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='macro')
        
        def precision_macro(y_true, y_pred):
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, average='macro')
        
        def recall_macro(y_true, y_pred):
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, average='macro')
        
        return {
            'balanced_accuracy': balanced_accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
    
    def generate_recommendations(self):
        """Generate recommendations for handling extreme imbalance"""
        logger.info("Generating recommendations for extreme class imbalance...")
        
        recommendations = [
            "🎯 CRITICAL: Extreme class imbalance detected (0.002% positive rate)",
            "📊 Use balanced accuracy and macro-averaged metrics instead of accuracy",
            "⚖️ Implement class weighting in loss function",
            "🔄 Use focal loss to focus on hard examples",
            "🎲 Implement data augmentation for minority class",
            "📈 Use stratified sampling for train/val/test splits",
            "🔍 Focus on precision and recall rather than overall accuracy",
            "🎪 Consider ensemble methods with different sampling strategies",
            "📋 Use confusion matrix to understand model behavior",
            "🎯 Set realistic expectations - perfect accuracy is impossible with this imbalance"
        ]
        
        for rec in recommendations:
            logger.info(rec)
        
        return recommendations
    
    def run_imbalance_analysis(self):
        """Run complete imbalance analysis"""
        logger.info("Running complete imbalance analysis...")
        
        # Load data
        self.load_data()
        
        # Analyze imbalance
        imbalance_info = self.analyze_imbalance()
        
        if imbalance_info:
            # Create balanced splits
            splits = self.create_balanced_splits()
            
            if splits:
                train_mask, val_mask, test_mask = splits
                
                # Compute class weights
                class_weights = self.compute_class_weights(train_mask)
                
                # Create focal loss
                focal_loss = self.create_focal_loss()
                
                # Create balanced metrics
                metrics = self.create_balanced_metrics()
                
                # Generate recommendations
                recommendations = self.generate_recommendations()
                
                logger.info("Imbalance analysis completed!")
                
                return {
                    'imbalance_info': imbalance_info,
                    'splits': (train_mask, val_mask, test_mask),
                    'class_weights': class_weights,
                    'focal_loss': focal_loss,
                    'metrics': metrics,
                    'recommendations': recommendations
                }
        
        return None

def main():
    """Main function to run imbalance analysis"""
    strategy = ImbalancedDataStrategy("data")
    results = strategy.run_imbalance_analysis()
    
    if results:
        print("✅ Imbalance analysis completed successfully!")
        print(f"Imbalance ratio: {results['imbalance_info']['imbalance_ratio']:.2f}:1")
        print(f"Class weights: {results['class_weights']}")
    else:
        print("❌ Imbalance analysis failed!")

if __name__ == "__main__":
    main()
