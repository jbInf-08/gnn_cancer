import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for GNN models
    """
    
    def __init__(self, data_path: str = "data/enhanced_multi_modal/enhanced_multi_modal_data.pt"):
        self.data_path = Path(data_path)
        self.results_dir = Path("results/hyperparameter_tuning")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.data = self._load_data()
        
        # Define hyperparameter grids
        self.hyperparameter_grids = {
            'GAT': {
                'hidden_dim': [64, 128, 256],
                'num_layers': [2, 3, 4],
                'num_heads': [4, 8, 16],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01, 0.1],
                'weight_decay': [1e-4, 1e-3, 1e-2],
                'batch_norm': [True, False],
                'skip_connections': [True, False]
            },
            'GCN': {
                'hidden_dim': [64, 128, 256],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01, 0.1],
                'weight_decay': [1e-4, 1e-3, 1e-2],
                'batch_norm': [True, False],
                'skip_connections': [True, False]
            },
            'GraphSAGE': {
                'hidden_dim': [64, 128, 256],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01, 0.1],
                'weight_decay': [1e-4, 1e-3, 1e-2],
                'batch_norm': [True, False],
                'skip_connections': [True, False],
                'aggregator': ['mean', 'max', 'sum']
            }
        }
        
        # Training parameters
        self.training_params = {
            'num_epochs': 100,
            'patience': 20,
            'k_folds': 5,
            'random_state': 42
        }
    
    def _load_data(self) -> Data:
        """Load the enhanced multi-modal data"""
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return None
        
        try:
            data = torch.load(self.data_path)
            logger.info(f"Loaded data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def create_model(self, model_type: str, input_dim: int, num_classes: int, **kwargs) -> torch.nn.Module:
        """Create a GNN model with given parameters"""
        if model_type == 'GAT':
            return GATModel(input_dim, num_classes, **kwargs)
        elif model_type == 'GCN':
            return GCNModel(input_dim, num_classes, **kwargs)
        elif model_type == 'GraphSAGE':
            return GraphSAGEModel(input_dim, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def generate_hyperparameter_combinations(self, model_type: str) -> List[Dict]:
        """Generate all combinations of hyperparameters for a model type"""
        grid = self.hyperparameter_grids[model_type]
        keys = grid.keys()
        values = grid.values()
        combinations = []
        
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} hyperparameter combinations for {model_type}")
        return combinations
    
    def train_and_evaluate(self, model: torch.nn.Module, train_data: Data, val_data: Data, 
                          test_data: Data, **training_kwargs) -> Dict[str, float]:
        """Train and evaluate a model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Move data to device
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=training_kwargs.get('learning_rate', 0.01),
            weight_decay=training_kwargs.get('weight_decay', 1e-4)
        )
        
        # Loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.training_params['num_epochs']):
            # Training
            model.train()
            optimizer.zero_grad()
            out = model(train_data.x, train_data.edge_index)
            loss = criterion(out, train_data.y)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x, val_data.edge_index)
                val_loss = criterion(val_out, val_data.y)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.training_params['patience']:
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_out = model(test_data.x, test_data.edge_index)
            test_probs = F.softmax(test_out, dim=1)
            test_preds = test_out.argmax(dim=1)
            
            # Calculate metrics
            accuracy = accuracy_score(test_data.y.cpu(), test_preds.cpu())
            precision = precision_score(test_data.y.cpu(), test_preds.cpu(), average='weighted', zero_division=0)
            recall = recall_score(test_data.y.cpu(), test_preds.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(test_data.y.cpu(), test_preds.cpu(), average='weighted', zero_division=0)
            
            # ROC AUC (if binary classification)
            if test_data.num_classes == 2:
                try:
                    roc_auc = roc_auc_score(test_data.y.cpu(), test_probs[:, 1].cpu())
                except:
                    roc_auc = 0.5
            else:
                roc_auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'val_loss': best_val_loss.item()
        }
    
    def cross_validate(self, model_type: str, hyperparams: Dict) -> Dict[str, float]:
        """Perform k-fold cross-validation for a set of hyperparameters"""
        logger.info(f"Cross-validating {model_type} with hyperparams: {hyperparams}")
        
        # Create k-fold splits
        kfold = StratifiedKFold(
            n_splits=self.training_params['k_folds'], 
            shuffle=True, 
            random_state=self.training_params['random_state']
        )
        
        fold_metrics = []
        
        for fold, (train_val_idx, test_idx) in enumerate(kfold.split(
            np.arange(self.data.num_nodes), self.data.y.cpu().numpy()
        )):
            # Split train_val into train and validation
            train_val_data = self.data.clone()
            train_val_data.x = self.data.x[train_val_idx]
            train_val_data.y = self.data.y[train_val_idx]
            train_val_data.edge_index = self._get_subgraph_edges(train_val_idx)
            
            # Further split into train and validation
            val_size = len(train_val_idx) // 5
            train_idx = train_val_idx[:-val_size]
            val_idx = train_val_idx[-val_size:]
            
            # Create train, validation, and test data
            train_data = self.data.clone()
            train_data.x = self.data.x[train_idx]
            train_data.y = self.data.y[train_idx]
            train_data.edge_index = self._get_subgraph_edges(train_idx)
            
            val_data = self.data.clone()
            val_data.x = self.data.x[val_idx]
            val_data.y = self.data.y[val_idx]
            val_data.edge_index = self._get_subgraph_edges(val_idx)
            
            test_data = self.data.clone()
            test_data.x = self.data.x[test_idx]
            test_data.y = self.data.y[test_idx]
            test_data.edge_index = self._get_subgraph_edges(test_idx)
            
            # Create and train model
            model = self.create_model(
                model_type, 
                self.data.x.shape[1], 
                self.data.num_classes, 
                **hyperparams
            )
            
            # Train and evaluate
            metrics = self.train_and_evaluate(model, train_data, val_data, test_data, **hyperparams)
            fold_metrics.append(metrics)
            
            logger.info(f"Fold {fold + 1}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # Average metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
            avg_metrics[f'{metric}_std'] = np.std([fold[metric] for fold in fold_metrics])
        
        return avg_metrics
    
    def _get_subgraph_edges(self, node_indices: np.ndarray) -> torch.Tensor:
        """Get edges for a subgraph containing only the specified nodes"""
        node_set = set(node_indices)
        edge_list = []
        
        for i in range(self.data.edge_index.shape[1]):
            src, dst = self.data.edge_index[:, i].cpu().numpy()
            if src in node_set and dst in node_set:
                # Remap indices
                new_src = np.where(node_indices == src)[0][0]
                new_dst = np.where(node_indices == dst)[0][0]
                edge_list.append([new_src, new_dst])
        
        if edge_list:
            return torch.tensor(edge_list, dtype=torch.long).t()
        else:
            return torch.tensor([[], []], dtype=torch.long)
    
    def tune_hyperparameters(self, model_type: str, max_combinations: int = 50) -> Dict:
        """Perform hyperparameter tuning for a specific model type"""
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        # Generate hyperparameter combinations
        combinations = self.generate_hyperparameter_combinations(model_type)
        
        # Limit combinations if too many
        if len(combinations) > max_combinations:
            # Randomly sample combinations
            np.random.seed(self.training_params['random_state'])
            combinations = np.random.choice(combinations, max_combinations, replace=False)
            logger.info(f"Sampled {max_combinations} combinations for tuning")
        
        results = []
        
        for i, hyperparams in enumerate(combinations):
            logger.info(f"Testing combination {i + 1}/{len(combinations)}")
            
            try:
                # Perform cross-validation
                metrics = self.cross_validate(model_type, hyperparams)
                
                # Store results
                result = {
                    'model_type': model_type,
                    'hyperparameters': hyperparams,
                    'metrics': metrics
                }
                results.append(result)
                
                logger.info(f"Combination {i + 1} - F1: {metrics['f1_score']:.4f} ± {metrics['f1_score_std']:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing combination {i + 1}: {e}")
                continue
        
        # Find best hyperparameters
        if results:
            best_result = max(results, key=lambda x: x['metrics']['f1_score'])
            logger.info(f"Best {model_type} hyperparameters: {best_result['hyperparameters']}")
            logger.info(f"Best F1 score: {best_result['metrics']['f1_score']:.4f}")
        
        return results
    
    def tune_all_models(self) -> Dict:
        """Perform hyperparameter tuning for all model types"""
        all_results = {}
        
        for model_type in ['GAT', 'GCN', 'GraphSAGE']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Tuning {model_type}")
            logger.info(f"{'='*50}")
            
            results = self.tune_hyperparameters(model_type)
            all_results[model_type] = results
            
            # Save results for this model
            results_file = self.results_dir / f"{model_type}_tuning_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Create summary
        summary = self.create_tuning_summary(all_results)
        
        # Save summary
        summary_file = self.results_dir / "tuning_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return all_results
    
    def create_tuning_summary(self, all_results: Dict) -> Dict:
        """Create a summary of all tuning results"""
        summary = {
            'best_models': {},
            'model_comparison': {},
            'tuning_statistics': {}
        }
        
        for model_type, results in all_results.items():
            if not results:
                continue
            
            # Find best result
            best_result = max(results, key=lambda x: x['metrics']['f1_score'])
            summary['best_models'][model_type] = {
                'hyperparameters': best_result['hyperparameters'],
                'metrics': best_result['metrics']
            }
            
            # Model comparison
            summary['model_comparison'][model_type] = {
                'best_f1': best_result['metrics']['f1_score'],
                'best_accuracy': best_result['metrics']['accuracy'],
                'best_precision': best_result['metrics']['precision'],
                'best_recall': best_result['metrics']['recall'],
                'best_roc_auc': best_result['metrics']['roc_auc']
            }
            
            # Tuning statistics
            f1_scores = [r['metrics']['f1_score'] for r in results]
            summary['tuning_statistics'][model_type] = {
                'num_combinations_tested': len(results),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'min_f1': np.min(f1_scores),
                'max_f1': np.max(f1_scores)
            }
        
        return summary

# Model definitions
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=3, 
                 num_heads=8, dropout=0.2, batch_norm=True, skip_connections=True):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        if batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
            if batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            if hasattr(self, 'batch_norms') and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.skip_connections and x.shape == identity.shape:
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=3, 
                 dropout=0.2, batch_norm=True, skip_connections=True):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, num_classes))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            if hasattr(self, 'batch_norms') and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.skip_connections and x.shape == identity.shape:
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=3, 
                 dropout=0.2, batch_norm=True, skip_connections=True, aggregator='mean'):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        if batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            if batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggregator))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Input layer
        x = self.convs[0](x, edge_index)
        if hasattr(self, 'batch_norms') and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            if hasattr(self, 'batch_norms') and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.skip_connections and x.shape == identity.shape:
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x

def main():
    """Main function to run hyperparameter tuning"""
    logger.info("Starting comprehensive hyperparameter tuning")
    
    tuner = HyperparameterTuner()
    
    if tuner.data is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Run hyperparameter tuning for all models
    results = tuner.tune_all_models()
    
    logger.info("Hyperparameter tuning completed successfully")
    logger.info(f"Results saved to {tuner.results_dir}")

if __name__ == "__main__":
    main() 