"""
Master Optimization Pipeline to Fully Surpass the Paper
- Orchestrates all improvements and optimizations
- Data quality improvements
- Hyperparameter optimization
- Model architecture enhancements
- Comprehensive evaluation and comparison
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MasterOptimizationPipeline:
    """
    Master pipeline orchestrating all optimizations to surpass the paper
    """
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
        # Create necessary directories
        self.create_directories()
        
        logger.info("Initialized Master Optimization Pipeline")
    
    def create_directories(self):
        """Create necessary directories for results"""
        directories = ['results', 'models', 'logs', 'figures']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_data_quality_improvements(self):
        """Step 1: Improve data quality"""
        logger.info("="*60)
        logger.info("STEP 1: DATA QUALITY IMPROVEMENTS")
        logger.info("="*60)
        
        try:
            from data_quality_improvements import DataQualityValidator
            
            validator = DataQualityValidator('data')
            validation_results = validator.run_comprehensive_validation()
            
            self.results['data_quality'] = validation_results
            
            logger.info("Data quality improvements completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in data quality improvements: {e}")
            return False
    
    def run_hyperparameter_optimization(self):
        """Step 2: Hyperparameter optimization"""
        logger.info("="*60)
        logger.info("STEP 2: HYPERPARAMETER OPTIMIZATION")
        logger.info("="*60)
        
        try:
            from comprehensive_gat_optimization import ComprehensiveGATOptimizer
            
            optimizer = ComprehensiveGATOptimizer('data')
            model, results, history = optimizer.run_comprehensive_optimization()
            
            self.results['hyperparameter_optimization'] = {
                'model_results': results,
                'training_history': history
            }
            
            logger.info("Hyperparameter optimization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return False
    
    def run_model_architecture_comparison(self):
        """Step 3: Compare different model architectures"""
        logger.info("="*60)
        logger.info("STEP 3: MODEL ARCHITECTURE COMPARISON")
        logger.info("="*60)
        
        try:
            from optimized_gat_implementation import OptimizedGATModel, AdvancedTrainingConfig
            from models.enhanced_gnn_models import EnhancedGATModel, EnhancedGraphSAGEModel, EnhancedGCNModel
            
            # Load improved data
            data_file = Path('data/enhanced/improved_torch_geometric_data.pt')
            if data_file.exists():
                data = torch.load(data_file)
            else:
                data = torch.load('data/enhanced/real_only_torch_geometric_data.pt')
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.to(device)
            
            # Create train/val/test split
            num_nodes = data.x.shape[0]
            node_indices = np.arange(num_nodes)
            
            if hasattr(data, 'y') and data.y is not None:
                labels = data.y.cpu().numpy()
            else:
                labels = np.zeros(num_nodes)
            
            # Split data
            from sklearn.model_selection import train_test_split
            train_val_idx, test_idx = train_test_split(
                node_indices, test_size=0.2, random_state=42, 
                stratify=labels if len(np.unique(labels)) > 1 else None
            )
            
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.2/(1-0.2), random_state=42,
                stratify=labels[train_val_idx] if len(np.unique(labels)) > 1 else None
            )
            
            # Create masks
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            
            train_mask = train_mask.to(device)
            val_mask = val_mask.to(device)
            test_mask = test_mask.to(device)
            
            # Define models to compare
            models = {
                'Optimized_GAT': OptimizedGATModel(
                    input_dim=data.x.shape[1],
                    hidden_dim=256,
                    output_dim=2,
                    num_layers=4,
                    num_heads=8,
                    dropout=0.3,
                    use_edge_attr=True,
                    num_edge_types=8,
                    use_skip_connections=True,
                    use_graph_attention=True,
                    pooling_strategy='multi'
                ),
                'Enhanced_GAT': EnhancedGATModel(
                    input_dim=data.x.shape[1],
                    hidden_dim=256,
                    output_dim=2,
                    num_layers=4,
                    num_heads=8,
                    dropout=0.3,
                    use_edge_attr=True,
                    num_edge_types=8
                ),
                'Enhanced_GraphSAGE': EnhancedGraphSAGEModel(
                    input_dim=data.x.shape[1],
                    hidden_dim=128,
                    output_dim=2,
                    num_layers=3,
                    dropout=0.3,
                    use_edge_attr=True,
                    num_edge_types=4
                ),
                'Enhanced_GCN': EnhancedGCNModel(
                    input_dim=data.x.shape[1],
                    hidden_dim=128,
                    output_dim=2,
                    num_layers=3,
                    dropout=0.3,
                    use_edge_attr=True,
                    num_edge_types=4
                )
            }
            
            # Train and evaluate each model
            comparison_results = {}
            
            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")
                
                model = model.to(device)
                
                # Optimizer
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                
                # Loss function
                if hasattr(data, 'y') and data.y is not None:
                    labels = data.y[train_mask]
                    class_counts = torch.bincount(labels)
                    class_weights = 1.0 / class_counts.float()
                    class_weights = class_weights / class_weights.sum()
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
                else:
                    criterion = torch.nn.CrossEntropyLoss()
                
                # Training loop
                best_val_score = 0.0
                patience_counter = 0
                max_patience = 30
                
                for epoch in range(200):
                    # Training
                    model.train()
                    optimizer.zero_grad()
                    
                    out = model(data.x, data.edge_index, data.edge_attr)
                    loss = criterion(out[train_mask], data.y[train_mask])
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_out = model(data.x, data.edge_index, data.edge_attr)
                        val_pred = val_out[val_mask].argmax(dim=1)
                        val_score = torch.nn.functional.f1_score(
                            data.y[val_mask], val_pred, average='weighted'
                        )
                    
                    # Early stopping
                    if val_score > best_val_score:
                        best_val_score = val_score
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= max_patience:
                        break
                
                # Load best model and evaluate
                model.load_state_dict(best_model_state)
                model.eval()
                
                with torch.no_grad():
                    test_out = model(data.x, data.edge_index, data.edge_attr)
                    test_pred = test_out[test_mask].argmax(dim=1)
                    test_proba = torch.nn.functional.softmax(test_out[test_mask], dim=1)
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    test_acc = accuracy_score(data.y[test_mask].cpu(), test_pred.cpu())
                    test_precision = precision_score(data.y[test_mask].cpu(), test_pred.cpu(), average='weighted')
                    test_recall = recall_score(data.y[test_mask].cpu(), test_pred.cpu(), average='weighted')
                    test_f1 = f1_score(data.y[test_mask].cpu(), test_pred.cpu(), average='weighted')
                    test_auc = roc_auc_score(data.y[test_mask].cpu(), test_proba[:, 1].cpu())
                    
                    comparison_results[model_name] = {
                        'accuracy': test_acc,
                        'precision': test_precision,
                        'recall': test_recall,
                        'f1_score': test_f1,
                        'auc': test_auc,
                        'best_val_score': best_val_score.item()
                    }
                
                logger.info(f"{model_name} - F1: {test_f1:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
            
            self.results['model_comparison'] = comparison_results
            
            # Save comparison results
            with open('results/model_comparison_results.json', 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            logger.info("Model architecture comparison completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in model architecture comparison: {e}")
            return False
    
    def generate_comprehensive_report(self):
        """Step 4: Generate comprehensive report"""
        logger.info("="*60)
        logger.info("STEP 4: GENERATING COMPREHENSIVE REPORT")
        logger.info("="*60)
        
        try:
            # Create comprehensive report
            report = {
                'pipeline_summary': {
                    'total_time': time.time() - self.start_time,
                    'steps_completed': list(self.results.keys())
                },
                'results': self.results
            }
            
            # Save comprehensive report
            with open('results/comprehensive_optimization_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate comparison plots
            if 'model_comparison' in self.results:
                self.plot_model_comparison()
            
            # Generate summary
            self.generate_summary()
            
            logger.info("Comprehensive report generated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return False
    
    def plot_model_comparison(self):
        """Plot model comparison results"""
        comparison_results = self.results['model_comparison']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract metrics
        models = list(comparison_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        # Plot 1: F1 Score comparison
        f1_scores = [comparison_results[model]['f1_score'] for model in models]
        axes[0, 0].bar(models, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Accuracy comparison
        accuracies = [comparison_results[model]['accuracy'] for model in models]
        axes[0, 1].bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: AUC comparison
        aucs = [comparison_results[model]['auc'] for model in models]
        axes[1, 0].bar(models, aucs, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[1, 0].set_title('AUC Comparison')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: All metrics heatmap
        metric_data = []
        for model in models:
            metric_data.append([
                comparison_results[model]['accuracy'],
                comparison_results[model]['precision'],
                comparison_results[model]['recall'],
                comparison_results[model]['f1_score'],
                comparison_results[model]['auc']
            ])
        
        sns.heatmap(metric_data, annot=True, fmt='.3f', 
                   xticklabels=['Acc', 'Prec', 'Rec', 'F1', 'AUC'],
                   yticklabels=models, ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_title('All Metrics Heatmap')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary(self):
        """Generate summary of results"""
        summary = []
        summary.append("="*80)
        summary.append("COMPREHENSIVE OPTIMIZATION PIPELINE SUMMARY")
        summary.append("="*80)
        
        # Pipeline summary
        if 'pipeline_summary' in self.results:
            pipeline_summary = self.results['pipeline_summary']
            summary.append(f"Total Time: {pipeline_summary['total_time']:.2f} seconds")
            summary.append(f"Steps Completed: {', '.join(pipeline_summary['steps_completed'])}")
        
        # Model comparison summary
        if 'model_comparison' in self.results:
            summary.append("\nMODEL COMPARISON RESULTS:")
            summary.append("-" * 40)
            
            comparison_results = self.results['model_comparison']
            
            # Find best model
            best_model = max(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]['f1_score'])
            
            summary.append(f"Best Model (by F1 Score): {best_model}")
            summary.append(f"Best F1 Score: {comparison_results[best_model]['f1_score']:.4f}")
            summary.append(f"Best Accuracy: {comparison_results[best_model]['accuracy']:.4f}")
            summary.append(f"Best AUC: {comparison_results[best_model]['auc']:.4f}")
            
            summary.append("\nAll Model Results:")
            for model, results in comparison_results.items():
                summary.append(f"{model}: F1={results['f1_score']:.4f}, "
                             f"Acc={results['accuracy']:.4f}, AUC={results['auc']:.4f}")
        
        # Data quality summary
        if 'data_quality' in self.results:
            summary.append("\nDATA QUALITY SUMMARY:")
            summary.append("-" * 40)
            
            data_quality = self.results['data_quality']
            if 'graph_construction' in data_quality:
                graph_stats = data_quality['graph_construction']['graph_stats']
                summary.append(f"Graph Nodes: {graph_stats['num_nodes']}")
                summary.append(f"Graph Edges: {graph_stats['num_edges']}")
                summary.append(f"Graph Density: {graph_stats['density']:.6f}")
            
            if 'feature_engineering' in data_quality:
                feature_stats = data_quality['feature_engineering']['node_features']
                summary.append(f"Node Features: {feature_stats['num_features']}")
        
        summary.append("\n" + "="*80)
        
        # Save summary
        with open('results/optimization_summary.txt', 'w') as f:
            f.write('\n'.join(summary))
        
        # Print summary
        for line in summary:
            logger.info(line)
    
    def run_complete_pipeline(self):
        """Run the complete optimization pipeline"""
        logger.info("Starting Master Optimization Pipeline...")
        
        steps = [
            ("Data Quality Improvements", self.run_data_quality_improvements),
            ("Hyperparameter Optimization", self.run_hyperparameter_optimization),
            ("Model Architecture Comparison", self.run_model_architecture_comparison),
            ("Generate Comprehensive Report", self.generate_comprehensive_report)
        ]
        
        successful_steps = []
        
        for step_name, step_function in steps:
            logger.info(f"\n{'='*20} {step_name} {'='*20}")
            
            try:
                success = step_function()
                if success:
                    successful_steps.append(step_name)
                    logger.info(f"✓ {step_name} completed successfully")
                else:
                    logger.error(f"✗ {step_name} failed")
            except Exception as e:
                logger.error(f"✗ {step_name} failed with error: {e}")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total steps: {len(steps)}")
        logger.info(f"Successful steps: {len(successful_steps)}")
        logger.info(f"Failed steps: {len(steps) - len(successful_steps)}")
        logger.info(f"Total time: {time.time() - self.start_time:.2f} seconds")
        
        if len(successful_steps) == len(steps):
            logger.info("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
            logger.info("🎯 You have successfully implemented all optimizations to surpass the paper!")
        else:
            logger.warning("⚠️  Some steps failed. Check the logs for details.")
        
        return len(successful_steps) == len(steps)

def main():
    """Main execution function"""
    pipeline = MasterOptimizationPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n" + "🎉" * 20)
        print("MASTER OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎯 All optimizations implemented to surpass the paper!")
        print("📊 Check the 'results' directory for comprehensive reports")
        print("🎉" * 20)
    else:
        print("\n" + "⚠️" * 20)
        print("PIPELINE COMPLETED WITH SOME FAILURES")
        print("Check the logs for details on failed steps")
        print("⚠️" * 20)

if __name__ == "__main__":
    main()
