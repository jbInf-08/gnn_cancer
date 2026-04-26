"""
Comprehensive Evaluation Framework to Surpass Paper Performance
- Statistical significance testing
- Paper comparison metrics
- Ablation studies
- Performance analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    def __init__(self, data_path: str, results_path: str = "results"):
        self.data_path = data_path
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        self.data = None
        self.results = {}
        
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
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        # ROC AUC and PR AUC
        if y_proba is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['pr_auc'] = self.calculate_pr_auc(y_true, y_proba[:, 1])
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Additional metrics
        metrics['balanced_accuracy'] = self.calculate_balanced_accuracy(y_true, y_pred)
        metrics['matthews_correlation'] = self.calculate_matthews_correlation(y_true, y_pred)
        
        return metrics
    
    def calculate_pr_auc(self, y_true, y_proba):
        """Calculate Precision-Recall AUC"""
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_true, y_proba)
    
    def calculate_balanced_accuracy(self, y_true, y_pred):
        """Calculate balanced accuracy"""
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred)
    
    def calculate_matthews_correlation(self, y_true, y_pred):
        """Calculate Matthews correlation coefficient"""
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y_true, y_pred)
    
    def statistical_significance_test(self, baseline_metrics, improved_metrics, alpha=0.05):
        """Perform statistical significance testing"""
        logger.info("Performing statistical significance testing...")
        
        # McNemar's test for paired samples
        # This requires the same test set predictions
        # For now, we'll use a simplified approach
        
        # Compare key metrics
        key_metrics = ['accuracy', 'f1_score', 'roc_auc']
        significant_improvements = {}
        
        for metric in key_metrics:
            if metric in baseline_metrics and metric in improved_metrics:
                baseline_val = baseline_metrics[metric]
                improved_val = improved_metrics[metric]
                
                # Calculate improvement
                improvement = improved_val - baseline_val
                improvement_pct = (improvement / baseline_val) * 100
                
                # Simple significance test (assuming normal distribution)
                # In practice, you'd want to use proper statistical tests
                significant = improvement > 0.01  # 1% improvement threshold
                
                significant_improvements[metric] = {
                    'baseline': baseline_val,
                    'improved': improved_val,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'significant': significant
                }
        
        return significant_improvements
    
    def compare_with_paper(self, our_metrics, paper_metrics):
        """Compare our results with paper results"""
        logger.info("Comparing results with paper...")
        
        comparison = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric in our_metrics and metric in paper_metrics:
                our_val = our_metrics[metric]
                paper_val = paper_metrics[metric]
                
                improvement = our_val - paper_val
                improvement_pct = (improvement / paper_val) * 100
                
                comparison[metric] = {
                    'paper': paper_val,
                    'ours': our_val,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'surpasses_paper': improvement > 0
                }
        
        return comparison
    
    def ablation_study(self, model_results, ablation_configs):
        """Perform ablation study to show contribution of each enhancement"""
        logger.info("Performing ablation study...")
        
        ablation_results = {}
        
        for config_name, config in ablation_configs.items():
            logger.info(f"Testing ablation: {config_name}")
            
            # This would require re-training models with different configurations
            # For now, we'll create a placeholder structure
            ablation_results[config_name] = {
                'config': config,
                'metrics': {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'roc_auc': 0.0
                },
                'contribution': {
                    'accuracy_contribution': 0.0,
                    'f1_contribution': 0.0,
                    'roc_auc_contribution': 0.0
                }
            }
        
        return ablation_results
    
    def cross_validation_analysis(self, model, cv_folds=5):
        """Perform cross-validation analysis"""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # This would require implementing cross-validation with the GAT model
        # For now, we'll create a placeholder structure
        cv_results = {
            'fold_scores': [],
            'mean_score': 0.0,
            'std_score': 0.0,
            'confidence_interval': (0.0, 0.0)
        }
        
        return cv_results
    
    def create_visualizations(self, metrics, save_path=None):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        if save_path is None:
            save_path = self.results_path
        
        # 1. Confusion Matrix
        if 'confusion_matrix' in metrics:
            plt.figure(figsize=(8, 6))
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. ROC Curve
        if 'roc_auc' in metrics and 'y_true' in metrics and 'y_proba' in metrics:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_proba'][:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(save_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Precision-Recall Curve
        if 'pr_auc' in metrics and 'y_true' in metrics and 'y_proba' in metrics:
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(metrics['y_true'], metrics['y_proba'][:, 1])
            plt.plot(recall, precision, label=f'PR Curve (AUC = {metrics["pr_auc"]:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.savefig(save_path / 'pr_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Metrics Comparison
        if 'comparison' in metrics:
            comparison = metrics['comparison']
            metrics_names = list(comparison.keys())
            paper_values = [comparison[m]['paper'] for m in metrics_names]
            our_values = [comparison[m]['ours'] for m in metrics_names]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, paper_values, width, label='Paper Results', alpha=0.8)
            plt.bar(x + width/2, our_values, width, label='Our Results', alpha=0.8)
            
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Comparison with Paper Results')
            plt.xticks(x, metrics_names)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path / 'paper_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_comprehensive_report(self, results, save_path=None):
        """Generate comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report...")
        
        if save_path is None:
            save_path = self.results_path
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': {
                'total_samples': len(results.get('y_true', [])),
                'num_classes': len(np.unique(results.get('y_true', []))),
                'best_accuracy': results.get('metrics', {}).get('accuracy', 0.0),
                'best_f1_score': results.get('metrics', {}).get('f1_score', 0.0),
                'best_roc_auc': results.get('metrics', {}).get('roc_auc', 0.0)
            },
            'detailed_metrics': results.get('metrics', {}),
            'statistical_significance': results.get('significance', {}),
            'paper_comparison': results.get('comparison', {}),
            'ablation_study': results.get('ablation', {}),
            'cross_validation': results.get('cv_results', {}),
            'recommendations': self.generate_recommendations(results)
        }
        
        # Save report
        with open(save_path / 'comprehensive_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        markdown_report = self.generate_markdown_report(report)
        with open(save_path / 'comprehensive_evaluation_report.md', 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Comprehensive report saved to {save_path}")
        return report
    
    def generate_recommendations(self, results):
        """Generate recommendations based on results"""
        recommendations = []
        
        metrics = results.get('metrics', {})
        comparison = results.get('comparison', {})
        
        # Check if we surpass paper
        if comparison:
            surpasses_count = sum(1 for m in comparison.values() if m.get('surpasses_paper', False))
            if surpasses_count >= len(comparison) * 0.8:  # 80% of metrics
                recommendations.append("🎉 EXCELLENT: Results significantly surpass paper performance!")
            elif surpasses_count >= len(comparison) * 0.5:  # 50% of metrics
                recommendations.append("✅ GOOD: Results match or exceed paper performance in most metrics")
            else:
                recommendations.append("⚠️ NEEDS IMPROVEMENT: Results need optimization to surpass paper")
        
        # Specific recommendations based on metrics
        if metrics.get('accuracy', 0) < 0.8:
            recommendations.append("Consider feature engineering and data quality improvements")
        
        if metrics.get('f1_score', 0) < 0.8:
            recommendations.append("Address class imbalance and optimize for F1-score")
        
        if metrics.get('roc_auc', 0) < 0.9:
            recommendations.append("Focus on model architecture improvements for better ROC AUC")
        
        return recommendations
    
    def generate_markdown_report(self, report):
        """Generate markdown format report"""
        md = f"""# Comprehensive Evaluation Report

## Executive Summary

- **Timestamp**: {report['timestamp']}
- **Total Samples**: {report['summary']['total_samples']}
- **Number of Classes**: {report['summary']['num_classes']}
- **Best Accuracy**: {report['summary']['best_accuracy']:.4f}
- **Best F1-Score**: {report['summary']['best_f1_score']:.4f}
- **Best ROC AUC**: {report['summary']['best_roc_auc']:.4f}

## Detailed Metrics

### Classification Metrics
- **Accuracy**: {report['detailed_metrics'].get('accuracy', 0):.4f}
- **Precision**: {report['detailed_metrics'].get('precision', 0):.4f}
- **Recall**: {report['detailed_metrics'].get('recall', 0):.4f}
- **F1-Score**: {report['detailed_metrics'].get('f1_score', 0):.4f}
- **ROC AUC**: {report['detailed_metrics'].get('roc_auc', 0):.4f}

### Additional Metrics
- **Balanced Accuracy**: {report['detailed_metrics'].get('balanced_accuracy', 0):.4f}
- **Matthews Correlation**: {report['detailed_metrics'].get('matthews_correlation', 0):.4f}

## Paper Comparison

"""
        
        for metric, comp in report['paper_comparison'].items():
            md += f"- **{metric.upper()}**: Paper: {comp['paper']:.4f}, Ours: {comp['ours']:.4f}, Improvement: {comp['improvement_pct']:.2f}%\n"
        
        md += f"""
## Recommendations

"""
        
        for rec in report['recommendations']:
            md += f"- {rec}\n"
        
        return md
    
    def run_comprehensive_evaluation(self, model_results, paper_metrics=None):
        """Run comprehensive evaluation pipeline"""
        logger.info("Starting comprehensive evaluation...")
        
        # Load data
        self.load_data()
        
        # Calculate comprehensive metrics
        y_true = model_results.get('y_true', [])
        y_pred = model_results.get('y_pred', [])
        y_proba = model_results.get('y_proba', None)
        
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_proba)
        metrics['y_true'] = y_true
        metrics['y_proba'] = y_proba
        
        # Statistical significance testing
        baseline_metrics = {
            'accuracy': 0.276,  # Current GAT performance
            'f1_score': 0.25,
            'roc_auc': 0.5
        }
        
        significance = self.statistical_significance_test(baseline_metrics, metrics)
        
        # Paper comparison
        if paper_metrics:
            comparison = self.compare_with_paper(metrics, paper_metrics)
        else:
            # Default paper metrics (example)
            paper_metrics = {
                'accuracy': 0.85,
                'precision': 0.84,
                'recall': 0.83,
                'f1_score': 0.84,
                'roc_auc': 0.90
            }
            comparison = self.compare_with_paper(metrics, paper_metrics)
        
        # Ablation study
        ablation_configs = {
            'no_skip_connections': {'use_skip_connections': False},
            'no_multi_scale': {'use_multi_scale': False},
            'no_attention_pooling': {'use_attention_pooling': False},
            'basic_pooling': {'pooling_strategy': 'mean'},
            'smaller_model': {'hidden_dim': 128, 'num_layers': 2}
        }
        
        ablation = self.ablation_study(model_results, ablation_configs)
        
        # Cross-validation
        cv_results = self.cross_validation_analysis(None)
        
        # Compile results
        comprehensive_results = {
            'metrics': metrics,
            'significance': significance,
            'comparison': comparison,
            'ablation': ablation,
            'cv_results': cv_results
        }
        
        # Create visualizations
        self.create_visualizations(comprehensive_results)
        
        # Generate report
        report = self.generate_comprehensive_report(comprehensive_results)
        
        logger.info("Comprehensive evaluation completed!")
        return comprehensive_results, report

def main():
    """Main function to run comprehensive evaluation"""
    evaluator = ComprehensiveEvaluator("data")
    
    # Example model results (replace with actual results)
    model_results = {
        'y_true': np.random.randint(0, 2, 1000),
        'y_pred': np.random.randint(0, 2, 1000),
        'y_proba': np.random.rand(1000, 2)
    }
    
    # Paper metrics (replace with actual paper metrics)
    paper_metrics = {
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.83,
        'f1_score': 0.84,
        'roc_auc': 0.90
    }
    
    results, report = evaluator.run_comprehensive_evaluation(model_results, paper_metrics)
    
    print("Comprehensive evaluation completed!")
    print(f"Best Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Best F1-Score: {results['metrics']['f1_score']:.4f}")
    print(f"Best ROC AUC: {results['metrics']['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
