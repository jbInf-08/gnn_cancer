import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import numpy as np

def plot_learning_curves(train_losses, val_losses, model_name="model"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curves for {model_name}')
    plt.legend()
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_learning_curves.png')
    plt.close()

def plot_roc_curves(y_true, y_score, n_classes, model_name="model"):
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_roc_curves.png')
    plt.close()

def plot_pr_curves(y_true, y_score, n_classes, model_name="model"):
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_score[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_pr_curves.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names=None, model_name="model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close() 