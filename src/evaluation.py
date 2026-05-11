# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os

def plot_metrics(y_true, y_probs, save_dir='reports'):
    """
    Generate ROC, PR curves and Confusion Matrix.
    y_true: Ground truth labels (0 or 1)
    y_probs: Predicted probabilities
    """
    os.makedirs(save_dir, exist_ok=True)
    y_pred = (y_probs >= 0.5).astype(int)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Evaluation plots saved to {save_dir}")

def run_evaluation_demo():
    # Create some dummy evaluation data
    # In a real run, this would be model.predict(test_data)
    y_true = np.random.randint(0, 2, 100)
    y_probs = np.random.rand(100)
    
    plot_metrics(y_true, y_probs)

if __name__ == "__main__":
    run_evaluation_demo()
