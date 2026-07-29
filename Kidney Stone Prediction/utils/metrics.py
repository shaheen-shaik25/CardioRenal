import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def calculate_metrics(y_true, y_pred, y_probs=None):
    """Calculates evaluation metrics for medical classification."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    roc_auc = None
    if y_probs is not None and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc) if roc_auc is not None else 0.0,
        'confusion_matrix': cm.tolist()
    }

def plot_confusion_matrix_fig(cm, class_names=['Normal', 'Stone']):
    """Generates Matplotlib figure for Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_roc_curve_fig(y_true, y_probs):
    """Generates Matplotlib figure for ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('Receiver Operating Characteristic', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig
