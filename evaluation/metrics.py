# evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    計算分類指標
    
    Args:
        y_true: 真實標籤 (array-like)
        y_pred: 預測標籤 (array-like)
        y_prob: 預測機率 (array-like), optional, 用於 ROC-AUC / PR-AUC
    
    Returns:
        dict: metrics
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    else:
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan
    
    return metrics

def print_metrics(metrics_dict, prefix=""):
    """
    格式化輸出 metrics
    """
    print(f"[{prefix} Metrics] - metrics.py:46")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f} - metrics.py:48")
