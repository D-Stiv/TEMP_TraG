# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.

import numpy as np
import sklearn


def compute_binary_metrics(preds: np.array, labels: np.array):
    """
    Computes metrics based on raw/normalized model predictions.
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Dictionary containing accuracy, precision, recall, F1-score (per class and macro avg), and ROC AUC scores.
    """
    if len(preds.shape) > 1:
        probs = preds[:, 1]  # Probability for the positive class
        preds = preds.argmax(axis=-1)      
    else:
        probs = preds
    # Compute precision-recall curve & AUC
    precisions, recalls, _ = sklearn.metrics.precision_recall_curve(labels, probs)
    auc = sklearn.metrics.auc(recalls, precisions)
    
    # Compute per-class precision, recall, and F1-score
    precision_per_class = sklearn.metrics.precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = sklearn.metrics.recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class = sklearn.metrics.f1_score(labels, preds, average=None, zero_division=0)
    
    # Compute macro-average scores
    precision_macro = sklearn.metrics.precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = sklearn.metrics.recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = sklearn.metrics.f1_score(labels, preds, average='macro', zero_division=0)
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, preds)
    
    # Compute accuracy
    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    
    return {
        "accuracy": accuracy,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "pr_auc": auc,
        'confusion_matrix': confusion_matrix
    }
