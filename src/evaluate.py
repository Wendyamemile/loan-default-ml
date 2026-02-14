import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_classification(y_true, y_pred, baseline_accuracy=None):
    """
    Evaluate a classification model and optionally compare to a baseline.
    
    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels
    - baseline_accuracy: float, optional, baseline accuracy to compare
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("=== Classification Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    if baseline_accuracy is not None:
        improvement = accuracy - baseline_accuracy
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"Improvement over baseline: {improvement:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_model(y_true, y_pred, task="classification", baseline_metric=None):
    """
    General evaluation function for classification (and future regression).
    
    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels
    - task: "classification" (default) or "regression"
    - baseline_metric: baseline accuracy for classification or baseline MSE for regression
    """
    if task == "classification":
        evaluate_classification(y_true, y_pred, baseline_accuracy=baseline_metric)
    else:
        raise ValueError("Currently only 'classification' is implemented.")