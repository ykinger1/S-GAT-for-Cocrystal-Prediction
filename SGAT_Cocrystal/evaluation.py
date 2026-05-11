# ===================== 1. Import Dependencies =====================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


# ===================== 2. Core Evaluation Function =====================
def binary_classification_evaluation(y_true, y_pred, y_score, plot_roc=True, save_roc_path=None):
    """
    Evaluation function for binary classification tasks, returning key metrics, AUC score,
    and optionally plotting/saving ROC curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 and 1)
    y_pred : array-like of shape (n_samples,)
        Predicted binary class labels (0 or 1) from the model
    y_score : array-like of shape (n_samples,)
        Predicted probability scores for the positive class (class 1)
    plot_roc : bool, optional (default=True)
        Whether to plot the ROC curve
    save_roc_path : str, optional (default=None)
        Path to save the ROC curve image (e.g., './roc_curve.png').
        If None, the image will not be saved.

    Returns
    -------
    metrics_dict : dict
        Dictionary containing evaluation metrics:
        - Accuracy: Overall classification accuracy
        - BACC: Balanced accuracy (handles class imbalance)
        - Precision: Positive predictive value
        - Recall: True positive rate (sensitivity)
        - Specificity: True negative rate
        - F1 score: Harmonic mean of precision and recall
    auc_score : float
        Area Under the ROC Curve (AUC) score

    Notes
    -----
    - Handles zero-division cases for precision/recall/F1 score by setting to 0
    - ROC curve is plotted with a random guess baseline (dashed line)
    - All metrics are computed for binary classification (labels 0/1 only)
    """
    # 1. Compute confusion matrix (tn, fp, fn, tp)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # 2. Calculate core evaluation metrics
    metrics_dict = {
        'Accuracy': accuracy_score(y_true, y_pred),  # Overall accuracy
        'BACC': balanced_accuracy_score(y_true, y_pred),  # Balanced accuracy
        'Precision': precision_score(y_true, y_pred, zero_division=0),  # Precision (PPV)
        'Recall': recall_score(y_true, y_pred, zero_division=0),  # Recall (TPR/sensitivity)
        'Specificity': tn / (tn + fp) if (tn + fp) != 0 else 0,  # Specificity (TNR)
        'F1 score': f1_score(y_true, y_pred, zero_division=0)  # F1 harmonic mean
    }

    # 3. Calculate AUC and ROC curve data
    fpr, tpr, _ = roc_curve(y_true, y_score)  # False Positive Rate, True Positive Rate
    auc_score = roc_auc_score(y_true, y_score)  # AUC score

    # 4. Plot ROC curve if enabled
    if plot_roc:
        plt.figure(figsize=(8, 6))
        # Plot ROC curve with AUC annotation
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        # Plot random guess baseline (diagonal line)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # Set axis ranges and labels
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        # Save ROC curve if path is provided
        if save_roc_path:
            plt.savefig(save_roc_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to release memory
        else:
            plt.show()

    return metrics_dict, auc_score


# ===================== 3. Test Script (Main Function) =====================
if __name__ == '__main__':
    # 1. Generate synthetic data for binary classification evaluation
    np.random.seed(42)  # Fix random seed for reproducibility
    n_samples = 1000  # Number of samples
    y_true = np.random.randint(0, 2, size=n_samples)  # Ground truth labels (0/1)

    # Generate noisy predicted labels (simulate model predictions)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y_pred[noise_idx] = 1 - y_pred[noise_idx]

    # Generate predicted probabilities (positive class has higher probabilities)
    y_score = np.random.rand(n_samples)
    y_score[y_true == 1] = y_score[y_true == 1] * 0.7 + 0.2  # Boost positive class probabilities

    # 2. Run evaluation function
    metrics, auc = binary_classification_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        plot_roc=True,
        save_roc_path=None  # Set to './roc_curve.png' to save the plot
    )

    # 3. Print evaluation results
    print("=" * 50)
    print("Binary Classification Evaluation Metrics:")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<12}: {metric_value:.4f}")
    print(f"AUC score    : {auc:.4f}")
    print("=" * 50)