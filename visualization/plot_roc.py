# visualization/plot_roc.py

import os

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def plot_roc_curve(
    y_true,
    y_prob,
    output_path,
    model_name="RF"
):
    """
    Draw ROC curve and save figure.
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()