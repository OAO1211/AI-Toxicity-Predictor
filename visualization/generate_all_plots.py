# visualization/generate_all_plots.py

import os

from visualization.plot_roc import plot_roc_curve
from visualization.plot_confusion import plot_confusion_matrix
from visualization.plot_shap import (
    plot_mean_abs_shap,
    plot_shap_summary,
)


def generate_all_plots(
    y_true,
    y_prob,
    y_pred,
    shap_path,
    mean_shap_path,
    output_dir,
    model_name,
):
    """
    Generate all visualization outputs for one fold.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ROC
    plot_roc_curve(
        y_true=y_true,
        y_prob=y_prob,
        output_path=os.path.join(output_dir, "roc_curve.png"),
        model_name=model_name,
    )

    # Confusion Matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=os.path.join(
            output_dir,
            "confusion_matrix.png"
        ),
        model_name=model_name
    )

    # SHAP summary
    plot_shap_summary(
        shap_df_path=shap_path,
        output_dir=output_dir,
    )

    # Mean SHAP bar
    if mean_shap_path is not None:
        plot_mean_abs_shap(
            mean_shap_path,
            output_dir,
        )