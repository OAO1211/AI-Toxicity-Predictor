"""Variant-aware metrics aggregation for DILI v4.

Run from project root:
    python evaluation/aggregate_metrics.py

or import:
    aggregate_model_metrics(results_dir='results', output_path='results/comparison/metrics_summary.csv')

The key fix is that primary and sensitivity_conventional are parsed from the
result-directory name and kept as separate DatasetVariant groups.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_prob, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def _parse_variant(dataset_dir_name: str) -> str:
    if dataset_dir_name.endswith("_sensitivity_conventional"):
        return "sensitivity_conventional"
    if dataset_dir_name.endswith("_primary"):
        return "primary"
    # Keep unknown variants distinct rather than silently mixing them.
    m = re.search(r"_(primary|sensitivity_conventional)$", dataset_dir_name)
    if m:
        return m.group(1)
    return dataset_dir_name


def _parse_prediction_path(file: str, results_dir: str):
    """Return dataset dir, variant, feature set, model, fold from a prediction file."""
    rel = Path(os.path.relpath(file, results_dir))
    parts = rel.parts
    if len(parts) < 5 or parts[-1] != "predictions.csv":
        raise ValueError(f"Unexpected prediction path: {file}")

    dataset_dir_name = parts[-5]
    feature_set_name = parts[-4]
    model_name = parts[-3]
    fold_name = parts[-2]
    variant = _parse_variant(dataset_dir_name)
    return dataset_dir_name, variant, feature_set_name, model_name, fold_name


def aggregate_model_metrics(results_dir="results", output_path="results/comparison/metrics_summary.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prediction_files = sorted(
        glob.glob(os.path.join(results_dir, "**", "fold*", "predictions.csv"), recursive=True)
    )
    if not prediction_files:
        raise ValueError(f"No prediction files found under: {results_dir}")

    records = []
    for file in prediction_files:
        dataset_name, variant, feature_set, model, fold = _parse_prediction_path(file, results_dir)
        df = pd.read_csv(file)
        required = {"y_true", "y_prob", "y_pred"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{file} missing columns: {sorted(missing)}")

        metrics = calculate_metrics(df["y_true"], df["y_prob"], df["y_pred"])
        metrics.update(
            {
                "DatasetName": dataset_name,
                "DatasetVariant": variant,
                "FeatureSet": feature_set,
                "Model": model,
                "Fold": fold,
                "N": len(df),
                "PredictionFile": os.path.relpath(file, results_dir),
            }
        )
        records.append(metrics)

    fold_df = pd.DataFrame(records).sort_values(
        ["DatasetVariant", "FeatureSet", "Model", "Fold"]
    )

    fold_output = output_path.replace(".csv", "_folds.csv")
    fold_df.to_csv(fold_output, index=False)

    metric_cols = [
        "Accuracy", "BalancedAccuracy", "Precision", "Recall", "Specificity",
        "F1", "ROC-AUC", "PR-AUC"
    ]
    summary = (
        fold_df.groupby(["DatasetVariant", "FeatureSet", "Model"], as_index=False)
        .agg({m: ["mean", "std"] for m in metric_cols})
    )
    summary.columns = [
        c if isinstance(c, str) else "_".join(x for x in c if x)
        for c in summary.columns
    ]
    # pandas may represent grouping cols as tuples after multi-agg.
    summary.columns = [
        "_".join(x for x in c if x) if isinstance(c, tuple) else c
        for c in summary.columns
    ]
    # Clean accidental trailing underscores in group columns.
    summary = summary.rename(columns={
        "DatasetVariant_": "DatasetVariant",
        "FeatureSet_": "FeatureSet",
        "Model_": "Model",
    })
    summary.to_csv(output_path, index=False)

    # Convenience per-variant tables.
    stem, ext = os.path.splitext(output_path)
    for variant, sub in summary.groupby("DatasetVariant", sort=False):
        sub.to_csv(f"{stem}_{variant}{ext}", index=False)

    print(f"[DONE] Variant-aware metrics aggregation: {len(fold_df)} folds")
    print(f"[SAVED] {fold_output}")
    print(f"[SAVED] {output_path}")
    print("\nFold counts by variant / feature set / model:")
    print(fold_df.groupby(["DatasetVariant", "FeatureSet", "Model"]).size().to_string())
    return summary, fold_df


if __name__ == "__main__":
    aggregate_model_metrics()
