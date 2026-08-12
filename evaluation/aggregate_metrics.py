# evaluation/aggregate_metrics.py

import os
import glob
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)


def calculate_metrics(y_true, y_prob, y_pred):

    return {
        "Accuracy":
            accuracy_score(y_true, y_pred),

        "BalancedAccuracy":
            balanced_accuracy_score(y_true, y_pred),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_true,
                y_prob
            ),

        # 資料集 label 分佈約 267:183（非完全平衡），
        # PR-AUC 比 ROC-AUC 更能反映在少數類別（DILI 陽性）上的表現，
        # 兩者一起看比較不會被 ROC-AUC 的樂觀假象誤導。
        "PR-AUC":
            average_precision_score(
                y_true,
                y_prob
            )
    }



def aggregate_model_metrics(
    results_dir,
    output_path
):


    # create output directory
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    records = []

    prediction_files = glob.glob(
        os.path.join(
            results_dir,
            "**",
            "predictions.csv"
        ),
        recursive=True
    )

    print("\n[DEBUG] prediction files - aggregate_metrics.py:89")

    for f in prediction_files:
        print(f)

    print(
        f"Found {len(prediction_files)} files"
    )


    for file in prediction_files:

        # 目錄結構：results_dir/dataset_name/FeatureSet/model_name/foldN/predictions.csv
        fold_dir = os.path.dirname(file)
        model_dir = os.path.dirname(fold_dir)
        feature_set_dir = os.path.dirname(model_dir)

        model_name = os.path.basename(model_dir)
        feature_set_name = os.path.basename(feature_set_dir)
        fold_name = os.path.basename(fold_dir)


        df = pd.read_csv(file)


        metrics = calculate_metrics(
            df["y_true"],
            df["y_prob"],
            df["y_pred"]
        )


        metrics["FeatureSet"] = feature_set_name
        metrics["Model"] = model_name
        metrics["Fold"] = fold_name


        records.append(metrics)



    result = pd.DataFrame(records)
    print(result)

    if result.empty:
        raise ValueError(
            "No prediction files found. Check results directory."
        )

    # ==========================
    # fold result
    # ==========================
    fold_output = output_path.replace(
    ".csv",
    "_folds.csv"
)

    result.to_csv(
        fold_output,
        index=False
    )


    # ==========================
    # mean ± std
    # ==========================

    summary = (
        result
        .groupby(["FeatureSet", "Model"])
        .agg(
            {
                "Accuracy":["mean","std"],
                "BalancedAccuracy":["mean","std"],
                "Precision":["mean","std"],
                "Recall":["mean","std"],
                "F1":["mean","std"],
                "ROC-AUC":["mean","std"],
                "PR-AUC":["mean","std"]
            }
        )
    )
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        output_path
    )


    print(
        "[DONE] Metrics aggregation finished"
    )


if __name__ == "__main__":


    aggregate_model_metrics(
        results_dir="results",
        output_path=
        "results/comparison/metrics_summary.csv"
    )