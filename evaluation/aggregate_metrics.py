# evaluation/aggregate_metrics.py

import os
import glob
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(y_true, y_prob, y_pred):

    return {
        "Accuracy":
            accuracy_score(y_true, y_pred),

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
            "*",
            "*",
            "fold*",
            "predictions.csv"
        )
    )


    for file in prediction_files:

        path_parts = file.split(os.sep)

        model_name = path_parts[-3]

        fold_name = path_parts[-2]


        df = pd.read_csv(file)


        metrics = calculate_metrics(
            df["y_true"],
            df["y_prob"],
            df["y_pred"]
        )


        metrics["Model"] = model_name
        metrics["Fold"] = fold_name


        records.append(metrics)



    result = pd.DataFrame(records)


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
        .groupby("Model")
        .agg(
            {
                "Accuracy":["mean","std"],
                "Precision":["mean","std"],
                "Recall":["mean","std"],
                "F1":["mean","std"],
                "ROC-AUC":["mean","std"]
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