import os
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


def evaluate_model_cv(
    model_dir
):

    fold_results = []

    all_true = []
    all_prob = []
    all_pred = []


    for fold in range(1,6):

        pred_path = os.path.join(
            model_dir,
            f"fold{fold}",
            "predictions.csv"
        )


        df = pd.read_csv(pred_path)


        y_true = df["y_true"]
        y_prob = df["y_prob"]
        y_pred = df["y_pred"]


        metrics = {

            "fold": fold,

            "ROC-AUC":
                roc_auc_score(
                    y_true,
                    y_prob
                ),

            "Accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_true,
                    y_pred
                ),

            "Recall":
                recall_score(
                    y_true,
                    y_pred
                ),

            "F1":
                f1_score(
                    y_true,
                    y_pred
                ),

            "MCC":
                matthews_corrcoef(
                    y_true,
                    y_pred
                )
        }


        fold_results.append(metrics)


        all_true.extend(y_true)
        all_prob.extend(y_prob)
        all_pred.extend(y_pred)


    fold_df = pd.DataFrame(
        fold_results
    )


    summary = {

        "ROC-AUC":
            f"{fold_df['ROC-AUC'].mean():.3f} ± {fold_df['ROC-AUC'].std():.3f}",

        "Accuracy":
            f"{fold_df['Accuracy'].mean():.3f} ± {fold_df['Accuracy'].std():.3f}",

        "F1":
            f"{fold_df['F1'].mean():.3f} ± {fold_df['F1'].std():.3f}",

        "MCC":
            f"{fold_df['MCC'].mean():.3f} ± {fold_df['MCC'].std():.3f}"
    }


    return (
        fold_df,
        pd.DataFrame([summary]),
        all_true,
        all_prob,
        all_pred
    )