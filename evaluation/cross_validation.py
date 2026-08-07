#evaluation/cross_validation.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from explain.shap_analysis import (
    compute_shap,
    mean_abs_shap,
    save_shap_output
)

from visualization.generate_all_plots import generate_all_plots
from visualization.plot_shap import plot_mean_abs_shap


def run_5fold_cv(
    model_builder,
    ids,
    X,
    y,
    output_dir,
    model_name,
    model_type="rf"
):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    mean_shap_list = []


    # ==================================================
    # 5 Fold Cross Validation
    # ==================================================

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        1
    ):

        print(
            f"[INFO] {model_name} fold {fold}"
        )


        # -------------------------
        # Split data
        # -------------------------

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_ids = np.array(ids)[train_idx]
        test_ids = np.array(ids)[test_idx]


        # -------------------------
        # Train model
        # -------------------------

        model = model_builder()

        model.fit(
            X_train,
            y_train
        )


        pred_train = model.predict_proba(
            X_train
        )[:, 1]


        pred_test = model.predict_proba(
            X_test
        )[:, 1]


        pred_label = model.predict(X_test).astype(int)


        # -------------------------
        # Fold output directory
        # -------------------------

        fold_dir = os.path.join(
            output_dir,
            model_name,
            f"fold{fold}"
        )

        os.makedirs(
            fold_dir,
            exist_ok=True
        )
        # =====================
        # Save predictions FIRST
        # =====================

        prediction_df = pd.DataFrame({
            "SampleID": test_ids,
            "y_true": y_test.values,
            "y_prob": pred_test,
            "y_pred": pred_label
        })


        prediction_df.to_csv(
            os.path.join(
                fold_dir,
                "predictions.csv"
            ),
            index=False
        )

        # ==================================================
        # SHAP calculation
        # ==================================================

        shap_values_train, baseline, explainer = compute_shap(
            model,
            X_train,
            model_type=model_type
        )


        shap_values_test = explainer.shap_values(
            X_test
        )


        # SHAP old version compatibility
        if isinstance(shap_values_test, list):
            shap_values_test = shap_values_test[1]

        elif len(shap_values_test.shape) == 3:
            shap_values_test = shap_values_test[:, :, 1]


        # -------------------------
        # Mean SHAP storage
        # -------------------------

        fold_mean_shap = mean_abs_shap(
            shap_values_train
        )

        mean_shap_list.append(
            fold_mean_shap
        )


        # ==================================================
        # Save SHAP tables
        # ==================================================

        training_shap_path = os.path.join(
            fold_dir,
            "training_shap.tsv"
        )


        testing_shap_path = os.path.join(
            fold_dir,
            "testing_shap.tsv"
        )


        save_shap_output(
            ids=train_ids,
            shap_values=shap_values_train,
            baseline=baseline,
            pred_prob=pred_train,
            feature_names=X.columns,
            out_path=training_shap_path
        )


        save_shap_output(
            ids=test_ids,
            shap_values=shap_values_test,
            baseline=baseline,
            pred_prob=pred_test,
            feature_names=X.columns,
            out_path=testing_shap_path
        )

        # =====================
        # Save predictions
        # =====================

        prediction_df = pd.DataFrame({
            "Fold": fold,
            "SampleID": test_ids,
            "y_true": y_test.values,
            "y_prob": pred_test,
            "y_pred": pred_label
        })


        prediction_df.to_csv(
            os.path.join(
                fold_dir,
                "predictions.csv"
            ),
            index=False
        )
        # ==================================================
        # Generate fold visualization
        # ==================================================

        generate_all_plots(
            y_true=y_test,
            y_prob=pred_test,
            y_pred=pred_label,
            shap_path=training_shap_path,
            mean_shap_path=None,
            output_dir=fold_dir,
            model_name=model_name
        )


    # ==================================================
    # Aggregate SHAP after 5 folds
    # ==================================================

    mean_shap = np.mean(
        mean_shap_list,
        axis=0
    )


    # SHAP multiclass format
    if len(mean_shap.shape) == 2:
        mean_shap = mean_shap[:, 1]


    model_output_dir = os.path.join(
        output_dir,
        model_name
    )


    os.makedirs(
        model_output_dir,
        exist_ok=True
    )


    mean_shap_df = pd.DataFrame(
        {
            "Feature": X.columns,
            f"MeanAbsSHAP_{model_name}": mean_shap
        }
    )


    mean_shap_path = os.path.join(
        model_output_dir,
        "mean_abs_shap.tsv"
    )


    mean_shap_df.to_csv(
        mean_shap_path,
        sep="\t",
        index=False
    )


    # ==================================================
    # Overall SHAP importance plot
    # ==================================================

    plot_mean_abs_shap(
        mean_shap_path=mean_shap_path,
        output_dir=model_output_dir
    )


    print(
        f"[DONE] {model_name} 5fold CV finished"
    )