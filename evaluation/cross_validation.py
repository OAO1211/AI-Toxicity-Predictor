# evaluation/cross_validation.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from explain.shap_analysis import (
    compute_shap,
    mean_abs_shap,
    save_shap_output
)

from visualization.plot_roc import plot_roc_curve
from visualization.plot_confusion import plot_confusion_matrix
from visualization.plot_shap import (
    plot_mean_abs_shap,
    plot_shap_summary
)


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
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y), 1
    ):

        print(
            f"[INFO] {model_name} fold {fold}"
        )
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        train_ids = np.array(ids)[train_idx]
        test_ids = np.array(ids)[test_idx]


        # =====================
        # Train
        # =====================

        model = model_builder()
        model.fit(
            X_train,
            y_train
        )
        pred_prob = model.predict_proba(
            X_test
        )[:,1]
        pred_label = model.predict(
            X_test
        )

        # =====================
        # Fold output dir
        # =====================

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
        # Evaluation plots
        # =====================

        plot_roc_curve(
            y_true=y_test,
            y_prob=pred_prob,
            output_path=os.path.join(
                fold_dir,
                "roc_curve.png"
            ),
            model_name=model_name
        )
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=pred_label,
            output_path=os.path.join(
                fold_dir,
                "confusion_matrix.png"
            )
        )


        # =====================
        # SHAP train
        # =====================

        shap_train, baseline, explainer = compute_shap(
            model,
            X_train,
            model_type=model_type
        )
        shap_test = explainer.shap_values(
            X_test
        )
        if isinstance(shap_test, list):
            shap_test = shap_test[1]


        # =====================
        # Mean SHAP
        # =====================

        fold_mean_shap = mean_abs_shap(
            shap_train
        )
        mean_shap_list.append(
            fold_mean_shap
        )

        # =====================
        # Save SHAP
        # =====================

        save_shap_output(
            ids=train_ids,
            shap_values=shap_train,
            baseline=baseline,
            pred_prob=model.predict_proba(X_train)[:,1],
            feature_names=X.columns,
            out_path=os.path.join(
                fold_dir,
                "training_shap.tsv"
            )
        )

        save_shap_output(
            ids=test_ids,
            shap_values=shap_test,
            baseline=baseline,
            pred_prob=pred_prob,
            feature_names=X.columns,
            out_path=os.path.join(
                fold_dir,
                "testing_shap.tsv"
            )
        )
        # SHAP summary plot
        plot_shap_summary(
            shap_df_path=os.path.join(
                fold_dir,
                "training_shap.tsv"
            ),
            output_dir=fold_dir
        )

    # ==================================================
    # Aggregate SHAP after all folds
    # ==================================================

    mean_shap = np.mean(
        mean_shap_list,
        axis=0
    )
    if len(mean_shap.shape) == 2:
        mean_shap = mean_shap[:,1]
    out_dir = os.path.join(
        output_dir,
        model_name
    )
    os.makedirs(
        out_dir,
        exist_ok=True
    )
    mean_shap_df = pd.DataFrame({
        "Feature": X.columns,
        f"MeanAbsSHAP_{model_name}":
            mean_shap
    })
    mean_shap_path = os.path.join(
        out_dir,
        "mean_abs_shap.tsv"
    )
    mean_shap_df.to_csv(
        mean_shap_path,
        sep="\t",
        index=False
    )

    # Mean SHAP bar plot
    plot_mean_abs_shap(
        mean_shap_path,
        output_dir=out_dir
    )
    print(
        f"[DONE] {model_name} 5fold CV finished"
    )