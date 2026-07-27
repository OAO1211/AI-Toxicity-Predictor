# evaluation/cross_validation.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from explain.shap_analysis import compute_shap, mean_abs_shap, save_shap_output


def run_5fold_cv(
    model_builder,
    ids,
    X,
    y,
    output_dir,
    model_name,
    model_type="rf"
):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean_shap_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"[INFO] {model_name} fold {fold} - cross_validation.py:22")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_ids = np.array(ids)[train_idx]
        test_ids = np.array(ids)[test_idx]

        # =====================
        # Train model
        # =====================
        model = model_builder()
        model.fit(X_train, y_train)

        pred_train = model.predict_proba(X_train)[:, 1]
        pred_test = model.predict_proba(X_test)[:, 1]

        # =====================
        # SHAP (train)
        # =====================
        shap_values_train, baseline, explainer = compute_shap(
            model,
            X_train,
            model_type=model_type
        )

        # =====================
        # SHAP (test)
        # =====================
        shap_values_test = explainer.shap_values(X_test)
        if isinstance(shap_values_test, list):
            shap_values_test = shap_values_test[1]

        # =====================
        # Mean |SHAP|
        # =====================
        mean_abs = mean_abs_shap(shap_values_train)
        mean_shap_list.append(mean_abs)

        # =====================
        # Save
        # =====================
        fold_dir = os.path.join(output_dir, model_name, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        save_shap_output(
            ids=train_ids,
            shap_values=shap_values_train,
            baseline=baseline,
            pred_prob=pred_train,
            feature_names=X.columns,
            out_path=os.path.join(fold_dir, "training_shap.tsv")
        )

        save_shap_output(
            ids=test_ids,
            shap_values=shap_values_test,
            baseline=baseline,
            pred_prob=pred_test,
            feature_names=X.columns,
            out_path=os.path.join(fold_dir, "testing_shap.tsv")
        )

    # =====================
    # Aggregate mean SHAP
    # =====================
    mean_shap = np.mean(mean_shap_list, axis=0)

    # Handle SHAP multiclass output
    # New SHAP RF/XGB format:
    # (features, classes)
    if len(mean_shap.shape) == 2:
        mean_shap = mean_shap[:, 1]

    mean_shap_df = pd.DataFrame({
        "Feature": X.columns,
        f"MeanAbsSHAP_{model_name}": mean_shap
    })

    out_dir = os.path.join(output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    mean_shap_df.to_csv(
        os.path.join(out_dir, "mean_abs_shap.tsv"),
        sep="\t",
        index=False
    )

    print(f"[DONE] {model_name} 5fold CV finished - cross_validation.py:109")
