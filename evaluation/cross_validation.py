#evaluation/cross_validation.py
import os
import json
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
    builder,
    grid_search_fn,
    ids,
    X,
    y,
    output_dir,
    model_name,
    model_type="rf"
):
    """
    builder: 呼叫方式為 builder(**best_params) -> sklearn/xgboost estimator
             （也可能是一個 sklearn Pipeline，例如 scaler + 裸模型，
             見 model_selection.grid_search.make_scaled_builder）
    grid_search_fn: 呼叫方式為 grid_search_fn(X_train, y_train) -> (best_params, best_score)

    重要：grid_search_fn 在每個 outer fold 內部才會被呼叫，
    且只會看到該 fold 的訓練資料 (X_train, y_train)。
    這是為了避免「先在全部資料上做 grid search，再用同一份資料做 CV 評估」
    所造成的資訊洩漏（outer test fold 間接影響了超參數選擇），
    也就是正確做法的 nested cross-validation。

    如果這個 (feature set, model) 組合需要對 descriptors 做 scaling
    （目前只有 Descriptors/Combined_Scaled 的 LogReg），scaling 本身
    是包在 builder/grid_search_fn 內部的 sklearn Pipeline 裡處理，
    讓 GridSearchCV 在每一個 inner fold 都重新 fit scaler，不會有
    inner validation fold 的分布資訊外洩——比起「在 outer fold 開始
    時就先 fit 好 scaler、整個 inner CV 共用同一份 scaler」更嚴格。
    """

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


        # -------------------------
        # Grid search (inner CV, 只用本 fold 的訓練資料)
        # -------------------------

        print(
            f"[INFO] {model_name} fold {fold} grid search..."
        )

        best_params, best_cv_score = grid_search_fn(
            X_train,
            y_train
        )

        print(
            f"[INFO] {model_name} fold {fold} best params: {best_params}"
        )

        with open(
            os.path.join(fold_dir, "best_params.json"),
            "w"
        ) as f:
            json.dump(
                {
                    "best_params": best_params,
                    "inner_cv_best_score": best_cv_score
                },
                f,
                indent=2
            )


        # -------------------------
        # Train model
        # -------------------------

        model = builder(**best_params)

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


        # ==================================================
        # SHAP calculation
        # ==================================================
        #
        # 如果 model 是 Pipeline(scaler + clf)（目前只有做了 scaling 的
        # LogReg 會這樣），shap.TreeExplainer / LinearExplainer 都需要
        # 裸模型，不能直接吃 Pipeline，所以這裡先把 scaler 和 clf 拆開：
        # clf 用來建 explainer，X 則先手動過 scaler.transform() 再餵進去，
        # 這樣 SHAP 看到的就是模型實際「看到」的（已經 scale 過的）特徵空間。

        if hasattr(model, "named_steps") and "scaler" in model.named_steps:
            clf_for_shap = model.named_steps["clf"]
            X_train_for_shap = model.named_steps["scaler"].transform(X_train)
            X_test_for_shap = model.named_steps["scaler"].transform(X_test)
        else:
            clf_for_shap = model
            X_train_for_shap = X_train
            X_test_for_shap = X_test

        shap_values_train, baseline, explainer = compute_shap(
            clf_for_shap,
            X_train_for_shap,
            model_type=model_type
        )


        shap_values_test = explainer.shap_values(
            X_test_for_shap
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