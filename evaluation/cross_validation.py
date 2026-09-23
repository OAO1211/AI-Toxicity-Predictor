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
    model_type="rf",
    feature_transform_fn=None,
    feature_set_name=None
):
    """
    builder: 呼叫方式為 builder(**best_params) -> sklearn/xgboost estimator
             （也可能是一個 sklearn Pipeline，例如 scaler + 裸模型，
             見 model_selection.grid_search.make_scaled_builder）
    grid_search_fn: 呼叫方式為 grid_search_fn(X_train, y_train) -> (best_params, best_score)

    feature_transform_fn: 選填。呼叫方式為
        feature_transform_fn(X_train, X_test) -> (X_train_out, X_test_out)
        在每個 outer fold 內部、grid search 之前呼叫，用來做「只能用該
        fold 訓練資料 fit」的前處理（目前是 preprocess.scaling.
        make_descriptor_scaler_transform，用於 Combined_Scaled）。
        （之前這裡漏掉了這個參數，但 run_pipeline.py 已經在傳，會直接
        造成 TypeError；現在補上，行為符合原本的設計意圖。）

    feature_set_name: 選填，用來標記輸出的 OOF prediction 屬於哪個
        feature set（ECFP / Descriptors / Combined_Naive / Combined_Scaled）。
        沒有提供的話，退回用 output_dir 的資料夾名稱推斷。
    """

    if feature_set_name is None:
        feature_set_name = os.path.basename(
            os.path.normpath(output_dir)
        )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    mean_shap_list = []

    # Out-of-fold predictions across all 5 folds. Every sample lands in
    # exactly one outer test fold, so concatenating all folds' test
    # predictions gives one true OOF prediction per compound (item 8).
    oof_rows = []

    # Out-of-fold SHAP values (test_shap of every fold, concatenated).
    # Used both by the existing "top SHAP fragments" step and by the new
    # per-error-compound fragment extraction, so that neither is based
    # on fold1 alone (item 7).
    oof_shap_frames = []

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
        # Fold-local feature transform (e.g. descriptor scaling)
        # -------------------------
        #
        # Must run before grid search AND before the final fit below,
        # and must only ever be fit on X_train of this fold — the
        # closure returned by make_descriptor_scaler_transform enforces
        # that by fitting a fresh StandardScaler inside itself every
        # time it's called.

        if feature_transform_fn is not None:
            X_train, X_test = feature_transform_fn(
                X_train,
                X_test
            )

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

        # Re-read what was just written (rather than re-deriving the
        # DataFrame shape from scratch) so the OOF SHAP table is
        # guaranteed to have exactly the same columns/format as the
        # per-fold testing_shap.tsv files.
        fold_test_shap_df = pd.read_csv(testing_shap_path, sep="\t")
        fold_test_shap_df.insert(1, "fold", fold)
        oof_shap_frames.append(fold_test_shap_df)

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

        # -------------------------
        # Accumulate OOF predictions (item 1 / item 8)
        # -------------------------
        #
        # Same information as prediction_df above, but with the column
        # names/shape needed for error_analysis.py, and collected across
        # all 5 folds so every compound ends up with exactly one true
        # out-of-fold prediction.

        oof_rows.append(
            pd.DataFrame({
                "SampleID": test_ids,
                "y_true": y_test.values,
                "pred_prob": pred_test,
                "pred_label": pred_label,
                "fold": fold,
                "model": model_name,
                "feature_set": feature_set_name
            })
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

    # ==================================================
    # Save out-of-fold predictions (item 1 / item 8)
    # ==================================================

    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True)

    n_unique_ids = oof_df["SampleID"].nunique()
    if n_unique_ids != len(oof_df):
        print(
            f"[WARN] {model_name}/{feature_set_name}: OOF predictions "
            f"have {len(oof_df)} rows but only {n_unique_ids} unique "
            f"SampleID — some compound appears in more than one outer "
            f"test fold, which should not happen with StratifiedKFold."
        )

    oof_path = os.path.join(
        model_output_dir,
        "oof_predictions.tsv"
    )

    oof_df.to_csv(
        oof_path,
        sep="\t",
        index=False
    )

    print(
        f"[INFO] {model_name}/{feature_set_name}: saved "
        f"{len(oof_df)} out-of-fold predictions to {oof_path}"
    )

    # ==================================================
    # Save out-of-fold SHAP table (item 7)
    # ==================================================

    oof_shap_df = pd.concat(oof_shap_frames, axis=0, ignore_index=True)

    oof_shap_path = os.path.join(
        model_output_dir,
        "oof_shap.tsv"
    )

    oof_shap_df.to_csv(
        oof_shap_path,
        sep="\t",
        index=False
    )

    print(
        f"[INFO] {model_name}/{feature_set_name}: saved OOF SHAP table "
        f"({len(oof_shap_df)} rows, all 5 folds) to {oof_shap_path}"
    )


    print(
        f"[DONE] {model_name} 5fold CV finished"
    )