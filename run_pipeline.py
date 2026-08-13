# run_pipeline.py

import os
import sys
import argparse
import functools
from glob import glob

import pandas as pd

from evaluation.aggregate_metrics import aggregate_model_metrics

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(BASE_DIR)


# =========================
# Config
# =========================

from config import (
    RAW_DATA_DIR,
    FEATURE_DIR,
    RESULTS_DIR,
    LABEL_COL,
    NAME_COL,
    SMILES_COL,
    ECFP_RADIUS,
    ECFP_BITS,
    TOP_SHAP_BITS,
    MODEL_CONFIGS,
    FEATURE_SET_ECFP,
    FEATURE_SET_DESCRIPTORS,
    FEATURE_SET_COMBINED_NAIVE,
    FEATURE_SET_COMBINED_SCALED
)


# =========================
# Feature extraction
# =========================

from features.ecfp import extract_ecfp_features
from features.descriptors import extract_descriptor_features


# =========================
# Data loading
# =========================

from data_loader import load_data


# =========================
# Feature scaling (for Combined_Scaled)
# =========================

from preprocess.scaling import make_descriptor_scaler_transform


# =========================
# Evaluation
# =========================

from evaluation.cross_validation import run_5fold_cv


# =========================
# SHAP fragment extraction
# =========================

from features.bit_mapping import (
    extract_top_shap_fragments
)


# =========================
# Fragment visualization
# =========================

from visualization.draw_fragments import (
    draw_fragments_from_shap
)


# =========================
# Quick-mode results 另外放一個資料夾，
# 避免流程測試的結果不小心跟正式結果混在一起、污染 metrics_summary.csv
# =========================

RESULTS_DIR_QUICK = RESULTS_DIR + "_quicktest"

ALL_FEATURE_SETS = [
    FEATURE_SET_ECFP,
    FEATURE_SET_DESCRIPTORS,
    FEATURE_SET_COMBINED_NAIVE,
    FEATURE_SET_COMBINED_SCALED
]


def _run_models_for_feature_set(
    ids,
    X,
    y,
    output_dir,
    data_path,
    feature_set_name,
    do_fragment_extraction,
    model_names,
    quick_mode,
    feature_transform_fn=None
):
    """
    對單一特徵集合跑過指定的模型：nested CV + SHAP，
    並視情況（只有 ECFP）額外做 SHAP bit -> 化學片段 還原。

    output_dir 應該已經是「這個特徵集合專屬」的資料夾，例如：
    results/<dataset_name>/ECFP、.../Descriptors、
    .../Combined_Naive、.../Combined_Scaled
    這樣底下同名的模型資料夾（RF / XGB / LogReg）才不會互相覆蓋，
    evaluation/aggregate_metrics.py 也才能正確區分四者。

    model_names: 只跑這個清單裡的模型名稱（例如 quick smoke test 時只跑 ["LogReg"]）
    quick_mode: 傳給 grid_search_fn，True 時只用單一參數組合（僅供流程測試）
    feature_transform_fn: 選填，傳給 run_5fold_cv 做 fold 內部的前處理
                           （目前只有 Combined_Scaled 會用到，對 descriptors 做
                           StandardScaler，ECFP bits 維持 0/1）
    """

    os.makedirs(output_dir, exist_ok=True)

    for model_name in model_names:

        if model_name not in MODEL_CONFIGS:
            print(f"[WARN] Unknown model '{model_name}', skipped. - run_pipeline.py:138")
            continue

        cfg = MODEL_CONFIGS[model_name]

        print(
            f"\n[MODEL] {feature_set_name} / {model_name}"
            + (" [QUICK MODE]" if quick_mode else "")
        )

        # -------------------------
        # 5 Fold CV + nested grid search + SHAP
        # -------------------------
        #
        # Grid search 不再對全部資料 (X, y) 執行一次；
        # 而是把 builder / grid_search 函式交給 run_5fold_cv，
        # 讓每個 outer fold 內部只用該 fold 的訓練資料做超參數搜尋，
        # 避免 test fold 的資訊洩漏進超參數選擇 (nested CV)。
        #
        # quick_mode 用 functools.partial 綁進 grid_search_fn，
        # 讓 run_5fold_cv 仍然只需要呼叫 grid_search_fn(X_train, y_train)。
        # -------------------------

        print(
            "[STEP] 5-fold nested CV (grid search + SHAP)..."
        )

        grid_search_fn = functools.partial(
            cfg["grid_search"],
            quick_mode=quick_mode
        )

        run_5fold_cv(
            builder=cfg["builder"],
            grid_search_fn=grid_search_fn,
            ids=ids,
            X=X,
            y=y,
            output_dir=output_dir,
            model_name=model_name,
            model_type=cfg["model_type"],
            feature_transform_fn=feature_transform_fn
        )

        if not do_fragment_extraction:
            # Descriptors / Combined 系列沒有做 bit -> fragment 還原：
            # Descriptors 是連續數值，本來就沒有 bit 對應關係；
            # Combined 雖然含有 ECFP bits，但目前先聚焦在整體效能比較，
            # fragment 解釋留給 ECFP-only 那組結果。
            continue

        # -------------------------
        # Fragment extraction (僅 ECFP)
        # -------------------------

        print(
            "[STEP] Extract SHAP fragments..."
        )

        shap_path = os.path.join(
            output_dir,
            model_name,
            "fold1",
            "training_shap.tsv"
        )

        fragment_dir = os.path.join(
            output_dir,
            "fragments",
            model_name
        )

        os.makedirs(
            fragment_dir,
            exist_ok=True
        )

        extract_top_shap_fragments(
            shap_df_path=shap_path,
            dataset_df_path=data_path,
            top_n=TOP_SHAP_BITS,
            output_dir=fragment_dir,
            radius=ECFP_RADIUS,
            n_bits=ECFP_BITS
        )

        # -------------------------
        # Fragment PNG
        # -------------------------

        fragment_tsv = os.path.join(
            fragment_dir,
            "top_shap_fragments.tsv"
        )

        png_dir = os.path.join(
            fragment_dir,
            "pngs"
        )

        if os.path.exists(fragment_tsv):

            draw_fragments_from_shap(
                fragments_df_path=fragment_tsv,
                output_dir=png_dir
            )


def run_pipeline(
    quick_mode=False,
    feature_sets=None,
    model_names=None
):
    """
    quick_mode: True 時，grid search 只用單一參數組合，且結果會寫到
                results_quicktest/（不是 results/），避免污染正式結果。
                僅用於確認整條 pipeline 能不能跑完，不可用於正式報告。
    feature_sets: 要跑的特徵集合，預設全部四種：
                  ECFP / Descriptors / Combined_Naive / Combined_Scaled
                  可以只跑其中幾種做 smoke test。
    model_names: 要跑的模型，預設 MODEL_CONFIGS 的全部 key（RF/XGB/LogReg）
                 可以只跑 ["LogReg"] 做 smoke test。
    """

    if feature_sets is None:
        feature_sets = list(ALL_FEATURE_SETS)

    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())

    results_dir = RESULTS_DIR_QUICK if quick_mode else RESULTS_DIR

    if quick_mode:
        print(
            "\n" + "=" * 50
        )
        print(
            "[QUICK MODE] 這次執行只用單一參數組合（沒有真正做"
            "超參數搜尋），結果會寫到:"
        )
        print(
            f"  {results_dir}"
        )
        print(
            "這個模式只用來確認 pipeline 能不能跑完，"
            "結果不可以拿來當作正式報告數字！"
        )
        print(
            "=" * 50 + "\n"
        )

    # Create folders

    os.makedirs(
        FEATURE_DIR,
        exist_ok=True
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    # =========================
    # Find datasets
    # =========================

    dataset_files = glob(
        os.path.join(
            RAW_DATA_DIR,
            "*.csv"
        )
    )

    if not dataset_files:

        raise ValueError(
            "No dataset found in data/raw/"
        )

    need_combined = (
        FEATURE_SET_COMBINED_NAIVE in feature_sets
        or FEATURE_SET_COMBINED_SCALED in feature_sets
    )

    # =========================
    # Process datasets
    # =========================

    for data_path in dataset_files:

        dataset_name = os.path.splitext(
            os.path.basename(data_path)
        )[0]

        print(
            f"\n=============================="
        )

        print(
            f"[INFO] Dataset: {dataset_name}"
        )

        print(
            f"=============================="
        )

        dataset_out = os.path.join(
            results_dir,
            dataset_name
        )

        os.makedirs(
            dataset_out,
            exist_ok=True
        )

        ecfp_ids, ecfp_X, ecfp_y = None, None, None
        desc_ids, desc_X, desc_y = None, None, None

        # =========================
        # Feature set 1: ECFP
        # =========================
        #
        # Combined 系列需要用到 ECFP 的特徵矩陣，所以只要
        # ECFP / Combined_Naive / Combined_Scaled 任一個有被要求，就要產生。

        if (
            FEATURE_SET_ECFP in feature_sets
            or need_combined
        ):

            ecfp_path = os.path.join(
                FEATURE_DIR,
                f"{dataset_name}_ecfp{ECFP_BITS}.tsv"
            )

            print(
                "[STEP 1] Generating ECFP features..."
            )

            extract_ecfp_features(
                input_path=data_path,
                output_path=ecfp_path,
                smiles_col=SMILES_COL,
                label_col=LABEL_COL,
                name_col=NAME_COL,
                radius=ECFP_RADIUS,
                n_bits=ECFP_BITS
            )

            ecfp_ids, ecfp_X, ecfp_y, _ = load_data(
                ecfp_path,
                LABEL_COL,
                NAME_COL
            )

            print(
                f"[INFO] ECFP - Samples: {ecfp_X.shape[0]}, "
                f"Features: {ecfp_X.shape[1]}"
            )

        if FEATURE_SET_ECFP in feature_sets:

            _run_models_for_feature_set(
                ids=ecfp_ids,
                X=ecfp_X,
                y=ecfp_y,
                output_dir=os.path.join(dataset_out, FEATURE_SET_ECFP),
                data_path=data_path,
                feature_set_name=FEATURE_SET_ECFP,
                do_fragment_extraction=True,
                model_names=model_names,
                quick_mode=quick_mode
            )

        # =========================
        # Feature set 2: Descriptors (baseline)
        # =========================

        if (
            FEATURE_SET_DESCRIPTORS in feature_sets
            or need_combined
        ):

            descriptors_path = os.path.join(
                FEATURE_DIR,
                f"{dataset_name}_descriptors.tsv"
            )

            print(
                "\n[STEP 2] Generating physicochemical descriptor features..."
            )

            extract_descriptor_features(
                input_path=data_path,
                output_path=descriptors_path,
                smiles_col=SMILES_COL,
                label_col=LABEL_COL,
                name_col=NAME_COL
            )

            desc_ids, desc_X, desc_y, _ = load_data(
                descriptors_path,
                LABEL_COL,
                NAME_COL
            )

            print(
                f"[INFO] Descriptors - Samples: {desc_X.shape[0]}, "
                f"Features: {desc_X.shape[1]}"
            )

        if FEATURE_SET_DESCRIPTORS in feature_sets:

            _run_models_for_feature_set(
                ids=desc_ids,
                X=desc_X,
                y=desc_y,
                output_dir=os.path.join(dataset_out, FEATURE_SET_DESCRIPTORS),
                data_path=data_path,
                feature_set_name=FEATURE_SET_DESCRIPTORS,
                do_fragment_extraction=False,
                model_names=model_names,
                quick_mode=quick_mode
            )

        # =========================
        # Feature set 3 & 4: Combined (ECFP + Descriptors)
        # =========================

        if need_combined:

            print(
                "\n[STEP 3] Building combined ECFP + Descriptors "
                "feature matrix..."
            )

            # 兩份特徵檔案是分別對同一份原始資料跑 RDKit 產生的，
            # 理論上 row 順序/數量會一致，但這裡明確檢查一次，
            # 對不上就直接中止，避免默默地把不同分子的 ECFP 和
            # descriptors 錯誤配對在一起。
            if (
                len(ecfp_ids) != len(desc_ids)
                or list(ecfp_ids) != list(desc_ids)
            ):
                raise ValueError(
                    "ECFP 和 Descriptors 的樣本 (ids) 對不起來，"
                    "無法直接橫向合併成 Combined 特徵。可能是某些 SMILES "
                    "在其中一邊解析失敗、但在另一邊成功，導致兩邊筆數或"
                    "順序不同。請檢查 features/ecfp.py 與 "
                    "features/descriptors.py 印出的 [WARNING] 訊息。"
                )

            combined_ids = ecfp_ids

            combined_X = pd.concat(
                [
                    ecfp_X.reset_index(drop=True),
                    desc_X.reset_index(drop=True)
                ],
                axis=1
            )

            combined_y = ecfp_y.reset_index(drop=True)

            print(
                f"[INFO] Combined - Samples: {combined_X.shape[0]}, "
                f"Features: {combined_X.shape[1]} "
                f"(ECFP {ecfp_X.shape[1]} + Descriptors {desc_X.shape[1]})"
            )

            # -------------------------
            # 3a. Naive Combined：直接串接，不做任何 scaling
            # -------------------------

            if FEATURE_SET_COMBINED_NAIVE in feature_sets:

                _run_models_for_feature_set(
                    ids=combined_ids,
                    X=combined_X,
                    y=combined_y,
                    output_dir=os.path.join(
                        dataset_out,
                        FEATURE_SET_COMBINED_NAIVE
                    ),
                    data_path=data_path,
                    feature_set_name=FEATURE_SET_COMBINED_NAIVE,
                    do_fragment_extraction=False,
                    model_names=model_names,
                    quick_mode=quick_mode,
                    feature_transform_fn=None
                )

            # -------------------------
            # 3b. Controlled Combined：只對 descriptors 做 StandardScaler
            # （在每個 outer fold 內部才 fit，避免資訊洩漏）
            # -------------------------

            if FEATURE_SET_COMBINED_SCALED in feature_sets:

                descriptor_scaler_fn = make_descriptor_scaler_transform(
                    combined_X.columns
                )

                _run_models_for_feature_set(
                    ids=combined_ids,
                    X=combined_X,
                    y=combined_y,
                    output_dir=os.path.join(
                        dataset_out,
                        FEATURE_SET_COMBINED_SCALED
                    ),
                    data_path=data_path,
                    feature_set_name=FEATURE_SET_COMBINED_SCALED,
                    do_fragment_extraction=False,
                    model_names=model_names,
                    quick_mode=quick_mode,
                    feature_transform_fn=descriptor_scaler_fn
                )

        # =========================
        # Aggregate all metrics
        # =========================

        print("\n[STEP 4] Aggregate metrics... - run_pipeline.py:563")

        aggregate_model_metrics(
            results_dir=results_dir,
            output_path=os.path.join(
                results_dir,
                "comparison",
                "metrics_summary.csv"
            )
        )

    print(
        "\n[INFO] Pipeline finished successfully!"
    )

    if quick_mode:
        print(
            f"[QUICK MODE] 結果在 {results_dir}/，"
            f"確認流程沒問題後，請用 `python run_pipeline.py`"
            f"（不加 --quick）跑正式結果。"
        )


def _parse_args():

    parser = argparse.ArgumentParser(
        description="AI-Toxicity-Predictor pipeline"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick mode：grid search 只用單一參數組合，"
            "結果寫到 results_quicktest/，僅供流程測試，不可用於正式報告。"
        )
    )

    parser.add_argument(
        "--feature-sets",
        type=str,
        default=None,
        help=(
            "要跑的特徵集合，逗號分隔，可選值："
            "ECFP, Descriptors, Combined_Naive, Combined_Scaled "
            "（預設全部都跑）"
        )
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "要跑的模型，逗號分隔，例如 --models LogReg "
            "或 --models RF,XGB,LogReg（預設全部都跑）"
        )
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    feature_sets = (
        [s.strip() for s in args.feature_sets.split(",")]
        if args.feature_sets else None
    )

    model_names = (
        [m.strip() for m in args.models.split(",")]
        if args.models else None
    )

    run_pipeline(
        quick_mode=args.quick,
        feature_sets=feature_sets,
        model_names=model_names
    )
