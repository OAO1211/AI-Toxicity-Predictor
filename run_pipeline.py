# run_pipeline.py

import os
import sys
from glob import glob

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
    MODEL_CONFIGS
)


# =========================
# Feature extraction
# =========================

from features.ecfp import extract_ecfp_features


# =========================
# Data loading
# =========================

from data_loader import load_data


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



def run_pipeline():

    # Create folders

    os.makedirs(
        FEATURE_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
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



        # =========================
        # Step 1
        # Generate ECFP
        # =========================


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



        # =========================
        # Step 2
        # Load features
        # =========================


        ids, X, y, _ = load_data(

            ecfp_path,

            LABEL_COL,

            NAME_COL

        )


        print(

            f"[INFO] Samples: {X.shape[0]}, "
            f"Features: {X.shape[1]}"

        )



        dataset_out = os.path.join(

            RESULTS_DIR,

            dataset_name

        )


        os.makedirs(

            dataset_out,

            exist_ok=True

        )



        # =========================
        # Step 3
        # Train models
        # =========================


        for model_name, cfg in MODEL_CONFIGS.items():


            print(
                f"\n[MODEL] {model_name}"
            )



            # -------------------------
            # 5 Fold CV + nested grid search + SHAP
            # -------------------------
            #
            # Grid search 不再對全部資料 (X, y) 執行一次；
            # 而是把 builder / grid_search 函式交給 run_5fold_cv，
            # 讓每個 outer fold 內部只用該 fold 的訓練資料做超參數搜尋，
            # 避免 test fold 的資訊洩漏進超參數選擇 (nested CV)。
            # -------------------------


            print(
                "[STEP 3] 5-fold nested CV (grid search + SHAP)..."
            )


            run_5fold_cv(

                builder=cfg["builder"],

                grid_search_fn=cfg["grid_search"],

                ids=ids,

                X=X,

                y=y,

                output_dir=dataset_out,

                model_name=model_name,

                model_type=cfg["model_type"]

            )
            


            # -------------------------
            # Fragment extraction
            # -------------------------


            print(
                "[STEP 3.3] Extract SHAP fragments..."
            )



            shap_path = os.path.join(

                dataset_out,

                model_name,

                "fold1",

                "training_shap.tsv"

            )



            fragment_dir = os.path.join(

                dataset_out,

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
        # =========================
        # Aggregate all metrics
        # =========================

        print("\n[STEP 4] Aggregate metrics... - run_pipeline.py:371")

        aggregate_model_metrics(
            results_dir=RESULTS_DIR,
            output_path=os.path.join(
                RESULTS_DIR,
                "comparison",
                "metrics_summary.csv"
            )
        )


    print(

        "\n[INFO] Pipeline finished successfully!"

    )




if __name__ == "__main__":


    run_pipeline()