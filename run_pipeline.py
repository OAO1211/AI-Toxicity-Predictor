# run_pipeline.py

import os
import sys
from glob import glob

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
            # Grid search
            # -------------------------


            print(
                "[STEP 3.1] Grid search..."
            )


            best_params, best_score = (
                cfg["grid_search"](X, y)
            )


            print(
                f"[INFO] Best params: {best_params}"
            )


            print(
                f"[INFO] Best CV score: {best_score:.4f}"
            )



            # -------------------------
            # 5 Fold CV + SHAP
            # -------------------------


            print(
                "[STEP 3.2] 5-fold CV + SHAP..."
            )


            run_5fold_cv(

                model_builder=lambda:
                    cfg["builder"](
                        **best_params
                    ),

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



    print(

        "\n[INFO] Pipeline finished successfully!"

    )




if __name__ == "__main__":


    run_pipeline()