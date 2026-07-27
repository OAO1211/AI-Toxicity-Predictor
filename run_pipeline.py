# run_pipeline.py
import sys
import os
from glob import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ===== Feature & data =====
from features.ecfp import extract_ecfp_features
from data_loader import load_data

# ===== Models =====
from models.train_rf import build_rf
from models.train_xgb import build_xgb
from models.train_logreg import build_logreg

# ===== Model selection =====
from model_selection.grid_search import rf_grid_search, xgb_grid_search, logreg_grid_search

# ===== Evaluation & explain =====
from evaluation.cross_validation import run_5fold_cv
from features.bit_mapping import extract_top_shap_fragments


def run_pipeline(ecfp_radius=3, ecfp_bits=1024, top_shap_bits=20):
    DATA_DIR = os.path.join(BASE_DIR, "data/raw")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    LABEL_COL = "label"
    NAME_COL = "LabelCompoundName"
    SMILES_COL = "SMILES"

    dataset_files = glob(os.path.join(DATA_DIR, "*.csv"))
    if not dataset_files:
        raise ValueError("No dataset found.")

    for data_path in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]
        print(f"\n[INFO] Dataset: {dataset_name} - run_pipeline.py:42")

        # ================= Step 1: ECFP =================
        ecfp_path = os.path.join(DATA_DIR, f"{dataset_name}_ecfp{ecfp_bits}.tsv")
        print("[STEP 1] Generating ECFP features... - run_pipeline.py:46")
        extract_ecfp_features(
            input_path=data_path,
            output_path=ecfp_path,
            smiles_col=SMILES_COL,
            label_col=LABEL_COL,
            name_col=NAME_COL,
            radius=ecfp_radius,
            n_bits=ecfp_bits
        )

        # ================= Step 2: Load data =================
        ids, X, y, _ = load_data(ecfp_path, LABEL_COL, NAME_COL)
        print(f"[INFO] Loaded {X.shape[0]} samples, {X.shape[1]} features - run_pipeline.py:59")

        dataset_out = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(dataset_out, exist_ok=True)

        # ================= Step 3: Define experiments =================
        experiments = {
            "RF": {
                "builder": build_rf,
                "grid_search": rf_grid_search,
                "model_type": "rf"
            },
            "XGB": {
                "builder": build_xgb,
                "grid_search": xgb_grid_search,
                "model_type": "xgb"
            },
            "LogReg": {
                "builder": build_logreg,
                "grid_search": logreg_grid_search,
                "model_type": "logreg"
            }
        }

        # ================= Step 4: Run models =================
        for model_name, cfg in experiments.items():
            print(f"\n[MODEL] {model_name} - run_pipeline.py:85")

            # ---- 4.1 Grid search ----
            print("[STEP 4.1] Grid search... - run_pipeline.py:88")
            best_params, best_score = cfg["grid_search"](X, y)
            print(f"[INFO] Best params: {best_params} - run_pipeline.py:90")
            print(f"[INFO] Best CV score: {best_score:.4f} - run_pipeline.py:91")

            # ---- 4.2 5-fold CV + SHAP ----
            print("[STEP 4.2] 5fold CV + SHAP... - run_pipeline.py:94")
            run_5fold_cv(
                model_builder=lambda: cfg["builder"](**best_params),
                ids=ids,
                X=X,
                y=y,
                output_dir=dataset_out,
                model_name=model_name,
                model_type=cfg["model_type"]
            )

            # ---- 4.3 Extract top SHAP fragments ----
            print("[STEP 4.3] Extracting SHAP fragments... - run_pipeline.py:106")
            shap_path = os.path.join(dataset_out, model_name, "fold1", "training_shap.tsv")
            frag_out = os.path.join(dataset_out, "fragments", model_name)
            os.makedirs(frag_out, exist_ok=True)

            extract_top_shap_fragments(
                shap_df_path=shap_path,
                dataset_df_path=data_path,
                top_n=top_shap_bits,
                output_dir=frag_out,
                radius=ecfp_radius,
                n_bits=ecfp_bits
            )

    print("\n[INFO] Pipeline finished successfully! - run_pipeline.py:120")


if __name__ == "__main__":
    run_pipeline(ecfp_radius=3, ecfp_bits=1024, top_shap_bits=20)
