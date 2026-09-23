# run_error_analysis.py
"""
Standalone entry point for Prediction Error Analysis (Chapter 4).

This does NOT retrain anything — it only reads the oof_predictions.tsv
/ oof_shap.tsv files that evaluation.cross_validation.run_5fold_cv
already wrote to disk (via run_pipeline.py), and produces:

    <output_dir>/<Model>/error_analysis/
        prediction_errors.tsv
        top_10_high_confidence_errors.tsv
        error_compound_shap_fragments.tsv
        error_nearest_neighbors.tsv
        error_analysis_summary.xlsx

for every (dataset, feature_set, model) combination it finds under
--results-dir that has an oof_predictions.tsv.

Usage
-----
    python run_error_analysis.py
    python run_error_analysis.py --results-dir results
    python run_error_analysis.py --datasets dili_binary_labeled
    python run_error_analysis.py --feature-sets ECFP,Combined_Naive
    python run_error_analysis.py --models RF,XGB

Run this any time after run_pipeline.py has produced
oof_predictions.tsv for at least one model — you do not need to wait
for every feature set / model to finish, and re-running it is cheap
(no retraining, no grid search).
"""

import argparse
import glob
import os

import pandas as pd

from config import (
    RAW_DATA_DIR,
    CURATED_DATA_DIR,
    RESULTS_DIR,
    NAME_COL,
    SMILES_COL,
    LABEL_COL,
    ECFP_RADIUS,
    ECFP_BITS,
    ECFP_USE_CHIRALITY,
    ERROR_FP_HIGH_CONF_THRESHOLD,
    ERROR_FN_HIGH_CONF_THRESHOLD,
    TOP_SHAP_BITS_PER_COMPOUND,
    SIMILARITY_TOP_K,
    SIMILARITY_LABEL_MISMATCH_THRESHOLD,
    TOP_N_WORST_ERRORS
)

from evaluation.error_analysis import run_error_analysis_for_model


def _find_oof_prediction_files(results_dir):
    """
    Directory layout produced by run_pipeline.py:
        results_dir/<dataset_name>/<FeatureSet>/<ModelName>/oof_predictions.tsv
    """
    return glob.glob(
        os.path.join(results_dir, "*", "*", "*", "oof_predictions.tsv")
    )


def _load_raw_dataset(dataset_name):
    """
    Finds and loads the original raw CSV for a dataset_name (the same
    file run_pipeline.py fed into features.ecfp / features.descriptors),
    matched by filename stem exactly as run_pipeline.py derives it:
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    """

    candidates = [
        os.path.join(CURATED_DATA_DIR, f"{dataset_name}.csv"),
        os.path.join(RAW_DATA_DIR, f"{dataset_name}.csv"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return pd.read_csv(candidate, encoding="latin1")

    raise FileNotFoundError(
        f"Could not find dataset for '{dataset_name}' in data/curated/ "
        f"or data/raw/. Error analysis needs the same CSV used during "
        f"the model run to reconstruct fragments and similarity."
    )


def run_error_analysis(
    results_dir=None,
    datasets=None,
    feature_sets=None,
    models=None
):
    results_dir = results_dir or RESULTS_DIR

    oof_files = _find_oof_prediction_files(results_dir)

    if not oof_files:
        print(
            f"[WARN] No oof_predictions.tsv found under {results_dir}. "
            f"Run run_pipeline.py first (with the updated "
            f"evaluation/cross_validation.py that saves OOF "
            f"predictions)."
        )
        return

    dataset_cache = {}

    n_run = 0

    for oof_file in sorted(oof_files):

        model_output_dir = os.path.dirname(oof_file)
        feature_set_dir = os.path.dirname(model_output_dir)
        dataset_dir = os.path.dirname(feature_set_dir)

        model_name = os.path.basename(model_output_dir)
        feature_set_name = os.path.basename(feature_set_dir)
        dataset_name = os.path.basename(dataset_dir)

        if datasets and dataset_name not in datasets:
            continue
        if feature_sets and feature_set_name not in feature_sets:
            continue
        if models and model_name not in models:
            continue

        print(
            f"\n[ERROR ANALYSIS] {dataset_name} / {feature_set_name} "
            f"/ {model_name}"
        )

        if dataset_name not in dataset_cache:
            dataset_cache[dataset_name] = _load_raw_dataset(dataset_name)

        dataset_df = dataset_cache[dataset_name]

        run_error_analysis_for_model(
            model_output_dir=model_output_dir,
            dataset_df=dataset_df,
            id_col=NAME_COL,
            smiles_col=SMILES_COL,
            label_col=LABEL_COL,
            fp_threshold=ERROR_FP_HIGH_CONF_THRESHOLD,
            fn_threshold=ERROR_FN_HIGH_CONF_THRESHOLD,
            top_n_shap_bits=TOP_SHAP_BITS_PER_COMPOUND,
            similarity_top_k=SIMILARITY_TOP_K,
            similarity_label_mismatch_threshold=SIMILARITY_LABEL_MISMATCH_THRESHOLD,
            top_n_worst_errors=TOP_N_WORST_ERRORS,
            ecfp_radius=ECFP_RADIUS,
            ecfp_n_bits=ECFP_BITS,
            ecfp_use_chirality=ECFP_USE_CHIRALITY
        )

        n_run += 1

    print(f"\n[DONE] Error analysis finished for {n_run} model(s). - run_error_analysis.py:163")


def _parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run Prediction Error Analysis (FP/FN classification, "
            "SHAP fragment mapping, structural similarity, summary "
            "table) on already-produced run_pipeline.py results."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Defaults to config.RESULTS_DIR ('results')."
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to restrict to (optional)."
    )

    parser.add_argument(
        "--feature-sets",
        type=str,
        default=None,
        help=(
            "Comma-separated feature sets to restrict to, e.g. "
            "ECFP,Combined_Naive (optional)."
        )
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to restrict to, e.g. RF,XGB."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    run_error_analysis(
        results_dir=args.results_dir,
        datasets=(
            [s.strip() for s in args.datasets.split(",")]
            if args.datasets else None
        ),
        feature_sets=(
            [s.strip() for s in args.feature_sets.split(",")]
            if args.feature_sets else None
        ),
        models=(
            [s.strip() for s in args.models.split(",")]
            if args.models else None
        )
    )
