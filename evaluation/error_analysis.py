# evaluation/error_analysis.py
"""
Prediction Error Analysis (Chapter 4).

Turns the out-of-fold predictions produced by
evaluation.cross_validation.run_5fold_cv into:

  1. prediction_errors.tsv           - every FP/FN, with confidence
                                        and a high-confidence flag
                                        (item 2 / item 3)
  2. error_compound_shap_fragments.tsv - per-error-compound SHAP bits
                                        -> molecular fragments, selected
                                        by DIRECTION relevant to the
                                        error (FP -> top positive SHAP,
                                        FN -> top negative SHAP), not
                                        just |SHAP| (uses OOF SHAP / all
                                        folds, item 4 / item 7)
  3. error_nearest_neighbors.tsv     - Top-K ECFP-Tanimoto neighbors
                                        for each HIGH-CONFIDENCE FP/FN,
                                        with a LabelMismatch flag
                                        (item 5)
  4. error_similarity_label_disagreement.tsv - for each high-confidence
                                        error compound, what fraction
                                        of ALL (not just Top-K) close
                                        neighbors (similarity >=
                                        threshold) have the OPPOSITE
                                        true label
  5. property_group_comparison.xlsx  - TP/TN/FP/FN physicochemical
                                        descriptor comparison, with
                                        Mann-Whitney U tests for
                                        FP vs TN and FN vs TP
  6. error_analysis_summary.xlsx     - one row per FP/FN, combining all
                                        of the above (item 6)
  7. top_10_worst_errors.tsv         - worst errors ranked by
                                        |pred_prob - y_true|, regardless
                                        of the HighConfidenceError flag
                                        (item 9) — named for what it
                                        actually is, not "high
                                        confidence"

This module only READS results that run_5fold_cv already wrote to disk
(oof_predictions.tsv, oof_shap.tsv) — it does not retrain anything, so
it can be re-run cheaply on its own (see run_error_analysis.py).
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from features.similarity import (
    find_nearest_neighbors_for_ids,
    find_neighbors_above_threshold,
    summarize_label_disagreement
)
from explain.error_fragments import extract_error_compound_shap_fragments
from evaluation.property_comparison import run_property_comparison


# =========================
# Step 1: classify predictions
# =========================

def classify_predictions(
    oof_df: pd.DataFrame,
    fp_threshold: float = 0.8,
    fn_threshold: float = 0.2
) -> pd.DataFrame:
    """
    Adds ErrorType / Confidence / AbsError / HighConfidenceError columns
    to an OOF predictions table.

    ErrorType is one of TP / TN / FP / FN, based on y_true vs pred_label
    (pred_label is the model's own predict() output, i.e. whatever
    decision threshold the estimator itself uses — normally 0.5).

    Confidence = abs(pred_prob - 0.5) * 2, rescaled to [0, 1]: 0 means
    the model was maximally unsure (pred_prob == 0.5), 1 means maximally
    confident in whichever class it predicted.

    AbsError = abs(pred_prob - y_true): used later to rank the worst
    errors (item 9) — this is the distance between the predicted
    probability and the true label, not the same thing as Confidence
    (a low-confidence prediction can still be a "small" error, and a
    high-confidence prediction that's wrong produces the largest
    possible AbsError).

    HighConfidenceError is True for:
        FP with pred_prob >= fp_threshold
        FN with pred_prob <= fn_threshold
    """

    df = oof_df.copy()

    y_true = df["y_true"].astype(int)
    y_pred = df["pred_label"].astype(int)

    conditions = [
        (y_true == 1) & (y_pred == 1),
        (y_true == 0) & (y_pred == 0),
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 0),
    ]
    choices = ["TP", "TN", "FP", "FN"]

    df["ErrorType"] = np.select(conditions, choices, default="UNKNOWN")

    df["Confidence"] = (df["pred_prob"] - 0.5).abs() * 2
    df["AbsError"] = (df["pred_prob"] - y_true).abs()

    df["HighConfidenceError"] = (
        ((df["ErrorType"] == "FP") & (df["pred_prob"] >= fp_threshold))
        | ((df["ErrorType"] == "FN") & (df["pred_prob"] <= fn_threshold))
    )

    if "UNKNOWN" in df["ErrorType"].values:
        n_unknown = int((df["ErrorType"] == "UNKNOWN").sum())
        print(
            f"[WARN] classify_predictions: {n_unknown} rows have "
            f"y_true/pred_label values other than 0/1 and could not be "
            f"classified as TP/TN/FP/FN."
        )

    return df


# =========================
# Step 2/3: save error tables
# =========================

def save_prediction_errors(
    classified_df: pd.DataFrame,
    output_path: str
) -> pd.DataFrame:
    """
    Saves EVERY FP/FN (not just high-confidence ones) to
    prediction_errors.tsv (item 2 + the "另外保留所有 FP/FN" note in
    item 3).
    """

    error_df = classified_df[
        classified_df["ErrorType"].isin(["FP", "FN"])
    ].sort_values(
        ["ErrorType", "Confidence"],
        ascending=[True, False]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    error_df.to_csv(output_path, sep="\t", index=False)

    n_high_conf = int(error_df["HighConfidenceError"].sum())

    print(
        f"[INFO] save_prediction_errors: {len(error_df)} FP/FN saved "
        f"to {output_path} ({n_high_conf} high-confidence)"
    )

    return error_df


def save_top_n_worst_errors(
    classified_df: pd.DataFrame,
    output_path: str,
    n: int = 10
) -> pd.DataFrame:
    """
    Ranks all FP/FN by |pred_prob - y_true| (AbsError) descending and
    keeps the worst n (item 9).
    """

    error_df = classified_df[
        classified_df["ErrorType"].isin(["FP", "FN"])
    ].copy()

    top_df = error_df.sort_values(
        "AbsError", ascending=False
    ).head(n).reset_index(drop=True)

    top_df.insert(0, "ErrorRank", range(1, len(top_df) + 1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_df.to_csv(output_path, sep="\t", index=False)

    print(
        f"[INFO] save_top_n_worst_errors: top {len(top_df)} worst "
        f"errors saved to {output_path}"
    )

    return top_df


# =========================
# Step 6: final summary table
# =========================

def build_error_analysis_summary(
    error_df: pd.DataFrame,
    fragments_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    output_path: str,
    disagreement_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Combines error classification + SHAP fragments + nearest neighbors
    + similarity label-disagreement into one row-per-compound summary
    table (item 6), and writes it to an .xlsx file with at least the
    columns:

        SampleID, TrueLabel, PredProb, PredLabel, ErrorType, Confidence,
        Top_SHAP_Bits, Top_Fragment, NearestNeighbor, Similarity

    Every FP/FN is included (fragments_df / neighbors_df /
    disagreement_df may only cover a subset — e.g. neighbors are
    usually only computed for high-confidence errors — in which case
    those columns are left blank for the rest, rather than dropping
    the row).
    """

    summary = error_df[[
        "SampleID", "y_true", "pred_prob", "pred_label",
        "ErrorType", "Confidence", "HighConfidenceError"
    ]].rename(columns={
        "y_true": "TrueLabel",
        "pred_prob": "PredProb",
        "pred_label": "PredLabel"
    }).copy()

    # ---- Top SHAP bits / fragment (one row per compound) ----
    #
    # fragments_df bits are already direction-aware (FP -> top positive
    # SHAP i.e. wrongly pushed toward DILI; FN -> top negative SHAP i.e.
    # wrongly pushed toward non-DILI) — see explain/error_fragments.py.

    if fragments_df is not None and not fragments_df.empty:

        top_frag = (
            fragments_df
            .sort_values(["SampleID", "Rank"])
            .groupby("SampleID")
            .first()
            .reset_index()
        )

        bits_by_id = (
            fragments_df
            .groupby("SampleID")["BitIndex"]
            .apply(lambda s: ", ".join(str(b) for b in s))
            .reset_index()
            .rename(columns={"BitIndex": "Top_SHAP_Bits"})
        )

        summary = summary.merge(
            bits_by_id, on="SampleID", how="left"
        )

        summary = summary.merge(
            top_frag[["SampleID", "FragmentSMILES"]].rename(
                columns={"FragmentSMILES": "Top_Fragment"}
            ),
            on="SampleID",
            how="left"
        )

    else:
        summary["Top_SHAP_Bits"] = np.nan
        summary["Top_Fragment"] = np.nan

    # ---- Nearest neighbor (top-1 only, one row per compound) ----

    if neighbors_df is not None and not neighbors_df.empty:

        top1_neighbor = (
            neighbors_df[neighbors_df["Rank"] == 1]
            [["SampleID", "NeighborID", "Similarity"]]
            .rename(columns={"NeighborID": "NearestNeighbor"})
        )

        summary = summary.merge(top1_neighbor, on="SampleID", how="left")

    else:
        summary["NearestNeighbor"] = np.nan
        summary["Similarity"] = np.nan

    # ---- Similarity label-disagreement (fraction of ALL close
    #      neighbors, not just Top-K, with the opposite true label) ----

    if disagreement_df is not None and not disagreement_df.empty:

        summary = summary.merge(
            disagreement_df[[
                "SampleID", "NeighborsAboveThreshold", "MismatchFraction"
            ]],
            on="SampleID",
            how="left"
        )

    else:
        summary["NeighborsAboveThreshold"] = np.nan
        summary["MismatchFraction"] = np.nan

    column_order = [
        "SampleID", "TrueLabel", "PredProb", "PredLabel",
        "ErrorType", "Confidence", "HighConfidenceError",
        "Top_SHAP_Bits", "Top_Fragment", "NearestNeighbor", "Similarity",
        "NeighborsAboveThreshold", "MismatchFraction"
    ]

    summary = summary[column_order]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary.to_excel(output_path, index=False)

    print(
        f"[INFO] build_error_analysis_summary: {len(summary)} rows "
        f"saved to {output_path}"
    )

    return summary


# =========================
# Orchestrator: everything for one (dataset, feature_set, model)
# =========================

def run_error_analysis_for_model(
    model_output_dir: str,
    dataset_df: pd.DataFrame,
    id_col: str,
    smiles_col: str = "SMILES",
    label_col: Optional[str] = None,
    fp_threshold: float = 0.8,
    fn_threshold: float = 0.2,
    top_n_shap_bits: int = 5,
    similarity_top_k: int = 5,
    similarity_label_mismatch_threshold: float = 0.7,
    top_n_worst_errors: int = 10,
    ecfp_radius: int = 3,
    ecfp_n_bits: int = 1024,
    ecfp_use_chirality: bool = False
) -> Optional[dict]:
    """
    Runs the full error-analysis pipeline for one already-trained
    (dataset, feature_set, model) combination.

    model_output_dir: the directory run_5fold_cv wrote
        oof_predictions.tsv / oof_shap.tsv / mean_abs_shap.tsv into
        (i.e. results/<dataset>/<FeatureSet>/<ModelName>).

    dataset_df: the ORIGINAL raw dataset (with SMILES / labels), used
        to look up SMILES and (optionally) true labels for the
        structural-similarity step. This is NOT the ECFP feature
        table — it's the same file passed to
        features.ecfp.extract_ecfp_features as input_path.

    id_col: column in dataset_df that matches the SampleID values used
        throughout the CV pipeline (LABEL_COL's companion — in this
        project, config.NAME_COL / "LabelCompoundName").

    Returns a dict of the produced DataFrames (or None if
    oof_predictions.tsv doesn't exist yet, e.g. this feature_set/model
    hasn't been trained), so callers/tests can inspect results without
    re-reading files from disk.
    """

    oof_pred_path = os.path.join(model_output_dir, "oof_predictions.tsv")
    oof_shap_path = os.path.join(model_output_dir, "oof_shap.tsv")

    if not os.path.exists(oof_pred_path):
        print(
            f"[WARN] run_error_analysis_for_model: no "
            f"oof_predictions.tsv found in {model_output_dir} — this "
            f"model/feature_set was likely trained with an older "
            f"version of cross_validation.py that didn't save OOF "
            f"predictions. Skipping error analysis here; re-run "
            f"run_pipeline.py to regenerate it."
        )
        return None

    error_dir = os.path.join(model_output_dir, "error_analysis")
    os.makedirs(error_dir, exist_ok=True)

    # -------------------------
    # 1. Classify OOF predictions
    # -------------------------

    oof_df = pd.read_csv(oof_pred_path, sep="\t")

    classified_df = classify_predictions(
        oof_df,
        fp_threshold=fp_threshold,
        fn_threshold=fn_threshold
    )

    error_df = save_prediction_errors(
        classified_df,
        output_path=os.path.join(error_dir, "prediction_errors.tsv")
    )

    top10_df = save_top_n_worst_errors(
        classified_df,
        output_path=os.path.join(
            error_dir, "top_10_worst_errors.tsv"
        ),
        n=top_n_worst_errors
    )

    high_conf_ids = error_df.loc[
        error_df["HighConfidenceError"], "SampleID"
    ].astype(str).tolist()

    all_error_ids = error_df["SampleID"].astype(str).tolist()

    error_type_lookup = dict(
        zip(
            error_df["SampleID"].astype(str),
            error_df["ErrorType"]
        )
    )

    # -------------------------
    # 2. SHAP -> fragment, for every FP/FN (uses OOF SHAP, all folds).
    #    Bits are selected by DIRECTION relevant to the error type
    #    (FP -> top positive SHAP, FN -> top negative SHAP), not |SHAP|.
    # -------------------------

    fragments_df = pd.DataFrame()

    if os.path.exists(oof_shap_path) and len(all_error_ids) > 0:

        oof_shap_df = pd.read_csv(oof_shap_path, sep="\t")

        fragments_df = extract_error_compound_shap_fragments(
            error_ids=all_error_ids,
            oof_shap_df=oof_shap_df,
            dataset_df=dataset_df,
            id_col=id_col,
            smiles_col=smiles_col,
            top_n_bits=top_n_shap_bits,
            radius=ecfp_radius,
            n_bits=ecfp_n_bits,
            use_chirality=ecfp_use_chirality,
            error_type_lookup=error_type_lookup
        )

        if not fragments_df.empty:
            fragments_df.to_csv(
                os.path.join(
                    error_dir, "error_compound_shap_fragments.tsv"
                ),
                sep="\t",
                index=False
            )
            print(
                f"[INFO] run_error_analysis_for_model: SHAP fragments "
                f"for {fragments_df['SampleID'].nunique()} error "
                f"compounds saved."
            )
    else:
        print(
            "[WARN] run_error_analysis_for_model: no oof_shap.tsv "
            "found (or no errors) — skipping SHAP fragment extraction."
        )

    # -------------------------
    # 3. Structural similarity, for HIGH-CONFIDENCE FP/FN only
    # -------------------------

    neighbors_df = pd.DataFrame()

    if len(high_conf_ids) > 0:

        neighbors_df = find_nearest_neighbors_for_ids(
            dataset_df=dataset_df,
            target_ids=high_conf_ids,
            id_col=id_col,
            smiles_col=smiles_col,
            label_col=label_col,
            radius=ecfp_radius,
            n_bits=ecfp_n_bits,
            top_k=similarity_top_k,
            use_chirality=ecfp_use_chirality
        )

        if not neighbors_df.empty:
            neighbors_df.to_csv(
                os.path.join(error_dir, "error_nearest_neighbors.tsv"),
                sep="\t",
                index=False
            )
            print(
                f"[INFO] run_error_analysis_for_model: nearest "
                f"neighbors for {neighbors_df['SampleID'].nunique()} "
                f"high-confidence error compounds saved."
            )
    else:
        print(
            "[INFO] run_error_analysis_for_model: no high-confidence "
            "FP/FN for this model/feature_set — skipping similarity "
            "search."
        )

    # -------------------------
    # 3b. Similarity label-disagreement, for HIGH-CONFIDENCE FP/FN
    #     — uses ALL neighbors above the similarity threshold, not
    #     just Top-K, so it isn't understated by the Top-K cutoff.
    # -------------------------

    disagreement_df = pd.DataFrame()

    if len(high_conf_ids) > 0 and label_col is not None:

        neighbors_above_df = find_neighbors_above_threshold(
            dataset_df=dataset_df,
            target_ids=high_conf_ids,
            id_col=id_col,
            smiles_col=smiles_col,
            label_col=label_col,
            radius=ecfp_radius,
            n_bits=ecfp_n_bits,
            threshold=similarity_label_mismatch_threshold,
            use_chirality=ecfp_use_chirality
        )

        disagreement_df = summarize_label_disagreement(
            neighbors_above_df,
            target_ids=high_conf_ids,
            threshold=similarity_label_mismatch_threshold
        )

        disagreement_df.to_csv(
            os.path.join(
                error_dir, "error_similarity_label_disagreement.tsv"
            ),
            sep="\t",
            index=False
        )

        n_with_neighbors = int(
            (disagreement_df["NeighborsAboveThreshold"] > 0).sum()
        )

        print(
            f"[INFO] run_error_analysis_for_model: similarity "
            f"label-disagreement computed for {len(disagreement_df)} "
            f"high-confidence error compounds ({n_with_neighbors} have "
            f"at least one neighbor with similarity >= "
            f"{similarity_label_mismatch_threshold})."
        )

    # -------------------------
    # 4. TP/TN/FP/FN physicochemical property comparison
    #    (does the model err in a particular part of chemical space?)
    # -------------------------

    run_property_comparison(
        classified_df=classified_df,
        dataset_df=dataset_df,
        id_col=id_col,
        smiles_col=smiles_col,
        output_path=os.path.join(
            error_dir, "property_group_comparison.xlsx"
        )
    )

    # -------------------------
    # 5. Final summary
    # -------------------------

    summary_df = build_error_analysis_summary(
        error_df=error_df,
        fragments_df=fragments_df,
        neighbors_df=neighbors_df,
        disagreement_df=disagreement_df,
        output_path=os.path.join(error_dir, "error_analysis_summary.xlsx")
    )

    return {
        "classified_df": classified_df,
        "error_df": error_df,
        "top10_df": top10_df,
        "fragments_df": fragments_df,
        "neighbors_df": neighbors_df,
        "disagreement_df": disagreement_df,
        "summary_df": summary_df
    }
