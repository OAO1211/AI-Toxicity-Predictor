# evaluation/property_comparison.py
"""
TP/TN/FP/FN physicochemical property comparison (Priority 2 of the
Prediction Error Analysis follow-up).

Answers: "does the model make mistakes in a particular part of
chemical space?" — i.e. are FP/FN compounds systematically different
(heavier, more lipophilic, more/fewer rotatable bonds, ...) from the
correctly classified compounds, rather than just listing which
compounds were wrong.

Two comparisons matter most and are computed explicitly:
    FP vs TN  (both true-label = non-DILI; does the model's mistake
               correlate with unusual descriptor values relative to
               the compounds it got right in the same true class?)
    FN vs TP  (both true-label = DILI; same idea)

Uses features.descriptors.calculate_descriptors so this reuses the
exact same descriptor definitions as the main feature pipeline,
computed fresh from SMILES for every compound in classified_df (not
just the errors), rather than re-deriving them differently here.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import mannwhitneyu

from features.descriptors import calculate_descriptors


def compute_compound_descriptors(
    dataset_df: pd.DataFrame,
    id_col: str,
    smiles_col: str,
    sample_ids: List[str]
) -> pd.DataFrame:
    """
    Computes the standard descriptor set (see
    features.descriptors.calculate_descriptors) for each id in
    sample_ids, looking up SMILES from dataset_df.

    Compounds with unparsable/missing SMILES are skipped with a
    warning rather than silently dropped without a trace.
    """

    smiles_lookup = (
        dataset_df.set_index(dataset_df[id_col].astype(str))[smiles_col]
        .to_dict()
    )

    rows = []

    for sample_id in sample_ids:

        sample_id = str(sample_id)
        smiles = smiles_lookup.get(sample_id)

        if smiles is None or pd.isna(smiles):
            print(
                f"[WARN] compute_compound_descriptors: SampleID "
                f"'{sample_id}' has no SMILES — skipped."
            )
            continue

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            print(
                f"[WARN] compute_compound_descriptors: SampleID "
                f"'{sample_id}' has an unparsable SMILES — skipped."
            )
            continue

        desc = calculate_descriptors(mol)
        desc["SampleID"] = sample_id

        rows.append(desc)

    if not rows:
        return pd.DataFrame(columns=["SampleID"])

    desc_df = pd.DataFrame(rows)

    cols = ["SampleID"] + [c for c in desc_df.columns if c != "SampleID"]

    return desc_df[cols]


def build_full_property_table(
    classified_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    id_col: str,
    smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    Merges classified_df (SampleID, ErrorType, ...) with per-compound
    descriptors, for every compound (not just errors) so TP/TN can be
    used as the baseline.
    """

    sample_ids = classified_df["SampleID"].astype(str).unique().tolist()

    desc_df = compute_compound_descriptors(
        dataset_df, id_col=id_col, smiles_col=smiles_col,
        sample_ids=sample_ids
    )

    merged = classified_df.copy()
    merged["SampleID"] = merged["SampleID"].astype(str)

    merged = merged.merge(desc_df, on="SampleID", how="left")

    return merged


def descriptor_columns(full_property_df: pd.DataFrame) -> List[str]:
    """
    Returns the list of descriptor columns present in a table produced
    by build_full_property_table (everything calculate_descriptors
    can return).
    """

    known = [
        "MolWt", "ExactMolWt", "MolLogP", "TPSA", "HBD", "HBA",
        "RotatableBonds", "RingCount", "AromaticRings", "AliphaticRings",
        "HeavyAtoms", "FractionCSP3", "MolMR", "LabuteASA", "BertzCT"
    ]

    return [c for c in known if c in full_property_df.columns]


def group_stats(
    full_property_df: pd.DataFrame,
    descriptor_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Mean / median / std / n of every descriptor, grouped by ErrorType
    (TP / TN / FP / FN).
    """

    descriptor_cols = descriptor_cols or descriptor_columns(full_property_df)

    stats = full_property_df.groupby("ErrorType")[descriptor_cols].agg(
        ["mean", "median", "std", "count"]
    )

    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]

    return stats.reset_index()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values), implemented
    directly (no statsmodels dependency, since it isn't already in
    requirements.txt and this is the only place that would need it).

    NaN entries (tests that didn't run, e.g. too few samples) are left
    as NaN and excluded from the correction — they were never part of
    the family of tests actually performed.
    """

    p = pd.Series(p_values).astype(float)
    valid = p.dropna()
    n = len(valid)

    result = p.copy()

    if n == 0:
        return result

    order = valid.sort_values().index
    ranks = np.arange(1, n + 1)
    sorted_p = valid.loc[order].values

    q = sorted_p * n / ranks
    # Enforce monotonicity: q-values must not decrease as rank
    # decreases (standard BH step-up correction).
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    result.loc[order] = q

    return result


def pairwise_group_test(
    full_property_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    descriptor_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    For every descriptor, runs a two-sided Mann-Whitney U test between
    group_a and group_b (e.g. "FP" vs "TN", "FN" vs "TP"), then applies
    a Benjamini-Hochberg FDR correction across all descriptors tested
    (running ~15-25 simultaneous tests without correction makes it easy
    to pick up a "significant" descriptor by chance alone).

    Non-parametric on purpose: descriptor distributions for a small
    error group are rarely normal, and n can be small.

    Groups with fewer than 3 samples are not tested (p_value/U left as
    NaN, with a Note) — Mann-Whitney U is not meaningful at that size,
    and NaN p-values are excluded from the FDR correction rather than
    treated as non-significant.

    Also reports a rank-biserial correlation as effect size
    (-1 to 1; magnitude matters more than the raw U statistic and
    doesn't depend on sample size the way p-values do).
    """

    descriptor_cols = descriptor_cols or descriptor_columns(full_property_df)

    a_df = full_property_df[full_property_df["ErrorType"] == group_a]
    b_df = full_property_df[full_property_df["ErrorType"] == group_b]

    rows = []

    for col in descriptor_cols:

        a = a_df[col].dropna()
        b = b_df[col].dropna()

        row = {
            "Descriptor": col,
            f"{group_a}_n": len(a),
            f"{group_b}_n": len(b),
            f"{group_a}_median": a.median() if len(a) else np.nan,
            f"{group_b}_median": b.median() if len(b) else np.nan,
            f"{group_a}_mean": a.mean() if len(a) else np.nan,
            f"{group_b}_mean": b.mean() if len(b) else np.nan,
        }

        if len(a) >= 3 and len(b) >= 3:
            try:
                u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
                row["U_statistic"] = u_stat
                row["p_value"] = p_val
                # Rank-biserial correlation: standard effect size for
                # Mann-Whitney U, in [-1, 1]. Positive means group_a
                # tends to have higher values than group_b.
                row["RankBiserialEffectSize"] = (
                    1 - (2 * u_stat) / (len(a) * len(b))
                )
                row["Note"] = ""
            except ValueError as e:
                row["U_statistic"] = np.nan
                row["p_value"] = np.nan
                row["RankBiserialEffectSize"] = np.nan
                row["Note"] = f"Mann-Whitney U failed: {e}"
        else:
            row["U_statistic"] = np.nan
            row["p_value"] = np.nan
            row["RankBiserialEffectSize"] = np.nan
            row["Note"] = (
                f"Too few samples (n<3) in {group_a} and/or {group_b} "
                f"to run a Mann-Whitney U test."
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    result["FDR_q_value"] = benjamini_hochberg(result["p_value"])

    # Kept for reference, but FDR_q_value is the column that should
    # actually be used to decide significance when testing this many
    # descriptors at once — uncorrected p<0.05 across ~15-25
    # simultaneous tests will falsely flag descriptors by chance.
    result["Significant_p<0.05_uncorrected"] = result["p_value"] < 0.05
    result["Significant_q<0.05_FDR"] = result["FDR_q_value"] < 0.05

    return result


def run_property_comparison(
    classified_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    id_col: str,
    output_path: str,
    smiles_col: str = "SMILES"
) -> Optional[dict]:
    """
    Orchestrator: builds the full property table (every compound,
    every descriptor), computes group-level stats, and runs the two
    comparisons that matter most (FP vs TN, FN vs TP), then writes
    everything to one multi-sheet .xlsx (property_group_comparison.xlsx):

        GroupStats   - mean/median/std/n per ErrorType per descriptor
        FP_vs_TN     - Mann-Whitney U test per descriptor
        FN_vs_TP     - Mann-Whitney U test per descriptor

    Returns None (with a printed warning) if there aren't enough valid
    compounds (e.g. everything failed SMILES parsing) to say anything.
    """

    full_df = build_full_property_table(
        classified_df, dataset_df, id_col=id_col, smiles_col=smiles_col
    )

    desc_cols = descriptor_columns(full_df)

    if not desc_cols or full_df[desc_cols].dropna(how="all").empty:
        print(
            "[WARN] run_property_comparison: no usable descriptor "
            "values — skipping property comparison."
        )
        return None

    stats_df = group_stats(full_df, desc_cols)

    has_fp_tn = (
        (full_df["ErrorType"] == "FP").sum() > 0
        and (full_df["ErrorType"] == "TN").sum() > 0
    )
    has_fn_tp = (
        (full_df["ErrorType"] == "FN").sum() > 0
        and (full_df["ErrorType"] == "TP").sum() > 0
    )

    fp_vs_tn = (
        pairwise_group_test(full_df, "FP", "TN", desc_cols)
        if has_fp_tn else pd.DataFrame()
    )
    fn_vs_tp = (
        pairwise_group_test(full_df, "FN", "TP", desc_cols)
        if has_fn_tp else pd.DataFrame()
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        stats_df.to_excel(writer, sheet_name="GroupStats", index=False)

        if not fp_vs_tn.empty:
            fp_vs_tn.to_excel(writer, sheet_name="FP_vs_TN", index=False)
        if not fn_vs_tp.empty:
            fn_vs_tp.to_excel(writer, sheet_name="FN_vs_TP", index=False)

    n_sig_fp_tn = (
        int(fp_vs_tn["Significant_q<0.05_FDR"].sum())
        if not fp_vs_tn.empty else 0
    )
    n_sig_fn_tp = (
        int(fn_vs_tp["Significant_q<0.05_FDR"].sum())
        if not fn_vs_tp.empty else 0
    )

    print(
        f"[INFO] run_property_comparison: saved to {output_path} "
        f"(FP vs TN: {n_sig_fp_tn}/{len(fp_vs_tn) if not fp_vs_tn.empty else 0} "
        f"descriptors significant at FDR q<0.05; "
        f"FN vs TP: {n_sig_fn_tp}/{len(fn_vs_tp) if not fn_vs_tp.empty else 0} "
        f"descriptors significant at FDR q<0.05)"
    )

    return {
        "full_df": full_df,
        "stats_df": stats_df,
        "fp_vs_tn": fp_vs_tn,
        "fn_vs_tp": fn_vs_tp
    }
