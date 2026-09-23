# explain/error_fragments.py
"""
Map SHAP explanations to molecular fragments for individual FP/FN
compounds (item 4 of the Prediction Error Analysis request).

Unlike features.bit_mapping.extract_top_shap_fragments (which reports
one dataset-wide Top-N SHAP bit list), this module works PER COMPOUND:
for each misclassified compound, it looks at that compound's own OOF
SHAP values and reconstructs the corresponding substructure from that
compound's own SMILES.

Direction-aware bit selection
------------------------------
SHAP sign convention in this project (see explain/shap_analysis.py):
positive SHAP pushes the prediction TOWARD class 1 (DILI / toxic),
negative SHAP pushes it toward class 0 (non-DILI).

Simply ranking by |SHAP| answers "which feature mattered most" — it
does NOT tell you whether that feature is what caused the error. For
error compounds, the more useful question is direction-specific:

    FP (true=non-DILI, predicted=DILI):
        the interesting bits are the ones with the most POSITIVE SHAP
        — the structural signals that wrongly pushed the model toward
        "DILI".

    FN (true=DILI, predicted=non-DILI):
        the interesting bits are the ones with the most NEGATIVE SHAP
        — the structural signals that wrongly pushed the model toward
        "non-DILI" (or failed to push it toward DILI hard enough).

So when an ErrorType is supplied for a compound, this module selects
bits by *signed* SHAP value in the direction relevant to that error
instead of by |SHAP|. If no ErrorType is available for a compound, it
falls back to |SHAP| ranking (documented via SelectionCriterion) so
this still works for e.g. TP/TN diagnostics if ever needed.

Bits are restricted to ones PRESENT in the molecule before ranking
----------------------------------------------------------------
A binary ECFP feature can carry a non-zero SHAP value for a molecule
even when the bit is 0 (absent) for that molecule — SHAP explains the
deviation caused by a feature taking its actual value (0 or 1) versus
the background distribution, so "bit absent" is itself a valid
contribution, not noise. But an absent bit has no atom environment
to reconstruct ON THIS MOLECULE (bit_to_fragment() only finds atom
environments for bits RDKit's own Morgan-fingerprint bitInfo marked as
set), so ranking candidate bits by SHAP FIRST — before checking
whether the bit is even present — can select an absent bit as "top",
which then just produces an uninterpretable None fragment, even when
a same-ranked PRESENT bit would have resolved cleanly.

So candidate bits are filtered to features.bit_mapping.get_present_bits(
smiles) BEFORE ranking, for every compound. If a compound has no
present ECFP bits at all in oof_shap_df's columns (shouldn't normally
happen unless SMILES parsing disagrees between feature extraction and
this analysis), all-bit ranking is used as a fallback with a Note
flagging it, rather than silently returning nothing.
"""

from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem

from features.bit_mapping import bit_to_fragment, get_present_bits


_DIRECTION_BY_ERROR_TYPE = {
    "FP": "positive",   # wrongly pushed toward DILI (class 1)
    "FN": "negative",   # wrongly pushed toward non-DILI (class 0)
}


def _select_top_bits(shap_row: pd.Series, top_n: int, direction: Optional[str]):
    """
    Returns a list of (bit_col, signed_shap_value, selection_criterion)
    tuples, length up to top_n.

    direction: "positive" -> highest signed SHAP first
               "negative" -> lowest (most negative) signed SHAP first
               None        -> highest |SHAP| first (fallback)
    """

    if direction == "positive":
        ranked = shap_row.sort_values(ascending=False)
        criterion = "top_positive_shap"
    elif direction == "negative":
        ranked = shap_row.sort_values(ascending=True)
        criterion = "top_negative_shap"
    else:
        ranked = shap_row.reindex(
            shap_row.abs().sort_values(ascending=False).index
        )
        criterion = "top_abs_shap"

    top = ranked.head(top_n)

    return [(col, float(val), criterion) for col, val in top.items()]


def extract_error_compound_shap_fragments(
    error_ids: List[str],
    oof_shap_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    id_col: str,
    smiles_col: str = "SMILES",
    top_n_bits: int = 5,
    radius: int = 3,
    n_bits: int = 1024,
    use_chirality: bool = False,
    error_type_lookup: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Parameters
    ----------
    error_ids   : SampleIDs (FP/FN compounds) to analyze.
    oof_shap_df : concatenated out-of-fold SHAP table (all folds),
                  as produced by evaluation.cross_validation.run_5fold_cv
                  (oof_shap.tsv) — must contain a "SampleID" column and
                  one column per ECFP_<bit> feature. Only meaningful for
                  feature sets that contain ECFP bits (ECFP,
                  Combined_Naive, Combined_Scaled); returns an empty
                  DataFrame otherwise.
    dataset_df  : original dataset with [id_col, smiles_col], used to
                  look up each compound's SMILES so the fragment can be
                  reconstructed from ITS OWN structure.
    top_n_bits  : how many bits to report per compound.
    error_type_lookup : optional dict {SampleID: "FP"/"FN"/...}. When a
                  compound's ErrorType is "FP" or "FN", bits are
                  selected by SIGNED SHAP value in the direction
                  relevant to that error (see module docstring) instead
                  of by |SHAP|. Falls back to |SHAP| ranking for
                  compounds not in the lookup or with any other
                  ErrorType.

    Returns
    -------
    DataFrame with columns:
        SampleID, ErrorType, Rank, BitIndex, SHAP_value,
        SelectionCriterion, FragmentSMILES, Note
    One row per (compound, selected bit) — up to top_n_bits rows per
    compound that has ECFP columns available and a resolvable SMILES.

    Important caveat (keep this in the write-up, not just the code):
    a SHAP-important ECFP bit is an important HASHED FEATURE, not
    automatically a single, unambiguous functional group — ECFP bit
    collisions mean the same bit index can correspond to different
    substructures across molecules. Report this as "SHAP identified an
    important ECFP feature; RDKit was used to retrieve a representative
    atom environment for that bit on this molecule" rather than
    "the model identified functional group X as causing DILI".
    """

    ecfp_cols = [c for c in oof_shap_df.columns if c.startswith("ECFP_")]

    if not ecfp_cols:
        print(
            "[WARN] extract_error_compound_shap_fragments: no ECFP_ "
            "columns found in oof_shap_df — this feature set has no "
            "bit -> fragment mapping (e.g. Descriptors-only). Skipping."
        )
        return pd.DataFrame(
            columns=[
                "SampleID", "ErrorType", "Rank", "BitIndex",
                "SHAP_value", "SelectionCriterion",
                "FragmentSMILES", "Note"
            ]
        )

    smiles_lookup: Dict[str, str] = (
        dataset_df.set_index(dataset_df[id_col].astype(str))[smiles_col]
        .to_dict()
    )

    shap_by_id = oof_shap_df.set_index(
        oof_shap_df["SampleID"].astype(str)
    )

    error_type_lookup = error_type_lookup or {}

    rows = []

    for sample_id in error_ids:

        sample_id = str(sample_id)

        if sample_id not in shap_by_id.index:
            print(
                f"[WARN] extract_error_compound_shap_fragments: "
                f"SampleID '{sample_id}' has no OOF SHAP row — skipped."
            )
            continue

        smiles = smiles_lookup.get(sample_id)

        if smiles is None or pd.isna(smiles):
            print(
                f"[WARN] extract_error_compound_shap_fragments: "
                f"SampleID '{sample_id}' has no SMILES in dataset_df "
                f"— skipped."
            )
            continue

        shap_row = shap_by_id.loc[sample_id, ecfp_cols]

        # A compound can (rarely) have duplicate SampleID rows across
        # folds if something upstream went wrong; guard against getting
        # a DataFrame instead of a Series here.
        if isinstance(shap_row, pd.DataFrame):
            shap_row = shap_row.iloc[0]

        # -------------------------
        # Restrict candidates to bits actually present in THIS
        # molecule before ranking (see module docstring).
        # -------------------------

        present_bits = get_present_bits(
            smiles, radius=radius, n_bits=n_bits,
            use_chirality=use_chirality
        )

        present_cols = [
            col for col in ecfp_cols
            if int(col.replace("ECFP_", "")) in present_bits
        ]

        fallback_used = False

        if present_cols:
            candidate_row = shap_row[present_cols]
        else:
            print(
                f"[WARN] extract_error_compound_shap_fragments: "
                f"SampleID '{sample_id}' has no ECFP bits marked as "
                f"present for this SMILES (radius={radius}, "
                f"n_bits={n_bits}) among the columns in oof_shap_df — "
                f"falling back to ranking ALL bits (results will "
                f"include a bit not present in this molecule)."
            )
            candidate_row = shap_row
            fallback_used = True

        error_type = error_type_lookup.get(sample_id)
        direction = _DIRECTION_BY_ERROR_TYPE.get(error_type)

        selected = _select_top_bits(candidate_row, top_n_bits, direction)

        for rank, (col, signed_shap, criterion) in enumerate(selected, 1):

            bit_idx = int(col.replace("ECFP_", ""))

            frag_dict = bit_to_fragment(
                smiles,
                [bit_idx],
                radius=radius,
                n_bits=n_bits,
                use_chirality=use_chirality
            )

            mol = frag_dict.get(bit_idx)
            fragment_smiles: Optional[str] = (
                Chem.MolToSmiles(mol) if mol is not None else None
            )

            if fragment_smiles is not None:
                note = ""
            elif fallback_used:
                note = (
                    "This bit is not present in this molecule (fallback "
                    "ranking was used because no ECFP bit in "
                    "oof_shap_df was flagged as present for this "
                    "SMILES) — fragment cannot be reconstructed here."
                )
            else:
                note = (
                    "RDKit could not reconstruct a valid substructure "
                    "for this bit on this molecule despite the bit "
                    "being marked present (possible aromaticity/"
                    "kekulization edge case in "
                    "Chem.FindAtomEnvironmentOfRadiusN)."
                )

            rows.append({
                "SampleID": sample_id,
                "ErrorType": error_type if error_type else "",
                "Rank": rank,
                "BitIndex": bit_idx,
                "SHAP_value": signed_shap,
                "SelectionCriterion": criterion,
                "FragmentSMILES": fragment_smiles,
                "Note": note
            })

    return pd.DataFrame(rows)
