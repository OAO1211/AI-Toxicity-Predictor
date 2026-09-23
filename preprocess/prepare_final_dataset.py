"""
Prepare the frozen DILIrank-derived datasets used by the final v4 run.

Design choices
--------------
Primary cohort membership is NOT redefined here. The input CSV is the
450-compound, structure-available DILIrank vMost/vNo cohort. This script:

1. regenerates the binary label directly from ``vDILIConcern``;
2. validates every source SMILES with RDKit;
3. writes canonical isomeric SMILES for QC only (the model still uses the
   original DILIrank SMILES representation);
4. creates objective structure-audit flags;
5. creates a predeclared sensitivity subset:
       single component AND metal-free AND MolWt <= 1000 Da.

No blind salt stripping, largest-fragment selection, neutralization,
tautomer canonicalization, or parent-molecule replacement is performed.
Those operations can collapse pharmacologically distinct products and are
therefore deliberately kept out of the primary preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from config import (
    CURATED_DATA_DIR,
    DILI_LABEL_MAP,
    LABEL_COL,
    NAME_COL,
    SENSITIVITY_MAX_MW,
    SMILES_COL,
    SOURCE_DILI_COL,
)


# Broad medicinal-chemistry metal/metalloid flag used only for an exploratory
# sensitivity subset / applicability-domain audit. Carbon, N, O, P, S,
# halogens and other usual organic elements are not flagged.
METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
}


def _has_metal(mol: Chem.Mol) -> bool:
    return any(
        atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS
        for atom in mol.GetAtoms()
    )


def _num_fragments(mol: Chem.Mol) -> int:
    return len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))


def _read_csv(path: str) -> pd.DataFrame:
    # The legacy project data are latin1-compatible. Keep the same loader to
    # avoid silently changing identifiers while preparing the final cohort.
    return pd.read_csv(path, encoding="latin1")


def prepare_final_datasets(
    input_path: str,
    output_dir: str = CURATED_DATA_DIR,
    max_sensitivity_mw: float = SENSITIVITY_MAX_MW,
) -> Dict[str, str]:
    """
    Validate and freeze the primary 450-compound dataset plus a sensitivity
    subset. Returns paths to the generated files.
    """

    df = _read_csv(input_path).copy()

    required = {NAME_COL, SMILES_COL, SOURCE_DILI_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------
    # 1. Rebuild labels from the source DILIrank category.
    # ------------------------------------------------------------
    unexpected = sorted(
        set(df[SOURCE_DILI_COL].dropna().astype(str))
        .difference(DILI_LABEL_MAP)
    )
    if unexpected:
        raise ValueError(
            "Final binary dataset should contain only vMost/vNo DILIrank "
            f"categories; found unexpected values: {unexpected}"
        )

    old_label = df[LABEL_COL].copy() if LABEL_COL in df.columns else None
    df[LABEL_COL] = df[SOURCE_DILI_COL].map(DILI_LABEL_MAP)

    if df[LABEL_COL].isna().any():
        bad = df.loc[df[LABEL_COL].isna(), [NAME_COL, SOURCE_DILI_COL]]
        raise ValueError(
            "Some rows could not be mapped to binary labels:\n"
            + bad.to_string(index=False)
        )

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if old_label is not None:
        old_numeric = pd.to_numeric(old_label, errors="coerce")
        mismatched = old_numeric.notna() & (old_numeric != df[LABEL_COL])
        if mismatched.any():
            print("[QC] Corrected legacy label mismatches:")
            cols = [NAME_COL, SOURCE_DILI_COL]
            tmp = df.loc[mismatched, cols].copy()
            tmp["OldLabel"] = old_numeric[mismatched].astype(int).values
            tmp["FinalLabel"] = df.loc[mismatched, LABEL_COL].values
            print(tmp.to_string(index=False))

    # ------------------------------------------------------------
    # 2. Structure validation / audit. Source SMILES are preserved.
    # ------------------------------------------------------------
    audit_rows = []
    canonical_seen: Dict[str, list] = {}

    for idx, row in df.iterrows():
        sample_id = str(row[NAME_COL])
        smiles = str(row[SMILES_COL])
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(
                f"RDKit failed to parse final-cohort SMILES for {sample_id}: "
                f"{smiles}"
            )

        canonical_iso = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )
        canonical_noniso = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False
        )
        n_frag = _num_fragments(mol)
        has_metal = _has_metal(mol)
        mw = float(Descriptors.MolWt(mol))

        conventional = (
            n_frag == 1
            and not has_metal
            and mw <= max_sensitivity_mw
        )

        canonical_seen.setdefault(canonical_iso, []).append(sample_id)

        audit_rows.append({
            "Row": idx,
            "SampleID": sample_id,
            "vDILIConcern": row[SOURCE_DILI_COL],
            "label": int(row[LABEL_COL]),
            "OriginalSMILES": smiles,
            "CanonicalIsomericSMILES": canonical_iso,
            "CanonicalNonIsomericSMILES": canonical_noniso,
            "RDKitValid": True,
            "NumFragments": n_frag,
            "HasMetal": has_metal,
            "MolWt": mw,
            "MW_gt_1000": mw > 1000.0,
            "ConventionalSensitivitySubset": conventional,
        })

    audit_df = pd.DataFrame(audit_rows)

    duplicate_groups = {
        smi: ids for smi, ids in canonical_seen.items() if len(ids) > 1
    }
    if duplicate_groups:
        print(
            "[WARN] Canonical-isomeric duplicate structures found. They are "
            "NOT silently removed; inspect the audit before a formal run:"
        )
        for smi, ids in duplicate_groups.items():
            print(f"  {ids}: {smi}")

    df["CanonicalIsomericSMILES_QC"] = audit_df[
        "CanonicalIsomericSMILES"
    ].values
    df["NumFragments_QC"] = audit_df["NumFragments"].values
    df["HasMetal_QC"] = audit_df["HasMetal"].values
    df["MolWt_QC"] = audit_df["MolWt"].values
    df["ConventionalSensitivitySubset"] = audit_df[
        "ConventionalSensitivitySubset"
    ].values

    sensitivity_df = df.loc[
        df["ConventionalSensitivitySubset"]
    ].copy().reset_index(drop=True)

    # ------------------------------------------------------------
    # 3. Freeze outputs.
    # ------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(input_path).stem
    primary_path = os.path.join(output_dir, f"{stem}_primary.csv")
    sensitivity_path = os.path.join(
        output_dir, f"{stem}_sensitivity_conventional.csv"
    )
    audit_path = os.path.join(output_dir, f"{stem}_structure_audit.tsv")
    summary_path = os.path.join(output_dir, f"{stem}_dataset_summary.csv")

    # Keep the original DILIrank SMILES in the SMILES column used by the model.
    df.to_csv(primary_path, index=False, encoding="latin1")
    sensitivity_df.to_csv(
        sensitivity_path, index=False, encoding="latin1"
    )
    audit_df.to_csv(audit_path, sep="\t", index=False)

    summary_rows = [
        ["Primary_N", len(df)],
        ["Primary_DILI", int((df[LABEL_COL] == 1).sum())],
        ["Primary_nonDILI", int((df[LABEL_COL] == 0).sum())],
        ["RDKit_valid", int(audit_df["RDKitValid"].sum())],
        ["Multi_component", int((audit_df["NumFragments"] > 1).sum())],
        ["Metal_containing", int(audit_df["HasMetal"].sum())],
        ["MW_gt_1000", int(audit_df["MW_gt_1000"].sum())],
        ["Sensitivity_N", len(sensitivity_df)],
        ["Sensitivity_DILI", int((sensitivity_df[LABEL_COL] == 1).sum())],
        ["Sensitivity_nonDILI", int((sensitivity_df[LABEL_COL] == 0).sum())],
        ["Canonical_isomeric_duplicate_groups", len(duplicate_groups)],
    ]
    pd.DataFrame(summary_rows, columns=["Metric", "Count"]).to_csv(
        summary_path, index=False
    )

    print("\n[FINAL DATASET QC]")
    print(f"Primary: N={len(df)} | DILI={(df[LABEL_COL] == 1).sum()} | "
          f"non-DILI={(df[LABEL_COL] == 0).sum()}")
    print(
        f"Structure flags: multi-component={(audit_df['NumFragments'] > 1).sum()}, "
        f"metal={audit_df['HasMetal'].sum()}, MW>1000={audit_df['MW_gt_1000'].sum()}"
    )
    print(
        f"Sensitivity: N={len(sensitivity_df)} | "
        f"DILI={(sensitivity_df[LABEL_COL] == 1).sum()} | "
        f"non-DILI={(sensitivity_df[LABEL_COL] == 0).sum()}"
    )
    print("No salt stripping / parent replacement was performed.")
    print(f"Primary saved: {primary_path}")
    print(f"Sensitivity saved: {sensitivity_path}")
    print(f"Audit saved: {audit_path}")

    return {
        "primary": primary_path,
        "sensitivity": sensitivity_path,
        "audit": audit_path,
        "summary": summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze final DILIrank-derived primary/sensitivity datasets."
    )
    parser.add_argument("input_csv", help="450-compound source CSV in data/raw/")
    parser.add_argument(
        "--output-dir", default=CURATED_DATA_DIR,
        help="Defaults to data/curated/."
    )
    parser.add_argument(
        "--max-sensitivity-mw", type=float, default=SENSITIVITY_MAX_MW,
        help="MW threshold for the exploratory conventional-structure subset."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_final_datasets(
        input_path=args.input_csv,
        output_dir=args.output_dir,
        max_sensitivity_mw=args.max_sensitivity_mw,
    )
