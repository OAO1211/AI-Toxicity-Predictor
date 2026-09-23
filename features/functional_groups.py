# features/functional_groups.py
"""
Curated functional group ("structural alert") counts, using RDKit's
SMARTS-based rdkit.Chem.Fragments module.

Only a curated subset is used — not all ~85 fr_* functions RDKit
ships — chosen for relevance to the reactive-metabolite / idiosyncratic
DILI literature (anilines, nitro groups, epoxides, thiophenes,
hydrazines, halogens, quinone-forming carbonyls, etc. are recurring
structural alerts in that literature). This keeps the property
comparison table (evaluation/property_comparison.py) focused on
groups with a plausible mechanistic story, rather than running 85
simultaneous hypothesis tests with no correction for multiple
comparisons.

If you want the full RDKit Fragments set, that's
`rdkit.Chem.Fragments.__dict__` filtered to callables starting with
"fr_" — swap FUNCTIONAL_GROUPS below for that list.
"""

from typing import Dict

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Fragments


# RDKit fr_* name -> readable label used in output columns
# (columns are prefixed "FG_" so they're easy to tell apart from the
# physicochemical descriptors in the same comparison table).
FUNCTIONAL_GROUPS = {
    "fr_aniline": "Aniline",
    "fr_nitro": "Nitro",
    "fr_nitro_arom": "AromaticNitro",
    "fr_epoxide": "Epoxide",
    "fr_thiophene": "Thiophene",
    "fr_hdrzine": "Hydrazine",
    "fr_hdrzone": "Hydrazone",
    "fr_azide": "Azide",
    "fr_diazo": "Diazo",
    "fr_halogen": "Halogen",
    "fr_alkyl_halide": "AlkylHalide",
    "fr_quatN": "QuaternaryN",
    "fr_amide": "Amide",
    "fr_aldehyde": "Aldehyde",
    "fr_ketone": "Ketone",
    "fr_phenol": "Phenol",
    "fr_furan": "Furan",
    "fr_imidazole": "Imidazole",
    "fr_sulfonamide": "Sulfonamide",
    "fr_sulfide": "Sulfide",
    "fr_urea": "Urea",
    "fr_guanido": "Guanidino",
    "fr_para_hydroxylation": "ParaHydroxylationSite",
    "fr_benzene": "Benzene",
    "fr_C_O": "Carbonyl",
}


def calculate_functional_groups(mol) -> Dict[str, int]:
    """
    Returns {"FG_<Label>": count} for every group in FUNCTIONAL_GROUPS.

    Uses getattr(..., default 0-returning lambda) rather than a direct
    attribute access so this doesn't hard-crash if a given RDKit
    version ever renames/drops one of these fr_* functions — it will
    just report 0 for that group and the rest still work.
    """

    counts = {}

    for rdkit_name, label in FUNCTIONAL_GROUPS.items():

        fn = getattr(Fragments, rdkit_name, None)

        counts[f"FG_{label}"] = int(fn(mol)) if fn is not None else 0

    return counts


def compute_functional_groups_for_ids(
    dataset_df: pd.DataFrame,
    id_col: str,
    smiles_col: str,
    sample_ids
) -> pd.DataFrame:
    """
    Computes the curated functional-group counts for each id in
    sample_ids, looking up SMILES from dataset_df.

    Returns a DataFrame with columns [SampleID, FG_<Label>, ...].
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
                f"[WARN] compute_functional_groups_for_ids: SampleID "
                f"'{sample_id}' has no SMILES — skipped."
            )
            continue

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            print(
                f"[WARN] compute_functional_groups_for_ids: SampleID "
                f"'{sample_id}' has an unparsable SMILES — skipped."
            )
            continue

        row = calculate_functional_groups(mol)
        row["SampleID"] = sample_id

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["SampleID"])

    fg_df = pd.DataFrame(rows)

    cols = ["SampleID"] + [c for c in fg_df.columns if c != "SampleID"]

    return fg_df[cols]


def present_groups_for_row(row: pd.Series) -> str:
    """
    Given a row that has FG_<Label> count columns, returns a
    comma-separated list of the group labels with count > 0 — a
    quick human-readable "which structural alerts does this compound
    have" summary for a single compound (used in the per-error-compound
    functional group report).
    """

    present = [
        col.replace("FG_", "")
        for col in row.index
        if col.startswith("FG_") and pd.notna(row[col]) and row[col] > 0
    ]

    return ", ".join(present)
