# features/scaffold.py
"""
Murcko scaffold utilities.

Directly answers the research direction's "相似 scaffold → 不同 label"
question and the reviewer's Priority-3 follow-up: Tanimoto similarity
on the full ECFP fingerprint can be high even when two molecules don't
share the same core scaffold (e.g. similar decorations, different
ring system), and conversely two molecules can share an identical
Murcko scaffold while differing enough elsewhere to pull their overall
Tanimoto score down. Reporting the scaffold explicitly (not just the
Tanimoto score) lets you tell "same scaffold, different DILI label"
cases apart from "merely similar, different scaffold" cases.
"""

from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def compute_murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    """
    Returns the canonical SMILES of the Murcko scaffold (ring systems +
    linkers, side chains stripped) for a molecule, or None if the
    SMILES is unparsable or the molecule has no ring system (fully
    acyclic — Murcko scaffold is then empty).
    """

    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None

    if mol is None:
        return None

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None

    return Chem.MolToSmiles(scaffold)


def build_scaffold_lookup(
    dataset_df: pd.DataFrame,
    id_col: str,
    smiles_col: str = "SMILES"
) -> Dict[str, Optional[str]]:
    """
    Returns {SampleID: MurckoScaffoldSMILES or None} for every compound
    in dataset_df.
    """

    lookup = {}

    for _, row in dataset_df.iterrows():
        sample_id = str(row[id_col])
        lookup[sample_id] = compute_murcko_scaffold_smiles(row[smiles_col])

    return lookup


def annotate_scaffolds(
    df: pd.DataFrame,
    scaffold_lookup: Dict[str, Optional[str]],
    id_column: str = "SampleID",
    out_column: str = "MurckoScaffold"
) -> pd.DataFrame:
    """
    Adds a MurckoScaffold column to df by looking up id_column's values
    in scaffold_lookup. Does not mutate df in place.
    """

    out = df.copy()
    out[out_column] = out[id_column].astype(str).map(scaffold_lookup)

    return out


def annotate_scaffold_match(
    neighbors_df: pd.DataFrame,
    scaffold_lookup: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    For a neighbors table with SampleID (the error compound) and
    NeighborID columns (as produced by features.similarity), adds:

        QueryScaffold, NeighborScaffold, ScaffoldMatch

    ScaffoldMatch is True only when both scaffolds are resolvable and
    identical (canonical-SMILES equal) — i.e. same core ring system,
    not just "structurally similar" by Tanimoto. None when either
    scaffold could not be computed (e.g. unparsable SMILES, or a fully
    acyclic molecule with no ring system to compare).
    """

    out = neighbors_df.copy()

    out["QueryScaffold"] = out["SampleID"].astype(str).map(scaffold_lookup)
    out["NeighborScaffold"] = out["NeighborID"].astype(str).map(scaffold_lookup)

    def _match(row):
        q, n = row["QueryScaffold"], row["NeighborScaffold"]
        if q is None or n is None:
            return None
        return q == n

    out["ScaffoldMatch"] = out.apply(_match, axis=1)

    return out
