# features/similarity.py
"""
Structural similarity analysis (item 5 of the Prediction Error Analysis
request).

For each high-confidence FP/FN compound, find its Top-K structurally
most similar compounds in the same dataset using ECFP Tanimoto
similarity. This is meant to answer questions like:

    "Is this misclassified compound actually a near-duplicate /
     close analogue of a compound with the OPPOSITE true label?"
    (i.e. a case where pure structural similarity can't explain the
    DILI phenotype difference)

Uses the same ECFP radius / n_bits as the main feature extraction
(features.ecfp) so that similarity is computed in the same chemical
space the model was trained on.
"""

from typing import List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def _build_fingerprint_table(
    dataset_df: pd.DataFrame,
    id_col: str,
    smiles_col: str = "SMILES",
    radius: int = 3,
    n_bits: int = 1024,
    use_chirality: bool = False
):
    """
    Build ECFP bit-vector fingerprints for every valid SMILES in
    dataset_df.

    Returns
    -------
    ids : list[str]            compound ids (same order as fps)
    fps : list[ExplicitBitVect] RDKit fingerprints (for BulkTanimoto)
    id_to_pos : dict[str, int]  id -> position in ids/fps
    n_failed : int              number of SMILES RDKit could not parse
    """

    ids = []
    fps = []
    n_failed = 0

    for _, row in dataset_df.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None

        if mol is None:
            n_failed += 1
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useFeatures=False,
            useChirality=use_chirality
        )

        ids.append(str(row[id_col]))
        fps.append(fp)

    id_to_pos = {cid: i for i, cid in enumerate(ids)}

    return ids, fps, id_to_pos, n_failed


def find_nearest_neighbors_for_ids(
    dataset_df: pd.DataFrame,
    target_ids: List[str],
    id_col: str,
    smiles_col: str = "SMILES",
    label_col: Optional[str] = None,
    radius: int = 3,
    n_bits: int = 1024,
    top_k: int = 5,
    use_chirality: bool = False
) -> pd.DataFrame:
    """
    For every id in target_ids, find its Top-K nearest neighbors (by
    ECFP Tanimoto similarity) among all OTHER compounds in dataset_df.

    Parameters
    ----------
    dataset_df : DataFrame containing at least [id_col, smiles_col]
                 (and optionally label_col) for the WHOLE dataset —
                 the pool of possible neighbors, not just the errors.
    target_ids : compound ids (matching id_col values) to search
                 neighbors for. Usually the high-confidence FP/FN ids.
    id_col      : column in dataset_df that matches the SampleID used
                  throughout the pipeline (LabelCompoundName).
    label_col   : optional, if given, the neighbor's true label is
                  included in the output for easier interpretation.

    Returns
    -------
    DataFrame with columns:
        SampleID, Rank, NeighborID, Similarity,
        NeighborTrueLabel (only if label_col is given)

    Compounds in target_ids whose SMILES could not be parsed by RDKit
    (or that are not present in dataset_df at all) are skipped with a
    printed warning, rather than silently dropped without a trace.
    """

    ids, fps, id_to_pos, n_failed = _build_fingerprint_table(
        dataset_df,
        id_col=id_col,
        smiles_col=smiles_col,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality
    )

    if n_failed > 0:
        print(
            f"[WARN] similarity.find_nearest_neighbors_for_ids: "
            f"{n_failed} SMILES in dataset_df could not be parsed by "
            f"RDKit and were excluded from the similarity search space."
        )

    label_lookup = None
    if label_col is not None and label_col in dataset_df.columns:
        label_lookup = (
            dataset_df.set_index(dataset_df[id_col].astype(str))[label_col]
            .to_dict()
        )

    rows = []

    for target_id in target_ids:

        target_id = str(target_id)

        if target_id not in id_to_pos:
            print(
                f"[WARN] similarity: SampleID '{target_id}' not found "
                f"(or SMILES unparsable) in dataset_df — skipped."
            )
            continue

        query_pos = id_to_pos[target_id]
        query_fp = fps[query_pos]
        query_label = (
            label_lookup.get(target_id) if label_lookup is not None
            else None
        )

        sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps)

        # Exclude the compound itself, then take the top_k highest
        # similarity scores.
        ranked = sorted(
            (
                (ids[i], s)
                for i, s in enumerate(sims)
                if i != query_pos
            ),
            key=lambda t: t[1],
            reverse=True
        )[:top_k]

        for rank, (neighbor_id, sim) in enumerate(ranked, 1):

            row = {
                "SampleID": target_id,
                "Rank": rank,
                "NeighborID": neighbor_id,
                "Similarity": round(float(sim), 4)
            }

            if label_lookup is not None:
                neighbor_label = label_lookup.get(neighbor_id)
                row["NeighborTrueLabel"] = neighbor_label
                row["QueryTrueLabel"] = query_label
                row["LabelMismatch"] = (
                    (neighbor_label != query_label)
                    if (query_label is not None and neighbor_label is not None)
                    else None
                )

            rows.append(row)

    return pd.DataFrame(rows)


def find_neighbors_above_threshold(
    dataset_df: pd.DataFrame,
    target_ids: List[str],
    id_col: str,
    smiles_col: str = "SMILES",
    label_col: Optional[str] = None,
    radius: int = 3,
    n_bits: int = 1024,
    threshold: float = 0.7,
    use_chirality: bool = False
) -> pd.DataFrame:
    """
    For every id in target_ids, find ALL other compounds in dataset_df
    with Tanimoto similarity >= threshold (not capped at Top-K).

    This is what "structurally near-identical but different label" is
    actually asking for: Top-5 can miss a 6th, 7th, 8th... neighbor
    that also clears the similarity bar, which would understate how
    many close structural analogues disagree on the true label.

    Returns
    -------
    DataFrame with columns:
        SampleID, NeighborID, Similarity, QueryTrueLabel,
        NeighborTrueLabel, LabelMismatch
    (label columns only present if label_col is given and found).
    Empty DataFrame (with these columns) if a target_id has no
    neighbor clearing the threshold.
    """

    ids, fps, id_to_pos, n_failed = _build_fingerprint_table(
        dataset_df,
        id_col=id_col,
        smiles_col=smiles_col,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality
    )

    if n_failed > 0:
        print(
            f"[WARN] similarity.find_neighbors_above_threshold: "
            f"{n_failed} SMILES in dataset_df could not be parsed by "
            f"RDKit and were excluded from the similarity search space."
        )

    label_lookup = None
    if label_col is not None and label_col in dataset_df.columns:
        label_lookup = (
            dataset_df.set_index(dataset_df[id_col].astype(str))[label_col]
            .to_dict()
        )

    rows = []

    for target_id in target_ids:

        target_id = str(target_id)

        if target_id not in id_to_pos:
            print(
                f"[WARN] similarity: SampleID '{target_id}' not found "
                f"(or SMILES unparsable) in dataset_df — skipped."
            )
            continue

        query_pos = id_to_pos[target_id]
        query_fp = fps[query_pos]
        query_label = (
            label_lookup.get(target_id) if label_lookup is not None
            else None
        )

        sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps)

        for i, sim in enumerate(sims):

            if i == query_pos or sim < threshold:
                continue

            neighbor_id = ids[i]

            row = {
                "SampleID": target_id,
                "NeighborID": neighbor_id,
                "Similarity": round(float(sim), 4)
            }

            if label_lookup is not None:
                neighbor_label = label_lookup.get(neighbor_id)
                row["QueryTrueLabel"] = query_label
                row["NeighborTrueLabel"] = neighbor_label
                row["LabelMismatch"] = (
                    (neighbor_label != query_label)
                    if (query_label is not None and neighbor_label is not None)
                    else None
                )

            rows.append(row)

    columns = ["SampleID", "NeighborID", "Similarity"]
    if label_lookup is not None:
        columns += ["QueryTrueLabel", "NeighborTrueLabel", "LabelMismatch"]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)[columns]


def summarize_label_disagreement(
    neighbors_above_threshold_df: pd.DataFrame,
    target_ids: List[str],
    threshold: float
) -> pd.DataFrame:
    """
    Collapses find_neighbors_above_threshold()'s output into one row
    per compound:

        SampleID, SimilarityThreshold, NeighborsAboveThreshold,
        MismatchedNeighbors, MismatchFraction

    Compounds with zero neighbors clearing the threshold still get a
    row (with NeighborsAboveThreshold=0, MismatchFraction=NaN) so the
    absence of close analogues is visible rather than silently missing.
    """

    rows = []

    has_labels = "LabelMismatch" in neighbors_above_threshold_df.columns

    for target_id in target_ids:

        target_id = str(target_id)

        sub = neighbors_above_threshold_df[
            neighbors_above_threshold_df["SampleID"] == target_id
        ]

        n_neighbors = len(sub)

        if has_labels and n_neighbors > 0:
            mismatched = int(sub["LabelMismatch"].fillna(False).sum())
            mismatch_fraction = round(mismatched / n_neighbors, 4)
        else:
            mismatched = 0 if n_neighbors > 0 else None
            mismatch_fraction = None

        rows.append({
            "SampleID": target_id,
            "SimilarityThreshold": threshold,
            "NeighborsAboveThreshold": n_neighbors,
            "MismatchedNeighbors": mismatched,
            "MismatchFraction": mismatch_fraction
        })

    return pd.DataFrame(rows)
