# feature_extraction/ecfp.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_ecfp(smiles, radius=3, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useFeatures=False
    )
    return np.array(fp, dtype=int)


def extract_ecfp_features(
    input_path,
    output_path,
    smiles_col="SMILES",
    label_col="label",
    name_col="CompoundName",
    radius=3,
    n_bits=1024
):
    df = pd.read_csv(input_path, sep=',', encoding='latin1')

    records = []
    failed = []

    for _, row in df.iterrows():
        smiles = row[smiles_col]
        fp = smiles_to_ecfp(smiles, radius, n_bits)

        if fp is None:
            failed.append(row.get(name_col, "UNKNOWN"))
            continue

        record = {
            "LabelCompoundName": row[name_col],
            "SMILES": smiles,
            "label": row[label_col],
        }

        record.update({f"ECFP_{i}": fp[i] for i in range(n_bits)})
        records.append(record)

    feature_df = pd.DataFrame(records)
    feature_df.to_csv(output_path, sep="\t", index=False)

    print(f"[✓] ECFP features saved to {output_path} - ecfp.py:57")
    print(f"[✓] Shape: {feature_df.shape} - ecfp.py:58")

    if failed:
        print(f"[WARN] {len(failed)} SMILES failed ECFP generation - ecfp.py:61")

    return feature_df
