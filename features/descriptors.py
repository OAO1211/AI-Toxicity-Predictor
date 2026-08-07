# features/descriptors.py

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


def calculate_descriptors(mol):
    """
    Calculate commonly used physicochemical descriptors.

    Descriptor categories
    ---------------------
    Molecular size:
        MolWt
        ExactMolWt
        HeavyAtoms

    Lipophilicity:
        MolLogP

    Polarity:
        TPSA

    Hydrogen bonding:
        HBD
        HBA

    Flexibility:
        RotatableBonds

    Topology:
        RingCount
        AromaticRings
        AliphaticRings

    Shape / Complexity:
        FractionCSP3
        MolMR
        LabuteASA
        BertzCT
    """

    return {
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),

        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),

        "RotatableBonds":
            rdMolDescriptors.CalcNumRotatableBonds(mol),

        "RingCount":
            rdMolDescriptors.CalcNumRings(mol),

        "AromaticRings":
            rdMolDescriptors.CalcNumAromaticRings(mol),

        "AliphaticRings":
            rdMolDescriptors.CalcNumAliphaticRings(mol),

        "HeavyAtoms":
            mol.GetNumHeavyAtoms(),

        "FractionCSP3":
            rdMolDescriptors.CalcFractionCSP3(mol),

        "MolMR":
            Descriptors.MolMR(mol),

        "LabuteASA":
            rdMolDescriptors.CalcLabuteASA(mol),

        "BertzCT":
            Descriptors.BertzCT(mol)
    }


def extract_descriptor_features(
    input_path,
    output_path,
    smiles_col,
    label_col,
    name_col
):
    """
    Generate descriptor table from SMILES.
    """

    df = pd.read_csv(input_path)

    rows = []

    for _, row in df.iterrows():

        mol = Chem.MolFromSmiles(row[smiles_col])

        if mol is None:
            print(
                f"[WARNING] Invalid SMILES skipped: {row[smiles_col]}"
            )
            continue

        desc = calculate_descriptors(mol)

        desc[name_col] = row[name_col]
        desc[label_col] = row[label_col]

        rows.append(desc)

    feature_df = pd.DataFrame(rows)
    # Replace missing descriptor values
    feature_df = feature_df.fillna(0)
    
    cols = [name_col, label_col] + [
        c for c in feature_df.columns
        if c not in [name_col, label_col]
    ]

    feature_df = feature_df[cols]

    feature_df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"[✓] Descriptor features saved to {output_path} - descriptors.py:133")
    print(
        f"[INFO] Samples: {len(feature_df)}"
    )

    print(
        f"[INFO] Descriptors: {feature_df.shape[1]-2}"
    )