# features/bit_mapping.py
from typing import List, Dict
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import os
import pandas as pd
def bit_to_fragment(smiles: str, bit_indices: List[int], radius: int = 3, n_bits: int = 1024) -> Dict[int, Chem.Mol]:
    """
    將指定 bit_indices 的 ECFP bits 還原為 substructure fragment

    Args:
        smiles (str): SMILES 字串
        bit_indices (List[int]): 需要還原的 bit index (0~n_bits-1)
        radius (int): ECFP 半徑
        n_bits (int): ECFP bits 長度
    Returns:
        Dict[int, Chem.Mol]: {bit_idx: RDKit Mol (fragment)}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=info
    )

    fragments = {}
    for bit in bit_indices:
        if bit in info:
            for atom_idx, r in info[bit]:
                try:
                    submol = Chem.PathToSubmol(mol, Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx))
                except Exception:
                    # RDKit 在還原子結構時偶爾會因為不合法的
                    # aromaticity/kekulization 而丟出例外，
                    # 這裡不讓整個流程中斷，用 None 標記「這個 bit
                    # 在這個分子上還原失敗」
                    submol = None
                fragments[bit] = submol
                break
    return fragments


def save_fragments_as_png(fragments_dict: Dict[int, Chem.Mol], out_dir: str, prefix: str = "frag"):
    """
    Save fragments as PNG images

    Args:
        fragments_dict: {bit_idx: Mol}
        out_dir: output folder
        prefix: file name prefix
    """
    os.makedirs(out_dir, exist_ok=True)
    for bit, mol in fragments_dict.items():
        if mol is None:
            continue
        out_path = os.path.join(out_dir, f"{prefix}_bit{bit}.png")
        Draw.MolToFile(mol, out_path)


def extract_top_shap_fragments(
    shap_df_path: str,
    dataset_df_path: str,
    top_n: int = 20,
    output_dir: str = "fragments",
    radius: int = 3,
    n_bits: int = 1024
):
    """
    從 SHAP 結果提取 top N bits，還原 fragment 並存圖 + CSV

    Args:
        shap_df_path: training_shap.tsv 或 testing_shap.tsv
        dataset_df_path: 原始含 SMILES 的資料集
        top_n: 取平均 abs SHAP 前 N 高 bits
        output_dir: 儲存資料夾
        radius: ECFP radius
        n_bits: ECFP bits
    """
    shap_df = pd.read_csv(shap_df_path, sep='\t')
    dataset_df = pd.read_csv(dataset_df_path, sep=',', encoding='latin1')

    # 只保留 feature columns
    feature_cols = [c for c in shap_df.columns if c.startswith("ECFP_")]

    # 計算 mean abs SHAP
    mean_abs_shap = shap_df[feature_cols].abs().mean(axis=0)
    top_bits = mean_abs_shap.sort_values(ascending=False).head(top_n).index
    top_bit_indices = [int(b.replace("ECFP_", "")) for b in top_bits]

    print(f"[INFO] Top {top_n} SHAP bits: {top_bit_indices} - bit_mapping.py:92")

    # bit -> fragment
    all_fragments = []
    # 追蹤每個 top bit 是否曾經在資料集中還原出「合法」的 fragment
    # （FragmentSMILES 不是 None），用來判斷要不要在最後補一列警告
    bit_has_valid_fragment = {bit: False for bit in top_bit_indices}

    for idx, row in dataset_df.iterrows():
        smiles = row["SMILES"]
        name = row.get("CompoundName", f"sample_{idx}")
        frags = bit_to_fragment(smiles, top_bit_indices, radius=radius, n_bits=n_bits)
        for bit, mol in frags.items():
            fragment_smiles = Chem.MolToSmiles(mol) if mol else None
            if fragment_smiles is not None:
                bit_has_valid_fragment[bit] = True
            all_fragments.append({
                "CompoundName": name,
                "SMILES": smiles,
                "BitIndex": bit,
                "FragmentSMILES": fragment_smiles,
                "Note": "" if fragment_smiles is not None else
                        "RDKit 無法從這個分子還原出合法子結構"
            })
        # 存圖
        save_fragments_as_png(frags, os.path.join(output_dir, "pngs"), prefix=name)

    # 對完全沒有還原出任何合法 fragment 的 top bit，明確補一列，
    # 而不是讓它從輸出裡悄悄消失。這種情況常見於 ECFP bit collision：
    # 不同子結構被雜湊進同一個 bit index，導致模型認為這個 bit 很重要
    # (mean |SHAP| 高)，但反推不出一個穩定、單一的化學子結構。
    for bit in top_bit_indices:
        if not bit_has_valid_fragment[bit]:
            all_fragments.append({
                "CompoundName": None,
                "SMILES": None,
                "BitIndex": bit,
                "FragmentSMILES": None,
                "Note": (
                    "在整個資料集裡都沒有還原出合法的子結構"
                    "（可能是 ECFP bit collision：多個不同子結構"
                    "共用同一個 bit index，建議在報告中列為 "
                    "unresolved / 或考慮增加 n_bits 降低碰撞率）"
                )
            })
            print(
                f"[WARN] Bit {bit} 在整個資料集裡沒有還原出合法子結構"
                f" - bit_mapping.py"
            )

    frag_df = pd.DataFrame(all_fragments)
    os.makedirs(output_dir, exist_ok=True)
    frag_df.to_csv(os.path.join(output_dir, "top_shap_fragments.tsv"), sep='\t', index=False)

    print(f"[✓] Fragments saved to {output_dir}/top_shap_fragments.tsv - bit_mapping.py:146")
    print(f"[✓] Fragment images saved to {output_dir}/pngs/ - bit_mapping.py:147")
