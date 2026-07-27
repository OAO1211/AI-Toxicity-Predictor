# explain/shap_to_fragments.py

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

def bit_to_fragment(smiles, bit_indices, radius=3, n_bits=1024):
    """
    將指定 bit_indices 的 ECFP bits 還原為 fragment
    Args:
        smiles (str): SMILES 字串
        bit_indices (list[int]): 需要還原的 bit index
        radius (int): ECFP 半徑
        n_bits (int): bit 長度
    Returns:
        dict: {bit_idx: Mol fragment}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=info)

    fragments = {}
    for bit in bit_indices:
        if bit in info:
            atom_idx, r = info[bit][0]  # 取第一個 atom_idx
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            fragments[bit] = submol
    return fragments

def save_fragments_png(fragments_dict, out_dir, prefix="frag"):
    """
    將 fragment 存成 PNG 圖片
    Args:
        fragments_dict: {bit_idx: Mol}
        out_dir: 存放資料夾
        prefix: 檔名前綴
    """
    os.makedirs(out_dir, exist_ok=True)
    for bit, mol in fragments_dict.items():
        if mol:
            out_path = os.path.join(out_dir, f"{prefix}_bit{bit}.png")
            Draw.MolToFile(mol, out_path)

def shap_to_fragments(shap_df_path, dataset_df_path, top_n=20,
                       radius=3, n_bits=1024, output_dir="fragments"):
    """
    從 SHAP 結果提取 top N bits，還原 fragment 並存圖 + CSV
    Args:
        shap_df_path: training_shap.tsv 或 testing_shap.tsv
        dataset_df_path: 原始資料 CSV
        top_n: top N SHAP bits
        radius, n_bits: ECFP 設定
        output_dir: 輸出資料夾
    Returns:
        frag_df: DataFrame of fragments
    """
    shap_df = pd.read_csv(shap_df_path, sep='\t')
    dataset_df = pd.read_csv(dataset_df_path, sep=',', encoding='latin1')

    # 篩選 ECFP bit
    feature_cols = [c for c in shap_df.columns if c.startswith("ECFP_")]
    mean_abs_shap = shap_df[feature_cols].abs().mean(axis=0)
    top_bits = mean_abs_shap.sort_values(ascending=False).head(top_n).index
    top_bit_indices = [int(b.replace("ECFP_", "")) for b in top_bits]
    print(f"[INFO] Top {top_n} SHAP bits: {top_bit_indices} - shap_to_fragment.py:70")

    all_fragments = []
    for idx, row in dataset_df.iterrows():
        smiles = row["SMILES"]
        name = row.get("CompoundName", f"sample_{idx}")
        frags = bit_to_fragment(smiles, top_bit_indices, radius=radius, n_bits=n_bits)
        for bit, mol in frags.items():
            all_fragments.append({
                "CompoundName": name,
                "SMILES": smiles,
                "BitIndex": bit,
                "FragmentSMILES": Chem.MolToSmiles(mol) if mol else None
            })
        save_fragments_png(frags, os.path.join(output_dir, "pngs"), prefix=name)

    os.makedirs(output_dir, exist_ok=True)
    frag_df = pd.DataFrame(all_fragments)
    frag_df.to_csv(os.path.join(output_dir, "top_shap_fragments.tsv"), sep='\t', index=False)

    print(f"[✓] Fragments saved to {output_dir}/top_shap_fragments.tsv - shap_to_fragment.py:90")
    print(f"[✓] Fragment images saved to {output_dir}/pngs/ - shap_to_fragment.py:91")
    return frag_df
