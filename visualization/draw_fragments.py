# visualization/draw_fragments.py

import os
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

def draw_fragment_image(mol, out_path, size=(300, 300)):
    """
    將 RDKit Mol 畫圖並存成 PNG
    """
    if mol is None:
        return
    img = Draw.MolToImage(mol, size=size)
    img.save(out_path)


def draw_fragments_from_shap(fragments_df_path, output_dir="fragment_images"):
    """
    fragments_df_path: top_shap_fragments.tsv (含 BitIndex / FragmentSMILES / CompoundName)
    output_dir: 存圖的資料夾
    """
    os.makedirs(output_dir, exist_ok=True)
    frag_df = pd.read_csv(fragments_df_path, sep="\t")
    
    for idx, row in frag_df.iterrows():
        frag_smiles = row["FragmentSMILES"]
        if pd.isna(frag_smiles) or frag_smiles.strip() == "":
            continue
        mol = Chem.MolFromSmiles(frag_smiles)
        compound_name = row.get("CompoundName", f"sample_{idx}")
        bit_idx = row.get("BitIndex", idx)
        out_path = os.path.join(output_dir, f"{compound_name}_bit{bit_idx}.png")
        draw_fragment_image(mol, out_path)


if __name__ == "__main__":
    fragments_df_path = "results/fragments/top_shap_fragments.tsv"
    draw_fragments_from_shap(fragments_df_path, output_dir="results/fragments/pngs")
    print("[✓] Fragment images drawn and saved. - draw_fragments.py:40")
