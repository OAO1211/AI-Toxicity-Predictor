import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem


def filter_valid_smiles(input_path, output_path, smiles_col="SMILES"):
    """
    讀取原始資料，只保留 RDKit 能成功解析（Chem.MolFromSmiles 不是 None）
    的 SMILES，寫出一份「乾淨版」的中繼 CSV。

    目的：ECFP 和 Descriptors 是分別獨立呼叫 RDKit 解析同一份原始資料，
    理論上兩邊會篩掉同一批（極少數）解析失敗的分子，但這是「理論上」，
    不該只靠事後比對兩邊的 ids 是否一致來把關。這裡先用同一套過濾邏輯
    產生一份共同的「乾淨資料集」，讓 ECFP-only、Descriptors-only、
    Combined 三種特徵集合從一開始就保證是在完全相同的一批分子上訓練
    與比較，而不是可能各自漏掉不同的分子。

    回傳 (n_total, n_valid, n_dropped)。
    """

    df = pd.read_csv(input_path, sep=",", encoding="latin1")

    n_total = len(df)

    is_valid = df[smiles_col].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )

    n_valid = int(is_valid.sum())
    n_dropped = n_total - n_valid

    if n_dropped > 0:
        dropped_names = df.loc[~is_valid, smiles_col].tolist()
        print(
            f"[WARNING] filter_valid_smiles: {n_dropped}/{n_total} 筆 SMILES "
            f"無法被 RDKit 解析，已從 ECFP / Descriptors / Combined 的資料"
            f"來源中一併排除，確保三者使用完全相同的樣本："
        )
        for s in dropped_names:
            print(f"{s} - clean_data.py:41")

    df_clean = df.loc[is_valid].reset_index(drop=True)

    df_clean.to_csv(output_path, index=False, encoding="latin1")

    return n_total, n_valid, n_dropped


def fit_encoders(X):
    """
    只在 training data 上 fit encoder
    """
    encoders = {}
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in non_numeric:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        encoders[col] = le

    return encoders


def transform_with_encoders(X, encoders):
    """
    使用已 fit 的 encoder 進行 transform
    """
    X = X.copy()
    for col, le in encoders.items():
        X[col] = le.transform(X[col].astype(str))
    return X


def split_xy(df, label_col, sample_col=None, drop_cols=("SMILES",)):
    ids = (
        df[sample_col].astype(str).values
        if sample_col
        else np.array([f"sample_{i}" for i in range(len(df))])
    )

    cols_to_drop = [label_col]
    if sample_col:
        cols_to_drop.append(sample_col)
    for col in drop_cols:
        if col in df.columns and col not in cols_to_drop:
            cols_to_drop.append(col)

    X = df.drop(columns=cols_to_drop)
    y = df[label_col]

    return ids, X, y