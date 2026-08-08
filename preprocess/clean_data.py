import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
