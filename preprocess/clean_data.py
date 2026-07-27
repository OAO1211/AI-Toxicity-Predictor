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


def split_xy(df, label_col, sample_col=None):
    ids = (
        df[sample_col].astype(str).values
        if sample_col
        else np.array([f"sample_{i}" for i in range(len(df))])
    )

    X = df.drop(columns=[label_col] + ([sample_col] if sample_col else []))
    y = df[label_col]

    return ids, X, y
