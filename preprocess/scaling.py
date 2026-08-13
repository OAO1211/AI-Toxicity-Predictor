# preprocess/scaling.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_descriptor_scaler_transform(feature_columns):
    """
    回傳一個 feature_transform_fn，符合
    evaluation.cross_validation.run_5fold_cv 的 feature_transform_fn 介面：

        feature_transform_fn(X_train, X_test) -> (X_train_scaled, X_test_scaled)

    只對「不是以 ECFP_ 開頭」的欄位（也就是 15 個 physicochemical
    descriptors）做 StandardScaler，ECFP bits（0/1 二元特徵）保持原樣不動。

    這個函式回傳的是一個「工廠」——真正的 StandardScaler 是在
    run_5fold_cv 每個 outer fold 呼叫這個閉包時才建立、才 fit，
    而且只用該 fold 的 X_train 去 fit，X_test 只做 transform，
    不會有 test fold 的分布資訊（mean/std）洩漏進前處理步驟。

    feature_columns: X 的完整欄位順序（例如 X.columns），
    用來判斷哪些欄位是 descriptors、哪些是 ECFP bits。
    """

    descriptor_cols = [
        c for c in feature_columns
        if not c.startswith("ECFP_")
    ]

    ecfp_cols = [
        c for c in feature_columns
        if c.startswith("ECFP_")
    ]

    if not descriptor_cols:
        raise ValueError(
            "make_descriptor_scaler_transform: 沒有找到任何非 ECFP_ 開頭的欄位，"
            "請確認傳入的 feature_columns 是否為 Combined 特徵集合的欄位。"
        )

    def _transform(X_train, X_test):

        scaler = StandardScaler()

        X_train_desc_scaled = pd.DataFrame(
            scaler.fit_transform(X_train[descriptor_cols]),
            columns=descriptor_cols,
            index=X_train.index
        )

        X_test_desc_scaled = pd.DataFrame(
            scaler.transform(X_test[descriptor_cols]),
            columns=descriptor_cols,
            index=X_test.index
        )

        # ECFP bits 保持原樣，只替換 descriptor 欄位，
        # 並維持跟輸入一致的欄位順序（feature_columns）
        X_train_out = pd.concat(
            [X_train[ecfp_cols], X_train_desc_scaled],
            axis=1
        )[list(feature_columns)]

        X_test_out = pd.concat(
            [X_test[ecfp_cols], X_test_desc_scaled],
            axis=1
        )[list(feature_columns)]

        return X_train_out, X_test_out

    return _transform
