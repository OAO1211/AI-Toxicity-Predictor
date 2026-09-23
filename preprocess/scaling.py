# preprocess/scaling.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class DescriptorOnlyScaler(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that only standardizes columns whose
    name does NOT start with "ECFP_" (i.e. the physicochemical
    descriptors), leaving the 0/1 ECFP bits untouched.

    This is what model_selection.grid_search.make_scaled_builder /
    Pipeline([("scaler", DescriptorOnlyScaler()), ("clf", model)])
    expects to import. It was referenced but never defined here before,
    which made `import model_selection.grid_search` (and therefore
    `import config`) fail with ImportError before any pipeline code
    could run at all — fixed by adding the class back.

    fit() only ever sees the training fold (GridSearchCV clones a fresh,
    unfit copy of this transformer for every inner fold), so there is no
    leakage of validation-fold statistics into the scaler.
    """

    def __init__(self):
        self.descriptor_cols_ = None
        self.ecfp_cols_ = None
        self.scaler_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.descriptor_cols_ = [
            c for c in self.columns_ if not str(c).startswith("ECFP_")
        ]
        self.ecfp_cols_ = [
            c for c in self.columns_ if str(c).startswith("ECFP_")
        ]

        self.scaler_ = StandardScaler()
        if self.descriptor_cols_:
            self.scaler_.fit(X[self.descriptor_cols_])
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)

        if self.descriptor_cols_:
            desc_scaled = pd.DataFrame(
                self.scaler_.transform(X[self.descriptor_cols_]),
                columns=self.descriptor_cols_,
                index=X.index
            )
        else:
            desc_scaled = X[self.descriptor_cols_]

        out = pd.concat(
            [X[self.ecfp_cols_], desc_scaled],
            axis=1
        )[self.columns_]

        return out


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
