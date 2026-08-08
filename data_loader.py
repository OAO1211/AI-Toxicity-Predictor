from preprocess.clean_data import fit_encoders, transform_with_encoders, split_xy
import pandas as pd

def load_data(file_path, label_col, sample_col=None, encoders=None, drop_cols=("SMILES",)):
    """
    載入特徵檔案 (ECFP tsv)

    drop_cols: 除了 label_col / sample_col 之外，還要從特徵矩陣 X 中排除的欄位。
    預設排除 "SMILES"：這是一個字串欄位，且每一列幾乎都是唯一值。
    若不排除，preprocess.clean_data.fit_encoders 會把它 LabelEncode 成一個
    近乎「唯一識別碼」的整數特徵，模型（尤其是樹模型）可能利用這個 ID
    去記憶訓練樣本，造成不真實的高分與 SHAP 重要性污染。
    """
    df = pd.read_csv(file_path, sep="\t", encoding="latin1")
    ids = df[sample_col].astype(str).values if sample_col else [f"sample_{i}" for i in range(len(df))]

    cols_to_drop = [label_col]
    if sample_col:
        cols_to_drop.append(sample_col)
    for col in drop_cols:
        if col in df.columns and col not in cols_to_drop:
            cols_to_drop.append(col)

    X = df.drop(columns=cols_to_drop)
    y = df[label_col]

    if encoders is None:
        encoders = fit_encoders(X)
    X = transform_with_encoders(X, encoders)

    return ids, X, y, encoders
