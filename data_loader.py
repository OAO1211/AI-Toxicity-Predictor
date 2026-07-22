from preprocess.clean_data import fit_encoders, transform_with_encoders, split_xy
import pandas as pd

def load_data(file_path, label_col, sample_col=None, encoders=None):
    df = pd.read_csv(file_path, sep="\t", encoding="latin1")
    ids = df[sample_col].astype(str).values if sample_col else [f"sample_{i}" for i in range(len(df))]
    X = df.drop(columns=[label_col] + ([sample_col] if sample_col else []))
    y = df[label_col]

    from preprocess.clean_data import fit_encoders, transform_with_encoders
    if encoders is None:
        encoders = fit_encoders(X)
    X = transform_with_encoders(X, encoders)

    return ids, X, y, encoders
