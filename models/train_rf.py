# models/train_rf.py

from sklearn.ensemble import RandomForestClassifier


def build_rf(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    min_samples_split=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
):
    """
    Random Forest for ECFP-based toxicity / DILI prediction

    Design principles:
    - ECFP bit-friendly
    - SHAP-stable (avoid extremely shallow trees)
    - GridSearch-compatible
    - Reviewer-safe default settings
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
