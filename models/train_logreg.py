# models/train_logreg.py

from sklearn.linear_model import LogisticRegression

def build_logreg(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
):
    """
    Logistic Regression for ECFP-based toxicity / DILI prediction

    Design principles:
    - Simple, interpretable baseline
    - GridSearchCV compatible
    - SHAP-friendly (tree-less, linear model)
    - Reviewer-safe default configuration
    """

    return LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state
    )
