# models/train_xgb.py

import xgboost as xgb

def build_xgb(
    max_depth=6,
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
    verbosity=0
):
    """
    XGBoost for ECFP-based toxicity / DILI prediction

    Design principles:
    - Tree-based SHAP friendly
    - Handles non-linear patterns
    - Reviewer-safe default configuration
    - GridSearchCV compatible
    """

    return xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        eval_metric=eval_metric,
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=verbosity
    )
