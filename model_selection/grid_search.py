# model_selection/grid_search.py
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# =========================
# Metric
# =========================
roc_auc = make_scorer(
    roc_auc_score,
    needs_proba=True
)

# =========================
# Model & param grid
# =========================
def get_model_and_param_grid(model_name: str):
    """
    回傳 (estimator, param_grid)
    """
    if model_name.lower() == "rf":
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 10, 30],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 3],
            "clf__max_features": ["sqrt", "log2"]
        }

    elif model_name.lower() in ["logreg", "logistic"]:
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=500,
            random_state=42
        )
        param_grid = {
            "clf__C": [0.01, 0.1, 1, 10]
        }

    elif model_name.lower() == "xgb":
        model = xgb.XGBClassifier(
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            eval_metric="logloss"
        )
        param_grid = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 6, 10],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.7, 0.8],
            "clf__colsample_bytree": [0.7, 0.8],
            "clf__min_child_weight": [1, 5]
        }

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, param_grid

# =========================
# Grid search main
# =========================
def run_grid_search(
    X,
    y,
    model_name: str = "rf",
    cv_splits: int = 5,
    scoring=roc_auc,
    verbose: int = 2,
    n_jobs: int = -1
):
    """
    執行 GridSearchCV 並回傳 (best_params, best_score)
    """
    model, param_grid = get_model_and_param_grid(model_name)
    pipeline = Pipeline([("clf", model)])

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True
    )

    grid.fit(X, y)

    best_params = {k.replace("clf__", ""): v for k, v in grid.best_params_.items()}
    best_score = grid.best_score_

    print(f"[✓] Grid search finished ({model_name}) - grid_search.py:113")
    print(f"Best score ({scoring._score_func.__name__}): {best_score:.4f} - grid_search.py:114")
    print("Best params: - grid_search.py:115")
    for k, v in best_params.items():
        print(f"{k}: {v} - grid_search.py:117")

    return best_params, best_score

# =========================
# Wrapper functions
# =========================
def rf_grid_search(X, y): return run_grid_search(X, y, model_name="rf")
def xgb_grid_search(X, y): return run_grid_search(X, y, model_name="xgb")
def logreg_grid_search(X, y): return run_grid_search(X, y, model_name="logreg")
