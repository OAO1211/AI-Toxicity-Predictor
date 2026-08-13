# model_selection/grid_search.py
import xgboost as xgb

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# =========================
# Metric
# =========================


# =========================
# Model & param grid
# =========================
def get_model_and_param_grid(model_name: str, quick_mode: bool = False):
    """
    回傳 (estimator, param_grid)

    quick_mode=True 時，每個模型只會回傳「單一組」參數，
    GridSearchCV 實質上等於直接訓練一次、沒有真正在搜尋超參數。
    這個模式唯一的用途是快速確認整條 pipeline
    （feature extraction -> grid search -> nested CV -> SHAP -> aggregate）
    能不能正常跑完，不能拿 quick_mode 的結果當作正式報告數字。
    """
    if model_name.lower() == "rf":
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        if quick_mode:
            param_grid = {
                "clf__n_estimators": [200],
                "clf__max_depth": [None],
                "clf__min_samples_split": [2],
                "clf__min_samples_leaf": [1],
                "clf__max_features": ["sqrt"]
            }
        else:
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

        if quick_mode:
            param_grid = {
                "clf__C": [1]
            }
        else:
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

        if quick_mode:
            param_grid = {
                "clf__n_estimators": [200],
                "clf__max_depth": [6],
                "clf__learning_rate": [0.05],
                "clf__subsample": [0.8],
                "clf__colsample_bytree": [0.8],
                "clf__min_child_weight": [1]
            }
        else:
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

from sklearn.metrics import roc_auc_score

from preprocess.scaling import DescriptorOnlyScaler


# =========================
# Grid search main
# =========================

def run_grid_search(
    X,
    y,
    model_name="rf",
    cv_splits=5,
    verbose=2,
    n_jobs=-1,
    quick_mode=False,
    use_scaler=False
):
    """
    use_scaler=True 時，會在 Pipeline 裡加一個 DescriptorOnlyScaler
    (只 scale 非 ECFP_ 開頭的欄位)：

        Pipeline([("scaler", DescriptorOnlyScaler()), ("clf", model)])

    因為 scaler 是 Pipeline 的一個 step，GridSearchCV 會在每一個
    inner CV fold 自動 clone 一份全新、未 fit 的 scaler，只用該 inner
    fold 的 training 部分去 fit——inner validation fold 不會有任何
    分布資訊洩漏進 scaler，這是嚴格定義下的 nested cross-validation
    （而不是「先在整個 outer training set 上 fit 一次 scaler，再做
    inner CV」這種較寬鬆的做法）。
    """

    if quick_mode:
        print(
            f"[QUICK MODE] {model_name}: 只用單一參數組合，"
            f"結果僅供流程測試，不可用於正式報告！"
        )

    model, param_grid = get_model_and_param_grid(
        model_name,
        quick_mode=quick_mode
    )

    if use_scaler:
        pipeline = Pipeline([
            ("scaler", DescriptorOnlyScaler()),
            ("clf", model)
        ])
    else:
        pipeline = Pipeline([
            ("clf", model)
        ])


    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )


    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )


    grid.fit(X,y)


    best_params = {
        k.replace("clf__",""):v
        for k,v in grid.best_params_.items()
    }


    best_score = grid.best_score_


    print(
        f"[✓] Grid search finished ({model_name}, quick_mode={quick_mode}, "
        f"use_scaler={use_scaler})"
    )

    print(
        f"Best ROC-AUC: {best_score:.4f}"
    )

    print(
        f"Best params: {best_params}"
    )


    return best_params,best_score
# =========================
# Wrapper functions
# =========================
def rf_grid_search(X, y, quick_mode=False, use_scaler=False):
    return run_grid_search(
        X,
        y,
        model_name="rf",
        quick_mode=quick_mode,
        use_scaler=use_scaler
    )


def xgb_grid_search(X, y, quick_mode=False, use_scaler=False):
    return run_grid_search(
        X,
        y,
        model_name="xgb",
        quick_mode=quick_mode,
        use_scaler=use_scaler
    )


def logreg_grid_search(X, y, quick_mode=False, use_scaler=False):
    return run_grid_search(
        X,
        y,
        model_name="logreg",
        quick_mode=quick_mode,
        use_scaler=use_scaler
    )


# =========================
# Scaled builder wrapper
# =========================

def make_scaled_builder(base_builder):
    """
    把一個「回傳裸模型」的 builder（例如 build_logreg）包成
    「回傳 Pipeline(scaler + 裸模型)」的 builder，讓 run_5fold_cv
    在 outer fold 重新 fit 最終模型時，用的是跟 grid search 內部
    完全相同的 Pipeline 結構（scaler + clf），兩邊一致。

    每次呼叫都會建立一個全新、未 fit 的 DescriptorOnlyScaler，
    不會跟 grid search 內部 clone 出來的 scaler 共享任何已 fit 的狀態。
    """
    def _builder(**params):
        return Pipeline([
            ("scaler", DescriptorOnlyScaler()),
            ("clf", base_builder(**params))
        ])
    return _builder