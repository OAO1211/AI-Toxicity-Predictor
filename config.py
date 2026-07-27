# config.py

import os

# =========================
# Base directories
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
FRAGMENTS_DIR = os.path.join(RESULTS_DIR, "fragments")

# =========================
# Feature extraction (ECFP)
# =========================
ECFP_RADIUS = 3
ECFP_BITS = 1024

# =========================
# SHAP & fragment extraction
# =========================
TOP_SHAP_BITS = 20        # top N bits to extract fragments
SHAP_SAMPLE_SIZE = 100    # 用於 KernelExplainer 的 sample size
FRAGMENT_PNG = True       # 是否存 fragment PNG

# =========================
# Cross-validation
# =========================
N_FOLDS = 5
RANDOM_STATE = 42
STRATIFIED = True

# =========================
# Models
# =========================
MODEL_CONFIGS = {
    "RF": {
        "builder": "models.train_rf.build_rf",
        "type": "rf"
    },
    "XGB": {
        "builder": "models.train_xgb.build_xgb",
        "type": "xgb"
    },
    "LogReg": {
        "builder": "models.train_logreg.build_logreg",
        "type": "linear"
    }
}

# =========================
# Grid Search (for hyperparameter tuning)
# =========================
GRID_SEARCH = {
    "enabled": True,
    "cv": 5,
    "scoring": "roc_auc",
    "n_jobs": -1,
    "models": {
        "RF": {
            "n_estimators": [200, 500],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 5]
        },
        "XGB": {
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.05],
            "n_estimators": [200, 400]
        },
        "LogReg": {
            "C": [0.1, 1.0, 10],
            "penalty": ["l2"]
        }
    }
}

# =========================
# Visualization
# =========================
PLOT_TOP_N_SHAP = 20

