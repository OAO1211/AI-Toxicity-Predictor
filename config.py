# config.py

import os

# =========================
# Model imports
# =========================

from models.train_rf import build_rf
from models.train_xgb import build_xgb
from models.train_logreg import build_logreg


from model_selection.grid_search import (
    rf_grid_search,
    xgb_grid_search,
    logreg_grid_search
)



# =========================
# Base directories
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# Data

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


RAW_DATA_DIR = os.path.join(
    DATA_DIR,
    "raw"
)


FEATURE_DIR = os.path.join(
    DATA_DIR,
    "features"
)



# Results

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


PLOTS_DIR = os.path.join(
    RESULTS_DIR,
    "plots"
)


FRAGMENTS_DIR = os.path.join(
    RESULTS_DIR,
    "fragments"
)




# =========================
# Dataset columns
# =========================


LABEL_COL = "label"

NAME_COL = "LabelCompoundName"

SMILES_COL = "SMILES"


# =========================
# Feature set names
# =========================
#
# 用於區分不同的特徵集合（ECFP vs 物理化學 descriptors），
# 兩者會分別跑過同一組模型，作為 baseline 比較。
# run_pipeline.py / evaluation/aggregate_metrics.py 共用這兩個常數，
# 避免字串在不同檔案裡打錯導致比較表對不起來。

FEATURE_SET_ECFP = "ECFP"

FEATURE_SET_DESCRIPTORS = "Descriptors"




# =========================
# ECFP parameters
# =========================


ECFP_RADIUS = 3

ECFP_BITS = 1024




# =========================
# SHAP parameters
# =========================


TOP_SHAP_BITS = 20


SHAP_SAMPLE_SIZE = 100


FRAGMENT_PNG = True




# =========================
# Cross validation
# =========================


N_FOLDS = 5


RANDOM_STATE = 42


STRATIFIED = True




# =========================
# Model Registry
# =========================
#
# run_pipeline.py 會從這裡讀取模型
#
#


MODEL_CONFIGS = {

    "RF": {

        "builder": build_rf,

        "grid_search": rf_grid_search,

        "model_type": "rf"

    },


    "XGB": {

        "builder": build_xgb,

        "grid_search": xgb_grid_search,

        "model_type": "xgb"

    },


    "LogReg": {

        "builder": build_logreg,

        "grid_search": logreg_grid_search,

        "model_type": "logreg"

    }

}




# =========================
# Grid Search parameters
# =========================
#
# 注意：實際使用的參數網格定義在
# model_selection/grid_search.py 的 get_model_and_param_grid()。
# 這裡先前有一份重複、且未被任何程式引用的 GRID_SEARCH 設定，
# 內容與 grid_search.py 不一致，容易讓人誤以為改這裡就能調整搜尋範圍，
# 已移除以避免混淆。若要調整超參數搜尋範圍，請直接修改
# model_selection/grid_search.py。




# =========================
# Visualization
# =========================


PLOT_TOP_N_SHAP = 20