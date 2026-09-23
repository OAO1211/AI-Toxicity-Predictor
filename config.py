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


CURATED_DATA_DIR = os.path.join(
    DATA_DIR,
    "curated"
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

# Source DILIrank category. Final binary labels are regenerated from this
# column so manual label drift (e.g. the legacy temsirolimus error) cannot
# propagate into a formal run.
SOURCE_DILI_COL = "vDILIConcern"

DILI_LABEL_MAP = {
    "vMost-DILI-Concern": 1,
    "vNo-DILI-Concern": 0,
}

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

# Combined = ECFP + Descriptors 串接。分兩層：
# Naive：直接串接，不做任何 scaling（RF/XGB 對 scaling 不敏感，這層對它們才是真正的結果；
#        但對 LogReg 而言，descriptors 跟 0/1 的 ECFP bits 尺度差很多，會讓 L2 regularization
#        的係數解讀失真）
# Scaled：只對 15 個 descriptors 做 StandardScaler（ECFP bits 維持 0/1），
#         主要是為了讓 LogReg 的比較公平；RF/XGB 理論上應該跟 Naive 幾乎一樣，
#         可以拿來當作「scaling 只影響線性模型」的 sanity check。
FEATURE_SET_COMBINED_NAIVE = "Combined_Naive"

FEATURE_SET_COMBINED_SCALED = "Combined_Scaled"




# =========================
# ECFP parameters
# =========================


ECFP_RADIUS = 3

ECFP_BITS = 1024

# Final v4 representation is chirality-aware. This removes the five legacy
# stereochemical fingerprint collisions found during the structure audit.
ECFP_USE_CHIRALITY = True

# Predeclared sensitivity subset: this does NOT replace the 450-compound
# primary cohort. It is only used to test whether key findings persist in a
# more conventional organic small-molecule region.
SENSITIVITY_MAX_MW = 1000.0




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


# =========================
# Error analysis (Prediction Error Analysis / Chapter 4)
# =========================
#
# High-confidence error thresholds on the OOF predicted probability
# (probability of the positive / DILI class):
#   False Positive (true=non-DILI, pred=DILI): pred_prob >= FP threshold
#   False Negative (true=DILI, pred=non-DILI): pred_prob <= FN threshold

ERROR_FP_HIGH_CONF_THRESHOLD = 0.8

ERROR_FN_HIGH_CONF_THRESHOLD = 0.2

# How many top |SHAP| ECFP bits to report per individual error compound
# (item 4). Deliberately smaller than TOP_SHAP_BITS (which is a
# dataset-wide aggregate), since this is per-compound.
TOP_SHAP_BITS_PER_COMPOUND = 5

# Top-K structurally similar compounds (Tanimoto on ECFP) to report per
# high-confidence error compound (item 5).
SIMILARITY_TOP_K = 5

# Similarity threshold used for the "label disagreement among close
# neighbors" statistic: for each high-confidence error compound, what
# fraction of ALL compounds with Tanimoto similarity >= this threshold
# have a DIFFERENT true label. Answers "structurally near-identical,
# but different DILI outcome?" without being capped at Top-K.
SIMILARITY_LABEL_MISMATCH_THRESHOLD = 0.7

# How many of the worst errors (by |pred_prob - y_true|) to keep in
# top_10_high_confidence_errors.tsv (item 9).
TOP_N_WORST_ERRORS = 10

ERROR_ANALYSIS_DIRNAME = "error_analysis"