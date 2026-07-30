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


GRID_SEARCH = {

    "enabled": True,


    "cv": 5,


    "scoring": "roc_auc",


    "n_jobs": -1,



    "models": {


        "RF": {

            "n_estimators": [
                200,
                500
            ],

            "max_depth": [
                None,
                20
            ],

            "min_samples_leaf": [
                1,
                5
            ]

        },



        "XGB": {

            "max_depth": [
                3,
                6
            ],

            "learning_rate": [
                0.01,
                0.05
            ],

            "n_estimators": [
                200,
                400
            ]

        },



        "LogReg": {

            "C": [
                0.1,
                1.0,
                10
            ],

            "penalty": [
                "l2"
            ]

        }

    }

}




# =========================
# Visualization
# =========================


PLOT_TOP_N_SHAP = 20