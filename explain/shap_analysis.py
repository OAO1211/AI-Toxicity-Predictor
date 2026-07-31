# explain/shap_analysis.py
from xml.parsers.expat import model

import shap
import numpy as np
import pandas as pd
import warnings
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


def compute_shap(model,X,model_type):

    if model_type in ["rf","xgb"]:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)


        if isinstance(shap_values,list):
            shap_values = shap_values[1]


        if len(shap_values.shape)==3:
            shap_values = shap_values[:,:,1]


        expected_value = explainer.expected_value

        if isinstance(expected_value,(list,np.ndarray)):
            expected_value = expected_value[1]


        return (
            shap_values,
            expected_value,
            explainer
        )


    elif model_type=="logreg":


        explainer = shap.LinearExplainer(
            model,
            X,
            feature_perturbation="interventional"
        )


        shap_values = explainer.shap_values(X)


        expected_value = explainer.expected_value


        return (
            shap_values,
            expected_value,
            explainer
        )


    else:
        raise ValueError(
            f"Unknown model type {model_type}"
        )

def mean_abs_shap(shap_values):
    return np.mean(np.abs(shap_values), axis=0)


def save_shap_output(
    ids,
    shap_values,
    baseline,
    pred_prob,
    feature_names,
    out_path
):
    """
    Save SHAP values with correct ECFP bit names
    """

    # SHAP新版 RandomForest output:
    # (samples, features, classes)
    # 取 label=1 (toxicity class)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    df_shap = pd.DataFrame(
        shap_values,
        columns=feature_names
    )

    df_shap.insert(0, "SampleID", ids)
    df_shap.insert(1, "baseline_value", baseline)
    df_shap.insert(2, "prediction_prob", pred_prob)

    df_shap.to_csv(
        out_path,
        sep="\t",
        index=False
    )