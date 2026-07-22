DILI / Toxicity Prediction Pipeline

This repository contains a pipeline for drug-induced liver injury (DILI) and toxicity prediction using molecular fingerprints (ECFP) and machine learning models. The pipeline supports feature extraction, model training, cross-validation, SHAP explainability, and fragment visualization.

Table of Contents

Features

Installation

Project Structure

Data

Usage

Pipeline Overview

Modeling

SHAP Explainability

Fragment Visualization

Results

Requirements

License

Features

ECFP (Extended-Connectivity Fingerprints) feature extraction from SMILES.

Training with multiple ML models:

Random Forest (RF)

XGBoost (XGB)

Logistic Regression (LogReg)

5-fold cross-validation with SHAP explainability.

Extract top SHAP features and map them to molecular fragments.

Generate fragment images for top contributing ECFP bits.

Configurable for different fingerprint radius, bit length, and top fragments.

Installation

Clone the repository:

cd dili_ml_project


Install dependencies:

pip install -r requirements.txt


Note: RDKit is recommended to install via Conda for stability:

conda install -c conda-forge rdkit

Project Structure
dili_ml_project/
│
├─ data/                   # Input CSV datasets
├─ results/                # Output from pipeline
│   ├─ <dataset>/          # Dataset-specific folder
│   │   ├─ RF/             # RF fold outputs
│   │   ├─ XGB/            # XGB fold outputs
│   │   ├─ LogReg/         # Logistic Regression fold outputs
│   │   └─ fragments/      # Top SHAP fragments PNGs & TSV
├─ feature_extraction/
│   ├─ ecfp.py             # ECFP feature extraction
│   └─ bit_mapping.py      # Map ECFP bits to fragments
├─ models/
│   ├─ train_rf.py
│   ├─ train_xgb.py
│   └─ train_logreg.py
├─ evaluation/
│   ├─ cross_validation.py
│   └─ metrics.py
├─ explain/
│   ├─ shap_analysis.py
│   └─ shap_to_fragments.py
├─ preprocess/
│   └─ clean_data.py
├─ visualization/
│   ├─ draw_fragments.py
│   └─ plot_shap.py
├─ model_selection/
│   └─ grid_search.py
├─ config.py
├─ run_pipeline.py         # Main pipeline runner
├─ requirements.txt
└─ README.md

Data

Place your CSV datasets in the data/ folder. Each CSV should contain at least:

SMILES column (molecule representation)

label column (binary toxicity label)

LabelCompoundName column (compound name, optional)

Example:

LabelCompoundName	SMILES	label
Aspirin	CC(=O)OC1=CC	0
Paracetamol	CC(=O)NC1=CC	1
Usage

Run the full pipeline:

python run_pipeline.py --ecfp_radius 3 --ecfp_bits 1024 --top_shap_bits 20


Parameters:

ecfp_radius – radius for ECFP generation (default: 3)

ecfp_bits – length of fingerprint vector (default: 1024)

top_shap_bits – number of top SHAP bits to extract (default: 20)

Pipeline Overview

Feature Extraction: Convert SMILES to ECFP fingerprints (feature_extraction/ecfp.py).

Data Loading: Encode categorical features and split X/y (preprocess/clean_data.py).

Grid Search: Optimize hyperparameters for each model (model_selection/grid_search.py).

5-Fold Cross-Validation: Train, validate, and compute SHAP values (evaluation/cross_validation.py).

Fragment Extraction: Map top SHAP features back to molecular fragments (feature_extraction/bit_mapping.py).

Visualization: Save fragment images and SHAP plots (visualization/).

Modeling

Supported models:

Model	Description
RF	Random Forest, class-balanced, SHAP stable
XGB	XGBoost, tree-based, handles non-linear patterns
LogReg	Logistic Regression, interpretable baseline

Grid search is applied for hyperparameter tuning before cross-validation.

SHAP Explainability

Uses TreeExplainer for RF/XGB.

Supports KernelExplainer for linear or other models.

Computes mean absolute SHAP values per feature.

Saves per-sample SHAP values as TSV files in each fold folder.

Fragment Visualization

Extract top N SHAP features (bits).

Map bits to substructures using RDKit.

Save fragment images in results/<dataset>/fragments/<model>/pngs/.

Example folder structure:

results/
└─ acetaminophen/
   ├─ RF/
   ├─ XGB/
   ├─ LogReg/
   └─ fragments/
      ├─ RF/pngs/
      ├─ XGB/pngs/
      └─ LogReg/pngs/

Results

After running the pipeline, you will get:

training_shap.tsv and testing_shap.tsv per fold.

mean_abs_shap.tsv for aggregated SHAP values.

Top fragments extracted and visualized as PNG images.

Pipeline logs in console.

Requirements

Key Python packages:

numpy
pandas
scikit-learn
rdkit-pypi
xgboost
shap
matplotlib
seaborn


See requirements.txt for the complete list.

License

MIT License. See LICENSE file for details.