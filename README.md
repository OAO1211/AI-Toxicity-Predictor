# AI-Toxicity-Predictor

## Explainable AI Framework for Drug-Induced Liver Injury (DILI) Prediction Using Molecular Fingerprints and Machine Learning

---

## Overview

Drug-induced liver injury (DILI) is one of the major causes of drug failure during clinical development and post-market withdrawal. Early identification of hepatotoxic compounds remains a critical challenge in drug discovery.

This project develops an **explainable machine learning framework** for predicting DILI toxicity from molecular structures and identifying chemical substructures associated with toxicity prediction.

The framework integrates:

- Molecular representation using RDKit and ECFP fingerprints
- Multiple machine learning models
- Cross-validation evaluation
- SHAP-based model interpretation
- Chemical fragment visualization

The goal is not only to predict toxicity risk, but also to provide **chemically interpretable explanations** for model decisions.

---

# Pipeline Overview
Molecular Structure (SMILES)
              |
              v
        RDKit Processing
              |
              v
  ECFP Molecular Fingerprints
              |
              v
 Machine Learning Classification
 (Random Forest / XGBoost / Logistic Regression)
              |
              v
     Cross-validation Evaluation
              |
              v
        SHAP Explainability
              |
              v
 Molecular Fragment Interpretation
 
---

# Features

## Molecular Feature Extraction

- Convert SMILES strings into molecular fingerprints using RDKit
- Generate Extended Connectivity Fingerprints (ECFP)
- Configurable fingerprint radius and bit length

Default configuration:
ECFP radius = 3
ECFP bits = 1024
---

## Machine Learning Models

The framework supports multiple classification algorithms:

| Model | Description |
|---|---|
| Random Forest | Non-linear ensemble classifier with SHAP interpretation |
| XGBoost | Gradient boosting model for complex molecular patterns |
| Logistic Regression | Interpretable baseline model |

---

## Model Optimization

Hyperparameter optimization is performed using grid search.

Examples:

- Number of estimators
- Tree depth
- Learning rate
- Regularization parameters

---

## Cross-validation

Model performance is evaluated using:

- Stratified 5-fold cross-validation
- ROC-AUC
- F1-score
- Precision
- Recall
- Confusion matrix

---

# Explainable AI Interpretation

A major focus of this project is transforming machine learning predictions into chemically meaningful information.

Instead of treating models as black boxes, SHAP (SHapley Additive exPlanations) is applied to identify molecular features contributing to toxicity predictions.

Workflow:
SHAP Importance
        |
        v
Important ECFP Bits
        |
        v
RDKit Bit Mapping
        |
        v
Chemical Fragment Visualization
This enables identification of structural patterns potentially associated with hepatotoxicity.

---

# Project Structure
dili_ml_project/
├── data/
│   └── raw molecular datasets
├── features/
│   ├── ecfp.py
│   └── bit_mapping.py
├── models/
│   ├── train_rf.py
│   ├── train_xgb.py
│   └── train_logreg.py
├── evaluation/
│   ├── cross_validation.py
│   └── metrics.py
├── explain/
│   ├── shap_analysis.py
│   └── shap_to_fragments.py
├── preprocess/
│   └── clean_data.py
├── visualization/
│   ├── draw_fragments.py
│   └── plot_shap.py
├── model_selection/
│   └── grid_search.py
├── results/
│   ├── model performance
│   ├── SHAP results
│   └── fragment visualization
├── config.py
├── run_pipeline.py
├── requirements.txt
└── README.md
---

# Dataset

Input datasets should contain:

| Column | Description |
|---|---|
| SMILES | Molecular representation |
| label | Binary toxicity label |
| LabelCompoundName | Compound name |

Example:

| LabelCompoundName | SMILES | label |
|---|---|---|
| Aspirin | CC(=O)OC1=CC | 0 |
| Example compound | CCN1CCC | 1 |

---

# Installation

## Clone repository

```bash
git clone https://github.com/OAO1211/AI-Toxicity-Predictor.git

cd dili_ml_projectInstall dependencies
pip install -r requirements.txt
RDKit Installation
RDKit is recommended to install through Conda:
conda install -c conda-forge rdkit
Usage
Run the complete pipeline:
python run_pipeline.py
The pipeline will automatically perform:
Molecular fingerprint generation
Dataset loading
Model training
Hyperparameter optimization
5-fold cross-validation
SHAP calculation
Fragment extraction
Visualization generation
Results
After running the pipeline, the following outputs will be generated:
Model Evaluation
Examples:
ROC-AUC
F1-score
Precision / Recall
Confusion matrix
SHAP Outputs
Generated files:
training_shap.tsv

testing_shap.tsv

mean_abs_shap.tsv
Fragment Visualization
Important molecular fragments identified by SHAP are converted back into chemical structures using RDKit.
Example output:
results/

└── fragments/

    ├── RF/

    ├── XGB/

    └── LogReg/
Future Development
Future improvements include:
Integration of molecular graph neural networks (GNN)
Incorporation of physicochemical properties
Integration of pharmacokinetic and ADMET features
Larger DILI benchmark datasets
Deployment as an interactive prediction platform
Research Direction
This project explores the intersection of:
Artificial Intelligence
Computational Chemistry
Drug Discovery
Explainable Machine Learning
The long-term goal is to develop AI systems capable of assisting early-stage drug development by predicting molecular risks and providing interpretable chemical insights.
Requirements
Main dependencies:
numpy
pandas
scikit-learn
rdkit
xgboost
shap
matplotlib
seaborn
openpyxl
See requirements.txt for the complete environment.
License
MIT License