# AI-Toxicity-Predictor

## Explainable AI Framework for Drug-Induced Liver Injury (DILI) Prediction Using Molecular Fingerprints and Machine Learning

---

## Overview

Drug-induced liver injury (DILI) is one of the major causes of drug failure during clinical development and post-market withdrawal. Early identification of hepatotoxic compounds remains a critical challenge in drug discovery.

This project develops an **explainable machine learning framework** for predicting DILI toxicity from molecular structures and identifying chemical substructures associated with toxicity prediction.

The framework integrates:

- Molecular representation using RDKit: ECFP fingerprints **and** physicochemical descriptors (as a baseline for comparison)
- Multiple machine learning models
- Nested cross-validation evaluation
- SHAP-based model interpretation
- Chemical fragment visualization (ECFP only)

The goal is not only to predict toxicity risk, but also to provide **chemically interpretable explanations** for model decisions, and to compare how much predictive signal comes from structural fingerprints versus simple physicochemical properties.

---

# Pipeline Overview
                          Molecular Structure (SMILES)
                                     |
                                     v
                              RDKit Processing
                                     |
                    ┌────────────────┴────────────────┐
                    v                                  v
          ECFP Molecular Fingerprints      Physicochemical Descriptors
                    |                                  |
                    |                    ┌─────────────┴─────────────┐
                    |                    v                           v
                    |         Combined_Naive              Combined_Scaled
                    |         (concat, no scaling)   (concat, descriptors standardized)
                    |                    |                           |
                    └───────┬────────────┴─────────────┬─────────────┘
                             v                          v
                     Machine Learning Classification
                (Random Forest / XGBoost / Logistic Regression)
                     — each of the 4 feature sets evaluated independently —
                                      |
                                      v
                     Nested 5-fold Cross-validation
                       (inner grid search per outer fold)
                                      |
                                      v
                            SHAP Explainability
                                      |
                                      v
                Molecular Fragment Interpretation (ECFP only)
                                      |
                                      v
        ECFP vs Descriptors vs Combined Performance Comparison
 
---

# Features

## Molecular Feature Extraction

Two independent feature representations are extracted from SMILES, evaluated separately, and then combined into a third feature set:

**1. ECFP (Extended Connectivity Fingerprints)**
- Convert SMILES strings into molecular fingerprints using RDKit
- Configurable fingerprint radius and bit length

Default configuration:
ECFP radius = 3
ECFP bits = 1024

**2. Physicochemical Descriptors (baseline)**
- 15 classic RDKit-computed descriptors: MolWt, ExactMolWt, MolLogP, TPSA, HBD, HBA, RotatableBonds, RingCount, AromaticRings, AliphaticRings, HeavyAtoms, FractionCSP3, MolMR, LabuteASA, BertzCT
- Low-dimensional, human-interpretable, no bit-hashing collisions
- Used as a baseline to quantify how much predictive value the high-dimensional structural fingerprint (ECFP) adds over simple physicochemical properties

**3. Combined (ECFP + Descriptors)**
- The two feature sets above concatenated into a single 1039-dimensional matrix, evaluated in two variants:
  - **Combined_Naive**: concatenated as-is, no scaling
  - **Combined_Scaled**: descriptors standardized with `StandardScaler` (fit on the outer training fold only, to avoid leakage), ECFP bits left untouched since they are already binary
- Tree-based models (RF/XGB) are expected to be invariant to this scaling by construction; the comparison mainly matters for Logistic Regression, where unscaled descriptors (e.g. MolWt ≈ 300) and binary ECFP bits (0/1) would otherwise distort L2-regularized coefficient estimates
- Used to test whether descriptors provide complementary information beyond what ECFP already captures

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

Hyperparameter optimization is performed using **nested** grid search: for every outer cross-validation fold, the grid search is run only on that fold's training data (inner 5-fold CV), so the held-out outer test fold never influences hyperparameter selection.

Examples:

- Number of estimators
- Tree depth
- Learning rate
- Regularization parameters

---

## Cross-validation

Model performance is evaluated using nested stratified 5-fold cross-validation (5 outer folds, each with its own inner grid search), reported as mean ± std across the 5 outer folds:

- ROC-AUC
- PR-AUC (average precision — more informative than ROC-AUC given the moderately imbalanced label distribution, ~267:183)
- Balanced Accuracy
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
│   ├── descriptors.py
│   └── bit_mapping.py
├── models/
│   ├── train_rf.py
│   ├── train_xgb.py
│   └── train_logreg.py
├── evaluation/
│   ├── cross_validation.py
│   └── aggregate_metrics.py
├── explain/
│   └── shap_analysis.py
├── preprocess/
│   ├── clean_data.py
│   └── scaling.py
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
After running the pipeline, the following outputs will be generated, separately under results/<dataset_name>/ECFP/ and results/<dataset_name>/Descriptors/:
Model Evaluation
Examples:
ROC-AUC
PR-AUC
Balanced Accuracy
F1-score
Precision / Recall
Confusion matrix
SHAP Outputs
Generated files:
training_shap.tsv

testing_shap.tsv

mean_abs_shap.tsv
Fragment Visualization (ECFP only)
Important molecular fragments identified by SHAP are converted back into chemical structures using RDKit. Descriptors don't have a bit-to-substructure mapping, so this step only runs for the ECFP feature set.
Example output:
results/<dataset_name>/ECFP/

└── fragments/

    ├── RF/

    ├── XGB/

    └── LogReg/
Feature Set Comparison
results/comparison/metrics_summary.csv aggregates all folds, grouped by FeatureSet + Model (mean ± std), across all four feature sets (ECFP / Descriptors / Combined_Naive / Combined_Scaled) and all three models, so they can be compared side by side.
Future Development
Future improvements include:
Integration of molecular graph neural networks (GNN)
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