# pPred — PD-1/PD-L1 Inhibitor Predictor

A Streamlit web application for screening small-molecule PD-1/PD-L1 inhibitors using machine learning models trained on ChEMBL bioactivity data. The application provides both **classification** (Active/Inactive) and **regression** (pIC50) predictions, together with drug-likeness profiling, PAINS alerts, and applicability domain analysis.

## Features

### Prediction Pipelines

* **Classifier:** ExtraTrees classifier predicting Active/Inactive compounds using an IC50 ≤ 1000 nM activity threshold.
* **Regressor:** Stacking ensemble combining LightGBM, XGBoost, and GradientBoosting models with Ridge as the final estimator to predict pIC50.

### Drug-Likeness Panel

* Lipinski's Rule of Five
* Veber rules
* QED drug-likeness score
* Pfizer 3/75 rule
* GSK 4/400 rule

### PAINS Alerts

* Pan-Assay Interference Compounds detection using RDKit's built-in PAINS patterns.

### Applicability Domain

* Tanimoto similarity-based assessment against 4,398 training compounds.
* Nearest-neighbor similarity determines whether a compound is within or outside the applicability domain.

### Model Performance Dashboard

* Cross-validation metrics for candidate models across both prediction pipelines.
* Scaffold-split test set results.
* Interactive model performance visualizations.

### Input Options

* **Single Prediction:** Enter a SMILES string.
* **Batch Screening:** Upload a CSV file with a `smiles` column, an SDF file, or paste multiple SMILES strings.

---

## Installation

### Prerequisites

* Python 3.9+
* RDKit
* Git

### Setup

Clone the repository:

```bash
git clone https://github.com/Ericaakanko/pPred_Project.git
cd pPred_Project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r app/requirements.txt
```

### RDKit

If RDKit cannot be installed using pip, install it through conda:

```bash
conda install -c conda-forge rdkit
```

---

## Usage

From the project root, run:

```bash
streamlit run app/app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

### Quick Start

1. Navigate to the **Single Prediction** tab.
2. Enter a SMILES string, such as `CCO` for ethanol, or a compound of interest.
3. Click **Predict**.
4. The application returns:

   * Classification result (Active/Inactive)
   * Classification probability
   * Predicted pIC50
   * Drug-likeness profile
   * PAINS alerts
   * Applicability domain assessment
   * 2D molecular structure

### Batch Screening

1. Navigate to the **Batch Screening** tab.
2. Upload a CSV containing a `smiles` column, an SDF file, or paste SMILES strings.
3. Click **Run Screening**.
4. Review the prediction results.
5. Download the results as a CSV file.

---

## Notebooks

The `notebooks/` directory contains the model development and analysis notebooks used for the pPred prediction pipelines.

### Classifier Notebook

[`pPred_classifier.ipynb`](notebooks/pPred_classifier.ipynb)

This notebook documents the development and evaluation of the classification pipeline for predicting whether compounds are Active or Inactive against the PD-1/PD-L1 target.

The workflow includes:

* Dataset preparation and cleaning
* Molecular feature generation
* Morgan fingerprint calculation
* RDKit molecular descriptors
* Feature selection
* Candidate model training
* Cross-validation
* Scaffold-based train/test evaluation
* Model comparison
* Y-scrambling validation
* Permutation testing
* Final classifier selection

### Regressor Notebook

[`pPred_regressor.ipynb`](notebooks/pPred_regressor.ipynb)

This notebook documents the development and evaluation of the regression pipeline for predicting compound potency as pIC50.

The workflow includes:

* Dataset preparation
* Molecular feature generation
* Morgan fingerprints and molecular descriptors
* Feature selection
* Candidate regression models
* Cross-validation
* Scaffold-based train/test evaluation
* Ensemble model development
* Y-scrambling validation
* Permutation testing
* Final regressor selection

The notebooks provide a reproducible record of the model development process, while the trained models in `models/` are used by the Streamlit application.

---

## Model Details

### Dataset

Models were trained using bioactivity data retrieved from ChEMBL targeting PD-1/PD-L1-related targets:

* **CHEMBL3580522** — PD-L1
* **CHEMBL4523993** — PD-1/PD-L1 complex
* **CHEMBL6066575** — PD-L1/2 family

After pooling and deduplication, the dataset contained **4,398 unique compounds**.

### Molecular Features

The prediction pipelines use molecular fingerprints and physicochemical descriptors generated with RDKit.

#### Morgan Fingerprints

* Radius: 2
* Fingerprint size: 2,048 bits

#### RDKit Molecular Descriptors

The descriptor set includes:

* Molecular weight
* LogP
* TPSA
* Number of H-bond donors
* Number of H-bond acceptors
* Number of rotatable bonds
* Number of aromatic rings
* Number of aliphatic rings
* Fraction CSP3
* Number of heavy atoms
* Number of rings
* BalabanJ
* BertzCT
* MolMR
* LabuteASA
* Number of valence electrons
* QED

Feature selection was performed using `VarianceThreshold` to remove constant features.

---

## Best Models

| Pipeline   | Best Model                                                        | Key Performance              |
| ---------- | ----------------------------------------------------------------- | ---------------------------- |
| Classifier | ExtraTreesClassifier                                              | ROC AUC = 0.918, MCC = 0.624 |
| Regressor  | Stacking ensemble (LightGBM + XGBoost + GradientBoosting → Ridge) | R² = 0.864, RMSE = 0.615     |

---

## Validation

The prediction pipelines were evaluated using multiple validation strategies:

* **Scaffold-based train/test split** to reduce the risk of scaffold leakage.
* **10-fold cross-validation** across candidate models.
* **Y-scrambling tests** to assess whether model performance could arise from chance associations.
* **Permutation tests** to further evaluate model significance.

The reported Y-scrambling results were:

* Classifier: Z = 29.47
* Regressor: Z = 106.17

Both pipelines showed permutation-test significance at **p < 0.0001**.

---

## Project Structure

```text
pPred_Project/
├── app/
│   ├── app.py
│   ├── assets/
│   │   └── ppred_logo.png
│   └── requirements.txt
│
├── models/
│   ├── classifier_model.pkl
│   ├── regressor_model.pkl
│   ├── feature_selector.pkl
│   ├── scaler.pkl
│   ├── train_smiles.pkl
│   ├── classifier_cv_metrics.csv
│   ├── classifier_test_results.csv
│   ├── regressor_cv_metrics.csv
│   └── regressor_test_results.csv
│
├── notebooks/
│   ├── pPred_classifier.ipynb
│   └── pPred_regressor.ipynb
│
├── README.md
└── .gitignore
```

### Directory Overview

* `app/` — Streamlit application and application dependencies.
* `models/` — Trained machine learning models, feature-processing objects, training data references, and evaluation results.
* `notebooks/` — Model development, analysis, validation, and evaluation workflows.
* `README.md` — Project documentation.
* `.gitignore` — Files and directories excluded from version control.

---

## Developer

**Erica Akanko**

Email: [eakank15@gmail.com](mailto:eakank15@gmail.com)

---

## License

This project is intended for research and educational purposes.

