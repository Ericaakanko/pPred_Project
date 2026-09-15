# pPred — PD-1/PD-L1 Inhibitor Predictor

A Streamlit web application for screening small-molecule PD-1/PD-L1 inhibitors using machine learning models trained on ChEMBL bioactivity data. The app provides both **classification** (Active/Inactive) and **regression** (pIC50) predictions, along with drug-likeness profiling, PAINS alerts, and applicability domain analysis.

---

## Features

### Prediction Pipelines
- **Classifier**: ExtraTrees classifier predicting Active/Inactive (threshold: IC50 ≤ 1000 nM)
- **Regressor**: Stacking ensemble (LightGBM + XGBoost + GradientBoosting → Ridge) predicting pIC50

### Drug-Likeness Panel
- Lipinski's Rule of Five
- Veber rules
- QED drug-likeness score
- Pfizer 3/75 rule
- GSK 4/400 rule

### PAINS Alerts
- Pan-Assay Interference Compounds detection using RDKit's built-in PAINS patterns

### Applicability Domain
- Tanimoto similarity-based assessment against 4,398 training compounds
- Nearest-neighbor distance determines in-domain vs. out-of-domain status

### Model Performance Dashboard
- Cross-validation metrics for all 10 candidate models per pipeline
- Scaffold-split test set results
- Interactive bar charts comparing model performance

### Input Options
- **Single Prediction**: Paste a SMILES string
- **Batch Screening**: Upload CSV (with `smiles` column), SDF file, or paste multiple SMILES (one per line)

---

## Installation

### Prerequisites
- Python 3.9+
- RDKit (requires conda or pip installation)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pPred.git
cd pPred

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt
```

> **Note on RDKit**: If `pip install rdkit` fails, install via conda:
> ```bash
> conda install -c conda-forge rdkit
> ```

---

## Usage

```bash
cd pPred
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

### Quick Start
1. Navigate to the **Single Prediction** tab
2. Enter a SMILES string (e.g., `CCO` for ethanol, or a known PD-1/PD-L1 inhibitor SMILES)
3. Click **Predict** to see:
   - Classification result (Active/Inactive) with probability
   - Regression result (predicted pIC50)
   - Drug-likeness profile
   - PAINS alerts
   - Applicability domain assessment
   - 2D molecular structure rendering

### Batch Screening
1. Navigate to the **Batch Screening** tab
2. Upload a CSV file with a `smiles` column, an SDF file, or paste SMILES (one per line)
3. Click **Run Screening** to process all compounds
4. Results are displayed in a table and can be downloaded as CSV

---

## Model Details

### Dataset
Models were trained on bioactivity data from ChEMBL targeting:
- **CHEMBL3580522** — PD-L1 (739 compounds)
- **CHEMBL4523993** — PD-1/PD-L1 complex (3,256 compounds)
- **CHEMBL6066575** — PD-L1/2 family (427 compounds)

Pooled and deduplicated: **4,398 unique compounds**.

### Features
- **Morgan fingerprints**: radius=2, 2048 bits
- **17 RDKit descriptors**: MW, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, NumAromaticRings, NumAliphaticRings, FractionCSP3, NumHeavyAtoms, NumRings, BalabanJ, BertzCT, MolMR, LabuteASA, NumValenceElectrons, QED
- **Feature selection**: VarianceThreshold (removes constant features)

### Best Models

| Pipeline | Best Model | Key Metric |
|----------|-----------|------------|
| Classifier | ExtraTreesClassifier | ROC AUC = 0.918, MCC = 0.624 |
| Regressor | Stacking (LGBM + XGB + GB → Ridge) | R² = 0.864, RMSE = 0.615 |

### Validation
- **Scaffold-based train/test split** ensures no scaffold leakage
- **10-fold cross-validation** across 10 candidate models per pipeline
- **Y-scrambling test**: Both models PASS (classifier Z=29.47, regressor Z=106.17)
- **Permutation tests**: Both p < 0.0001

---

## Project Structure

```
pPred/
├── app/
│   ├── app.py                  # Main Streamlit application
│   ├── assets/
│   │   └── ppred_logo.png      # App logo
│   └── requirements.txt        # Python dependencies
├── models/
│   ├── classifier_model.pkl    # ExtraTrees classifier
│   ├── regressor_model.pkl     # Stacking regressor
│   ├── feature_selector.pkl    # VarianceThreshold selector
│   ├── scaler.pkl              # StandardScaler (included for completeness)
│   ├── train_smiles.pkl        # Training SMILES for AD analysis
│   ├── classifier_cv_metrics.csv
│   ├── classifier_test_results.csv
│   ├── regressor_cv_metrics.csv
│   └── regressor_test_results.csv
├── README.md
└── .gitignore
```

---

## Developer

**Erica Akanko**
Email: eakank15@gmail.com

---

## License

This project is intended for research and educational purposes.
