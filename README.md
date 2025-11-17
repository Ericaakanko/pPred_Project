# pPred: Machine Learning–Based PD-1/PD-L1 Inhibitor Predictor

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Status-Active-success)

---

## Overview

**pPred** is a machine learning–powered web application built with **Streamlit** for predicting inhibitors of the **PD-1/PD-L1 immune checkpoint pathway**, a crucial target in immuno-oncology.  
It accepts **SMILES strings** as molecular input, generates **Morgan fingerprints**, and uses pre-trained models to classify compounds as **Active** or **Inactive**.

The tool is designed to help students, chemists, and researchers quickly evaluate molecular activity and visualize results interactively.

---

## Key Features

- **Interactive Streamlit Web App**
  - Predict PD-1/PD-L1 inhibitor activity from SMILES.
  - Visualize molecular structure and applicability domain plots.

- **Multiple Trained Models**
  - Random Forest, K-Nearest Neighbors, AdaBoost, Extra Trees, Gradient Boosting.

- **Feature Selection Integration**
  - Applies a pre-computed `selection.pkl` feature selector to reduce redundancy.

- **Batch Prediction Mode**
  - Upload a CSV/Excel file containing a `SMILES` column for bulk predictions.

- **Applicability Domain Visualization**
  - PCA-based plot showing where the query molecule lies in the model’s chemical space.

- **Open and Reproducible**
  - All artifacts (models, data, scripts, notebook) are organized for transparency and reuse.

---

## Project Structure

```bash
pPred_Project/
├─ app/
│  ├─ app.py                   # Streamlit app entry point
│  └─ assets/
│     └─ ppred_logo.png        # Logo used in the UI
│
├─ models/                     # Serialized machine learning models
│  ├─ rf.pkl
│  ├─ knn.pkl
│  ├─ adaboost.pkl
│  ├─ et.pkl
│  ├─ gb.pkl
│  └─ selection.pkl
│
├─ artifacts/                  # Derived data used for visualization
│  ├─ X_train.pkl
│  └─ X_test.pkl
│
├─ data/
│  └─ raw/
│     └─ bioactivity_data_descriptors_morgan.csv
│
├─ notebooks/
│  └─ pPred_classifier.ipynb
  
            # Jupyter notebook for model development
│
├─ scripts/                    # Utility scripts (training, evaluation, etc.)
│
├─ requirements.txt
├─ environment.yml
├─ README.md
└─ LICENSE
```

---

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/Ericaakanko/pPred_Project.git
cd pPred_Project
```

### 2. (Recommended) Create the Conda environment
Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) installed.

```bash
conda env create -f environment.yml
conda activate ppred
```

Alternatively, use `pip` (Python 3.11+):
```bash
python -m venv .venv
.venv\Scripts\activate    # (Windows)
pip install -r requirements.txt
```

---

## Running the Application

After activating your environment, launch Streamlit:

```bash
streamlit run app/app.py
```

Then open your browser to:
```
http://localhost:8501
```

---

## How to Use

### Single Prediction
1. Go to the **Predict** tab.
2. Enter a valid **SMILES string**.
3. Select a model.
4. View:
   - Predicted label (Active / Inactive)
   - Confidence score
   - Molecular structure
   - PCA plot showing the molecule’s location in the descriptor space.

### Batch Prediction
1. Choose **Upload SMILES File**.
2. Upload a `.csv` or `.xlsx` file containing a `SMILES` column.
3. View and download prediction results as a CSV file.

---

## Dependencies

Core libraries used in pPred:

| Category | Packages |
|-----------|-----------|
| Web UI | `streamlit`, `pillow` |
| Data | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `joblib` |
| Chemistry | `rdkit` |
| Visualization | `matplotlib` |

All dependencies and versions are managed via `environment.yml` and `requirements.txt`.

---

## Models and Artifacts

| File | Description |
|------|--------------|
| `rf.pkl`, `knn.pkl`, `adaboost.pkl`, `et.pkl`, `gb.pkl` | Trained classification models |
| `selection.pkl` | Feature selection mask |
| `X_train.pkl`, `X_test.pkl` | Descriptor matrices for PCA visualization |

> Note: Large models are stored locally and excluded from Git tracking using `.gitignore`. You can regenerate them via the notebook or scripts.

---

## Methodology Summary

1. **Data Preparation**  
   Bioactivity data were curated and transformed into **Morgan fingerprints (radius=2, nBits=2048)**.

2. **Feature Selection**  
   Recursive feature elimination and variance thresholding were used to retain informative features.

3. **Model Training**  
   Five classical ML algorithms were trained and evaluated using 5-fold cross-validation. The best models were serialized with `joblib`.

4. **Deployment**  
   Models were wrapped in a Streamlit interface for real-time predictions and visual analytics.

---

## Developers

**Erica Akanko**  
*eakank001@gmail.com*

**Samuel Selasi**   
*[github](https://github.com/samuelselasi)*


---

## License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute it with attribution.

---

## Acknowledgements

- **RDKit** developers for open-source cheminformatics tools.  
- **Streamlit** team for simplifying ML web deployment.  
- Supervisors and colleagues for guidance during project development.

---


## Troubleshooting

| Issue | Fix |
|-------|-----|
| `conda` not recognized | Reopen “Anaconda Prompt” or ensure Conda is added to PATH |
| `RDKit` installation fails with pip | Use the provided Conda environment (`environment.yml`) |
| App can’t find models | Verify `models/` and `artifacts/` folders are in the repo root |
| Port already in use | Run with `streamlit run app/app.py --server.port 8502` |

---

## Reproducibility Statement

All data preprocessing, model training, and fingerprint generation steps are reproducible through the included **Jupyter notebook** (`notebooks/pPred.ipynb`) and **scripts** (`scripts/`).  
Results and figures match those used in the deployed Streamlit app.

---
