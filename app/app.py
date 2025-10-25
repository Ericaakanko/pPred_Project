# app/app.py

from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]          # repo root
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="pPred – PD-1/PD-L1 Inhibitor Predictor",
    page_icon="🧪",
    layout="wide"
)


# -----------------------------
# Cached loaders (fast reloads)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_logo():
    return Image.open(ASSETS_DIR / "ppred_logo.png")


@st.cache_resource(show_spinner=True)
def load_selector():
    return joblib.load(MODELS_DIR / "selection.pkl")


@st.cache_resource(show_spinner=True)
def load_models():
    return {
        "Random Forest": joblib.load(MODELS_DIR / "rf.pkl"),
        "K-Nearest Neighbors": joblib.load(MODELS_DIR / "knn.pkl"),
        "AdaBoost": joblib.load(MODELS_DIR / "adaboost.pkl"),
        "Extra Trees": joblib.load(MODELS_DIR / "et.pkl"),
        "Gradient Boosting": joblib.load(MODELS_DIR / "gb.pkl"),
    }


@st.cache_resource(show_spinner=True)
def load_artifacts():
    # X_train / X_test should be on the same feature basis as the selector
    X_train = joblib.load(ARTIFACTS_DIR / "X_train.pkl")
    X_test = joblib.load(ARTIFACTS_DIR / "X_test.pkl")
    return X_train, X_test


# -----------------------------
# Utilities
# -----------------------------
def smiles_to_morgan_fp_array(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Convert SMILES to RDKit Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return mol, arr


def predict_single(smiles: str, model, selector):
    """Return (pred_label_int, pred_prob_float, mol_obj, X_selected_2d) or error."""
    mol, arr = smiles_to_morgan_fp_array(smiles)
    if mol is None:
        return None, None, None, None, "Invalid SMILES"

    # shape -> (1, n_bits)
    X = np.asarray(arr, dtype=float)[None, :]

    # feature selection / transformation
    try:
        X_sel = selector.transform(X)
    except Exception:
        X_sel = X  # fallback if selector not applicable

    # classification
    try:
        proba = model.predict_proba(X_sel)[0]
        y_hat = int(np.argmax(proba))
        conf = float(proba[y_hat])
    except Exception:
        # models without predict_proba
        y_hat = int(model.predict(X_sel)[0])
        conf = 1.0

    return y_hat, conf, mol, X_sel, None


def build_applicability_plot(X_train, X_test, X_query):
    """Make a PCA plot placing query against train/test."""
    # stack arrays
    X_all = np.vstack([X_train, X_test, X_query])
    labels = np.array(
        ["train"] * len(X_train) + ["test"] * len(X_test) + ["query"]
    )

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=7)
    X_pca = pca.fit_transform(X_std)

    fig, ax = plt.subplots()
    for group, marker in zip(["train", "test", "query"], ["o", "s", "^"]):
        idx = labels == group
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=group,
            alpha=0.6,
            marker=marker,
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Applicability Domain: PCA of Descriptor Space")
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------
# Header (logo + title)
# -----------------------------
logo = load_logo()
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=80)
with col2:
    st.markdown("<h1 style='margin-top: 20px;'>pPred</h1>", unsafe_allow_html=True)


# -----------------------------
# Load resources
# -----------------------------
selector = load_selector()
models = load_models()
X_train, X_test = load_artifacts()


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Predict", "Tutorial", "FAQs"])


# --- HOME TAB ---
with tab1:
    st.title("Welcome to pPred")
    st.write(
        """
**pPred** is a machine learning-powered tool for predicting inhibitors of the **PD-1/PD-L1** immune checkpoint pathway.
It accepts SMILES strings to assess molecular activity using trained models.

**Developer**: Erica Akanko  
**Email**: eakank001@gmail.com
"""
    )


# --- PREDICT TAB ---
with tab2:
    st.header("Make a Prediction")

    left, right = st.columns([1, 2])
    with left:
        selected_model_name = st.selectbox("Select a prediction model:", list(models.keys()))
        model = models[selected_model_name]

        option = st.radio("Choose input method:", ["Input SMILES", "Upload SMILES File"])

    # small style helper
    st.markdown(
        """
        <style>
        .highlight-box {
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if option == "Input SMILES":
        smiles_input = st.text_input("Enter a SMILES string:", placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O")

        if smiles_input:
            y_hat, conf, mol, X_sel, err = predict_single(smiles_input, model, selector)
            if err:
                st.error(err)
            else:
                # Molecule depiction
                st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Chemical Structure")

                label = "Active" if y_hat == 1 else "Inactive"
                with st.container():
                    st.markdown(
                        f"""
                        <div class='highlight-box'>
                            <h4>Prediction Result</h4>
                            <p><strong>Model:</strong> {selected_model_name}</p>
                            <p><strong>Prediction:</strong> {label}</p>
                            <p><strong>Confidence Score:</strong> {conf:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Applicability Domain: place query vs train/test
                try:
                    fig = build_applicability_plot(X_train, X_test, X_sel)
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"Applicability plot unavailable: {e}")

    else:  # Upload SMILES File
        file = st.file_uploader(
            "Upload a CSV or Excel file with a 'SMILES' column",
            type=["csv", "xls", "xlsx"],
        )
        if file:
            # Load dataframe
            try:
                if file.name.lower().endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                df = None

            if df is not None:
                if "SMILES" not in df.columns:
                    st.error("File must contain a 'SMILES' column.")
                else:
                    results = []
                    for smi in df["SMILES"].astype(str).fillna(""):
                        if not smi.strip():
                            results.append((smi, "Invalid", None))
                            continue
                        y_hat, conf, mol, _Xsel, err = predict_single(smi, model, selector)
                        if err:
                            results.append((smi, "Invalid", None))
                        else:
                            label = "Active" if y_hat == 1 else "Inactive"
                            results.append((smi, label, round(conf, 3)))

                    result_df = pd.DataFrame(results, columns=["SMILES", "Prediction", "Confidence"])
                    st.markdown("### Batch Prediction Results")
                    st.dataframe(result_df, use_container_width=True)

                    # Download
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "predictions.csv",
                        "text/csv",
                    )


# --- TUTORIAL TAB ---
with tab3:
    st.header("How to Use pPred")
    st.markdown(
        """
1. Go to the **Predict** tab.  
2. Select a model and input method.  
3. Provide a SMILES string or upload a file.  
4. View predictions and download your results.
"""
    )


# --- FAQ TAB ---
with tab4:
    st.subheader("FAQs")

    st.markdown("**Q1: How does pPred work?**")
    st.write(
        "pPred uses multiple machine learning models trained on known PD-1/PD-L1 inhibitors "
        "to predict the inhibitory activity of compounds based on their molecular fingerprints."
    )

    st.markdown("**Q2: Is pPred free to use?**")
    st.write("Yes. pPred is completely free and open to all users.")

    st.markdown("**Q3: What kind of data do I need to provide to use pPred?**")
    st.write("No personal data is required. You only need to provide valid SMILES strings for the molecules you wish to analyze.")

    st.markdown("**Q4: Is pPred designed for professionals only?**")
    st.write("Not at all. Anyone with SMILES data—students, researchers, or hobbyists—can use pPred to explore potential inhibitory properties.")

    st.markdown("**Q5: How accurate are the predictions made by pPred?**")
    st.write(
        "Accuracy depends on the model and chemical space. Confidence scores indicate how certain the model is about each prediction."
    )

    st.markdown("**Q6: How can I provide feedback or report issues with pPred?**")
    st.write("If you have suggestions, feedback, or encounter issues, please contact Erica Akanko at **eakank001@gmail.com**.")

    st.markdown("---")
    st.subheader("Glossary of Terms")
    st.markdown(
        """
- **PD-1/PD-L1**: Immune checkpoint proteins involved in regulating immune responses.  
- **SMILES**: A text-based format for representing molecular structures.  
- **Random Forest, KNN, AdaBoost, Extra Trees, Gradient Boosting**: Common machine learning algorithms for classification.  
- **Applicability Domain**: The chemical space where a model's predictions are considered reliable.
"""
    )
# -----------------------------