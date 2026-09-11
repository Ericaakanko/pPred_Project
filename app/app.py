"""
pPred — PD-1/PD-L1 Inhibitor Predictor
A Streamlit web application for screening PD-1/PD-L1 inhibitors using
machine learning models trained on ChEMBL bioactivity data.

Supports both classification (Active/Inactive) and regression (pIC50) pipelines.

Developer: Erica Akanko
Email: eakank001@gmail.com
"""

from pathlib import Path
import io
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# -----------------------------
# Constants (must match notebook)
# -----------------------------
RADIUS = 2
N_BITS = 2048
ACTIVE_THRESHOLD = 1000  # nM

FP_COLUMNS = [f"FP_{i}" for i in range(N_BITS)]
DESC_COLUMNS = [
    "MW", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "FractionCSP3", "NumHeavyAtoms", "NumRings", "BalabanJ",
    "BertzCT", "MolMR", "LabuteASA", "NumValenceElectrons", "QED",
]
ALL_COLUMNS = FP_COLUMNS + DESC_COLUMNS

# Teal + Slate theme colors
TEAL = "#0D9488"
TEAL_LIGHT = "#14B8A6"
TEAL_BG = "#CCFBF1"
SLATE = "#1E293B"
SLATE_LIGHT = "#334155"
SLATE_BG = "#F1F5F9"
GREEN = "#16A34A"
GREEN_BG = "#DCFCE7"
RED = "#DC2626"
RED_BG = "#FEE2E2"
AMBER = "#D97706"
AMBER_BG = "#FEF3C7"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="pPred — PD-1/PD-L1 Inhibitor Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(f"""
<style>
    /* Global */
    .stApp {{ background-color: {SLATE_BG}; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {SLATE} 0%, {SLATE_LIGHT} 100%);
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stText {{
        color: #E2E8F0;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {TEAL_LIGHT};
    }}

    /* Metric cards */
    .metric-card {{
        background: white;
        border-left: 4px solid {TEAL};
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .metric-card h4 {{
        margin: 0 0 4px 0;
        color: {SLATE};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-card .value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {TEAL};
        margin: 0;
    }}
    .metric-card .sub {{
        font-size: 0.8rem;
        color: {SLATE_LIGHT};
        margin: 2px 0 0 0;
    }}

    /* Prediction result boxes */
    .pred-active {{
        background: {GREEN_BG};
        border: 1px solid {GREEN};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }}
    .pred-inactive {{
        background: {RED_BG};
        border: 1px solid {RED};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }}
    .pred-active h3, .pred-inactive h3 {{
        margin: 0 0 8px 0;
    }}
    .pred-active h3 {{ color: {GREEN}; }}
    .pred-inactive h3 {{ color: {RED}; }}

    /* AD badges */
    .ad-badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .ad-in-domain {{
        background: {GREEN_BG};
        color: {GREEN};
        border: 1px solid {GREEN};
    }}
    .ad-borderline {{
        background: {AMBER_BG};
        color: {AMBER};
        border: 1px solid {AMBER};
    }}
    .ad-out-domain {{
        background: {RED_BG};
        color: {RED};
        border: 1px solid {RED};
    }}

    /* Info card */
    .info-card {{
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .info-card h3 {{
        color: {TEAL};
        margin: 0 0 12px 0;
        font-size: 1.1rem;
    }}

    /* Feature grid */
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }}
    .feature-item {{
        background: white;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }}
    .feature-item .icon {{
        font-size: 1.8rem;
        margin-bottom: 6px;
    }}
    .feature-item .title {{
        font-weight: 600;
        color: {SLATE};
        font-size: 0.9rem;
    }}
    .feature-item .desc {{
        font-size: 0.78rem;
        color: {SLATE_LIGHT};
        margin-top: 4px;
    }}

    /* Pass/Fail badges */
    .pass-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background: {GREEN_BG};
        color: {GREEN};
    }}
    .fail-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background: {RED_BG};
        color: {RED};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {TEAL};
        color: {TEAL};
    }}

    /* Progress bar */
    .stProgress > div > div {{
        background-color: {TEAL};
    }}

    /* Download button */
    .stDownloadButton > button {{
        background-color: {TEAL};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
    }}
    .stDownloadButton > button:hover {{
        background-color: {SLATE};
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_logo():
    path = ASSETS_DIR / "ppred_logo.png"
    if path.exists():
        return Image.open(path)
    return None


@st.cache_resource(show_spinner=True)
def load_classifier_model():
    with open(MODELS_DIR / "classifier_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=True)
def load_regressor_model():
    with open(MODELS_DIR / "regressor_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=True)
def load_selector():
    with open(MODELS_DIR / "feature_selector.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_train_smiles():
    with open(MODELS_DIR / "train_smiles.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_train_fps():
    """Pre-compute Morgan fingerprints for all training compounds (for Tanimoto AD)."""
    smiles_list = load_train_smiles()
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS))
    return fps


@st.cache_data(show_spinner=False)
def load_metrics_csv(filename):
    path = MODELS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


# -----------------------------
# Core functions
# -----------------------------
def standardize_smiles(smi):
    """Standardize: largest fragment, neutralize charges, canonicalize."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol)


def compute_features(smiles):
    """
    Compute Morgan FP (2048 bits) + 17 RDKit descriptors.
    Returns (mol_obj, DataFrame with ALL_COLUMNS) or (None, None).
    """
    std_smi = standardize_smiles(smiles)
    if std_smi is None:
        return None, None
    mol = Chem.MolFromSmiles(std_smi)
    if mol is None:
        return None, None

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    fp_arr = np.zeros((N_BITS,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    # Physicochemical descriptors
    desc_vals = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.MolMR(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.NumValenceElectrons(mol),
        QED.qed(mol),
    ]

    features = np.hstack([fp_arr, np.array(desc_vals)])
    df = pd.DataFrame([features], columns=ALL_COLUMNS)
    return mol, df


def compute_tanimoto_ad(mol, train_fps):
    """Compute k-NN Tanimoto applicability domain. Returns (nn_sim, tier)."""
    if mol is None or not train_fps:
        return None, "N/A"
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    nn_sim = max(sims)
    if nn_sim >= 0.4:
        tier = "In-domain"
    elif nn_sim >= 0.3:
        tier = "Borderline"
    else:
        tier = "Out-of-domain"
    return nn_sim, tier


def compute_druglikeness(mol):
    """Compute drug-likeness properties. Returns dict."""
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    qed_val = QED.qed(mol)

    lipinski_pass = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
    veber_pass = n_rot <= 10 and tpsa <= 140

    return {
        "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba,
        "NumRotatableBonds": n_rot, "TPSA": tpsa, "QED": qed_val,
        "Lipinski_pass": lipinski_pass, "Veber_pass": veber_pass,
    }


@st.cache_resource(show_spinner=False)
def load_pains_patterns():
    """Load PAINS SMARTS patterns from RDKit."""
    pains_path = os.path.join(os.path.dirname(Chem.__file__), "Pains", "pains_pains.txt")
    patterns = []
    if os.path.exists(pains_path):
        with open(pains_path) as f:
            smarts = [line.strip().split(" ")[0] for line in f if line.strip()]
        patterns = [Chem.MolFromSmarts(s) for s in smarts if Chem.MolFromSmarts(s)]
    return patterns


def check_pains(mol):
    """Check for PAINS substructures. Returns (has_pains, matched_atoms)."""
    patterns = load_pains_patterns()
    if mol is None or not patterns:
        return False, None
    for pattern in patterns:
        if mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            return True, matches[0] if matches else None
    return False, None


def predict_compound(smiles, clf_model, reg_model, selector, train_fps):
    """
    Full prediction pipeline for a single SMILES.
    Returns dict with all results or None if invalid.
    """
    mol, feat_df = compute_features(smiles)
    if mol is None:
        return None

    # Apply feature selection
    feat_sel = selector.transform(feat_df)

    # Classification
    proba = clf_model.predict_proba(feat_sel)[0, 1]
    clf_pred = "Active" if proba >= 0.5 else "Inactive"

    # Regression
    pic50 = float(reg_model.predict(feat_sel)[0])

    # Applicability domain
    nn_sim, ad_tier = compute_tanimoto_ad(mol, train_fps)

    # Drug-likeness
    dl = compute_druglikeness(mol)

    # PAINS
    has_pains, _ = check_pains(mol)

    return {
        "mol": mol,
        "std_smiles": Chem.MolToSmiles(mol),
        "clf_pred": clf_pred,
        "clf_proba": proba,
        "pic50": pic50,
        "nn_sim": nn_sim,
        "ad_tier": ad_tier,
        "druglikeness": dl,
        "pains": has_pains,
    }


def predict_batch(smiles_list, clf_model, reg_model, selector, train_fps):
    """Batch prediction. Returns DataFrame."""
    results = []
    for smi in smiles_list:
        smi = str(smi).strip()
        if not smi:
            results.append({
                "SMILES": smi, "Prediction": "Empty", "Probability": "",
                "Predicted_pIC50": "", "AD_Tier": "N/A", "NN_Similarity": "",
                "Lipinski": "", "Veber": "", "QED": "", "PAINS": "",
            })
            continue
        r = predict_compound(smi, clf_model, reg_model, selector, train_fps)
        if r is None:
            results.append({
                "SMILES": smi, "Prediction": "Invalid", "Probability": "",
                "Predicted_pIC50": "", "AD_Tier": "N/A", "NN_Similarity": "",
                "Lipinski": "", "Veber": "", "QED": "", "PAINS": "",
            })
        else:
            dl = r["druglikeness"]
            results.append({
                "SMILES": smi,
                "Prediction": r["clf_pred"],
                "Probability": f"{r['clf_proba']:.3f}",
                "Predicted_pIC50": f"{r['pic50']:.2f}",
                "AD_Tier": r["ad_tier"],
                "NN_Similarity": f"{r['nn_sim']:.3f}" if r["nn_sim"] is not None else "",
                "Lipinski": "Pass" if dl["Lipinski_pass"] else "Fail",
                "Veber": "Pass" if dl["Veber_pass"] else "Fail",
                "QED": f"{dl['QED']:.3f}",
                "PAINS": "Alert" if r["pains"] else "None",
            })
    return pd.DataFrame(results)


# -----------------------------
# Load all resources
# -----------------------------
logo = load_logo()
clf_model = load_classifier_model()
reg_model = load_regressor_model()
selector = load_selector()
train_fps = load_train_fps()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    if logo is not None:
        st.image(logo, width=80)
    st.markdown(f"<h1 style='color:{TEAL_LIGHT}; margin:0;'>pPred</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:0.85rem; margin:0 0 16px 0;'>PD-1/PD-L1 Inhibitor Predictor</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Models")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.08); border-radius:8px; padding:12px; margin:8px 0;'>
        <p style='color:#E2E8F0; margin:0; font-size:0.85rem;'>
            <strong style='color:{TEAL_LIGHT};'>Classifier:</strong> ExtraTrees<br>
            <strong style='color:{TEAL_LIGHT};'>Regressor:</strong> Stacking Ensemble
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Dataset")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.08); border-radius:8px; padding:12px; margin:8px 0;'>
        <p style='color:#E2E8F0; margin:0; font-size:0.85rem;'>
            4,399 compounds<br>
            3 ChEMBL targets pooled<br>
            Active threshold: ≤{ACTIVE_THRESHOLD} nM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <p style='color:#64748B; font-size:0.75rem;'>
        Developer: Erica Akanko<br>
        Email: eakank001@gmail.com
    </p>
    """, unsafe_allow_html=True)


# -----------------------------
# Helper: metric card
# -----------------------------
def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <h4>{label}</h4>
        <p class="value">{value}</p>
        <p class="sub">{sub}</p>
    </div>
    """, unsafe_allow_html=True)


def ad_badge(tier):
    cls = {"In-domain": "ad-in-domain", "Borderline": "ad-borderline", "Out-of-domain": "ad-out-domain"}.get(tier, "")
    return f'<span class="ad-badge {cls}">{tier}</span>'


def pass_fail_badge(passed, label="Pass"):
    return f'<span class="pass-badge">{label}</span>' if passed else f'<span class="fail-badge">Fail</span>'


# -----------------------------
# Tabs
# -----------------------------
tab_home, tab_predict, tab_batch, tab_performance, tab_tutorial, tab_faq = st.tabs([
    "Home", "Single Prediction", "Batch Screening", "Model Performance", "Tutorial", "FAQs"
])


# ===== HOME TAB =====
with tab_home:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {SLATE} 0%, {SLATE_LIGHT} 100%); border-radius:16px; padding:32px; margin-bottom:24px;'>
        <h1 style='color:white; margin:0 0 8px 0;'>pPred</h1>
        <p style='color:#94A3B8; font-size:1.1rem; margin:0 0 16px 0;'>
            Machine learning-powered screening of PD-1/PD-L1 immune checkpoint inhibitors
        </p>
        <p style='color:#CBD5E1; font-size:0.9rem; margin:0;'>
            Trained on 4,399 compounds from 3 ChEMBL targets | ExtraTrees classifier (AUC 0.92) | Stacking regressor (R² 0.86)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Features")
    st.markdown(f"""
    <div class="feature-grid">
        <div class="feature-item">
            <div class="icon">🎯</div>
            <div class="title">Dual Prediction</div>
            <div class="desc">Active/Inactive classification + continuous pIC50 regression</div>
        </div>
        <div class="feature-item">
            <div class="icon">🛡️</div>
            <div class="title">Applicability Domain</div>
            <div class="desc">k-NN Tanimoto similarity with 3-tier reliability assessment</div>
        </div>
        <div class="feature-item">
            <div class="icon">💊</div>
            <div class="title">Drug-likeness</div>
            <div class="desc">Lipinski Ro5, Veber rules, QED score evaluation</div>
        </div>
        <div class="feature-item">
            <div class="icon">⚠️</div>
            <div class="title">PAINS Alerts</div>
            <div class="desc">Pan-assay interference compound substructure screening</div>
        </div>
        <div class="feature-item">
            <div class="icon">📊</div>
            <div class="title">Batch Screening</div>
            <div class="desc">CSV, SDF, or pasted SMILES — process hundreds at once</div>
        </div>
        <div class="feature-item">
            <div class="icon">📈</div>
            <div class="title">Model Performance</div>
            <div class="desc">CV metrics, scaffold test results, Y-scrambling, permutation tests</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Performance Highlights")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Classifier AUC", "0.918", "Scaffold test set")
    with col2:
        metric_card("Classifier MCC", "0.624", "Scaffold test set")
    with col3:
        metric_card("Regressor R²", "0.864", "Scaffold test set")
    with col4:
        metric_card("Regressor RMSE", "0.615", "pIC50 units")

    st.markdown("### Quick Start")
    st.markdown("""
    1. Go to the **Single Prediction** tab to screen one compound, or **Batch Screening** for multiple.
    2. Enter a SMILES string or upload a file.
    3. Review the prediction, applicability domain, and drug-likeness assessment.
    4. Download results from the batch screening tab.
    """)

    st.markdown("### Dataset Overview")
    st.markdown("""
    The models were trained on IC50 bioactivity data pooled from three ChEMBL targets:
    - **CHEMBL3580522** — PD-L1 single protein (739 compounds)
    - **CHEMBL4523993** — PD-1/PD-L1 complex (3,256 compounds)
    - **CHEMBL6066575** — PD-L1/2 family (427 compounds)

    After deduplication and SMILES standardization, 4,399 unique compounds were used.
    Features: Morgan fingerprints (2048 bits) + 17 physicochemical descriptors.
    """)


# ===== SINGLE PREDICTION TAB =====
with tab_predict:
    st.header("Single Compound Prediction")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Input")
        example_smiles = {
            "BMS-8 analog (PD-L1 inhibitor)": "Cc1ccc(C(=O)Nc2ccc3c(c2)OCCO3)cc1",
            "ChEMBL PD-L1 active": "COc1cc(OCc2cccc(-c3ccccc3)c2C)cc(OC)c1CN1CCCCC1",
            "Ethanol (non-inhibitor)": "CCO",
            "Caffeine (non-inhibitor)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        }
        example_choice = st.selectbox("Load example:", ["Custom input"] + list(example_smiles.keys()))
        default_smi = example_smiles.get(example_choice, "")
        smiles_input = st.text_input("Enter SMILES:", value=default_smi, placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O")

    if smiles_input.strip():
        result = predict_compound(smiles_input, clf_model, reg_model, selector, train_fps)

        if result is None:
            st.error("Invalid SMILES string. Please check the input and try again.")
        else:
            with col_right:
                # Molecule structure
                st.markdown("#### Structure")
                img = Draw.MolToImage(result["mol"], size=(300, 300))
                st.image(img, width=250)

                st.caption(f"Standardized: `{result['std_smiles']}`")

                # Classification result
                st.markdown("#### Prediction Results")
                col_a, col_b = st.columns(2)

                with col_a:
                    if result["clf_pred"] == "Active":
                        st.markdown(f"""
                        <div class="pred-active">
                            <h3>✓ Active</h3>
                            <p style='margin:0; font-size:1.1rem;'>
                                P(active) = <strong>{result['clf_proba']:.3f}</strong>
                            </p>
                            <p style='margin:4px 0 0 0; color:#666; font-size:0.85rem;'>
                                Threshold: IC50 ≤ {ACTIVE_THRESHOLD} nM
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pred-inactive">
                            <h3>✗ Inactive</h3>
                            <p style='margin:0; font-size:1.1rem;'>
                                P(active) = <strong>{result['clf_proba']:.3f}</strong>
                            </p>
                            <p style='margin:4px 0 0 0; color:#666; font-size:0.85rem;'>
                                Threshold: IC50 > {ACTIVE_THRESHOLD} nM
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                with col_b:
                    pic50 = result["pic50"]
                    ad = result["ad_tier"]
                    color = GREEN if ad == "In-domain" else (AMBER if ad == "Borderline" else RED)
                    st.markdown(f"""
                    <div class="info-card" style="border-left:4px solid {color};">
                        <h3 style="color:{color};">Predicted pIC50</h3>
                        <p style='font-size:1.8rem; font-weight:700; color:{SLATE}; margin:0;'>
                            {pic50:.2f}
                        </p>
                        <p style='margin:4px 0 0 0; color:#666; font-size:0.85rem;'>
                            IC50 ≈ {10**(-pic50) * 1e9:.0f} nM
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Applicability Domain
                st.markdown("#### Applicability Domain")
                nn = result["nn_sim"]
                ad_tier = result["ad_tier"]
                col_ad1, col_ad2 = st.columns([1, 2])

                with col_ad1:
                    st.markdown(f"""
                    <div style='text-align:center; padding:12px;'>
                        {ad_badge(ad_tier)}
                        <p style='margin:8px 0 0 0; font-size:0.85rem; color:{SLATE_LIGHT};'>
                            NN Tanimoto: <strong>{nn:.3f}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_ad2:
                    # Tanimoto gauge bar
                    fig, ax = plt.subplots(figsize=(4, 0.8))
                    ax.barh([0], [1], color="#E2E8F0", height=0.5)
                    ax.barh([0], [min(nn, 1.0)], color=color, height=0.5)
                    ax.axvline(x=0.3, color=AMBER, linestyle="--", linewidth=1)
                    ax.axvline(x=0.4, color=GREEN, linestyle="--", linewidth=1)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks([])
                    ax.set_xlabel("Tanimoto Similarity", fontsize=9)
                    ax.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Drug-likeness
                st.markdown("#### Drug-likeness Assessment")
                dl = result["druglikeness"]
                col_dl1, col_dl2, col_dl3 = st.columns(3)

                with col_dl1:
                    lip_pass = dl["Lipinski_pass"]
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Lipinski Ro5</h3>
                        <p style='margin:4px 0; font-size:0.85rem;'>MW: {dl['MW']:.1f} Da</p>
                        <p style='margin:4px 0; font-size:0.85rem;'>LogP: {dl['LogP']:.2f}</p>
                        <p style='margin:4px 0; font-size:0.85rem;'>HBD: {dl['HBD']} | HBA: {dl['HBA']}</p>
                        <p style='margin:8px 0 0 0;'>{pass_fail_badge(lip_pass)}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_dl2:
                    veb_pass = dl["Veber_pass"]
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Veber Rules</h3>
                        <p style='margin:4px 0; font-size:0.85rem;'>Rotatable bonds: {dl['NumRotatableBonds']}</p>
                        <p style='margin:4px 0; font-size:0.85rem;'>TPSA: {dl['TPSA']:.1f} Å²</p>
                        <p style='margin:8px 0 0 0;'>{pass_fail_badge(veb_pass)}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_dl3:
                    qed = dl["QED"]
                    qed_color = GREEN if qed >= 0.5 else (AMBER if qed >= 0.3 else RED)
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>QED Score</h3>
                        <p style='font-size:1.8rem; font-weight:700; color:{qed_color}; margin:4px 0;'>{qed:.3f}</p>
                        <p style='margin:4px 0 0 0; font-size:0.85rem; color:{SLATE_LIGHT};'>
                            {"Good" if qed >= 0.5 else ("Moderate" if qed >= 0.3 else "Poor")} drug-likeness
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # PAINS
                st.markdown("#### PAINS Alert")
                if result["pains"]:
                    st.markdown(f"""
                    <div class="pred-inactive">
                        <h3>⚠ PAINS Alert Detected</h3>
                        <p style='margin:0; font-size:0.9rem;'>
                            This compound contains a pan-assay interference substructure.
                            Predictions should be interpreted with caution.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-active">
                        <h3>✓ No PAINS Alerts</h3>
                        <p style='margin:0; font-size:0.9rem;'>
                            No pan-assay interference substructures detected.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        with col_right:
            st.info("Enter a SMILES string to see the prediction results.")


# ===== BATCH SCREENING TAB =====
with tab_batch:
    st.header("Batch Screening")

    input_method = st.radio("Select input method:", ["Upload CSV", "Upload SDF", "Paste SMILES"], horizontal=True)

    smiles_list = None

    if input_method == "Upload CSV":
        file = st.file_uploader("Upload a CSV file with a 'SMILES' column", type=["csv"])
        if file is not None:
            try:
                df = pd.read_csv(file)
                smi_col = None
                for col in df.columns:
                    if col.upper() == "SMILES":
                        smi_col = col
                        break
                if smi_col is None:
                    st.error("No 'SMILES' column found in the CSV file.")
                else:
                    smiles_list = df[smi_col].astype(str).fillna("").tolist()
                    st.success(f"Loaded {len(smiles_list)} compounds from column '{smi_col}'.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    elif input_method == "Upload SDF":
        file = st.file_uploader("Upload an SDF file", type=["sdf", "mol"])
        if file is not None:
            try:
                suppl = Chem.SDMolSupplier(file.getvalue())
                smiles_list = [Chem.MolToSmiles(m) for m in suppl if m is not None]
                st.success(f"Loaded {len(smiles_list)} valid molecules from SDF.")
            except Exception as e:
                st.error(f"Error reading SDF: {e}")

    elif input_method == "Paste SMILES":
        pasted = st.text_area("Paste SMILES (one per line):", height=150,
                              placeholder="Cc1ccc(C(=O)Nc2ccc3c(c2)OCCO3)cc1\nCCO\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        if pasted.strip():
            smiles_list = [s.strip() for s in pasted.strip().split("\n") if s.strip()]
            st.info(f"{len(smiles_list)} SMILES strings detected.")

    if smiles_list and len(smiles_list) > 0:
        if st.button("Run Screening", type="primary"):
            with st.spinner("Screening compounds..."):
                results_df = predict_batch(smiles_list, clf_model, reg_model, selector, train_fps)

            # Summary stats
            st.markdown("### Summary")
            n_total = len(results_df)
            n_valid = len(results_df[results_df["Prediction"].isin(["Active", "Inactive"])])
            n_active = len(results_df[results_df["Prediction"] == "Active"])
            n_in_domain = len(results_df[results_df["AD_Tier"] == "In-domain"])
            n_pains = len(results_df[results_df["PAINS"] == "Alert"])

            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            with col_s1:
                metric_card("Total", str(n_total), "compounds")
            with col_s2:
                metric_card("Valid", str(n_valid), f"{n_total - n_valid} invalid")
            with col_s3:
                metric_card("Active", str(n_active), f"{n_active/max(n_valid,1):.0%} of valid")
            with col_s4:
                metric_card("In-domain", str(n_in_domain), f"{n_in_domain/max(n_valid,1):.0%} of valid")
            with col_s5:
                metric_card("PAINS Alerts", str(n_pains), f"{n_pains/max(n_valid,1):.0%} of valid")

            # Results table
            st.markdown("### Results")
            st.dataframe(results_df, use_container_width=True, height=400)

            # Download
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                csv_data,
                "ppred_screening_results.csv",
                "text/csv",
            )


# ===== MODEL PERFORMANCE TAB =====
with tab_performance:
    st.header("Model Performance")

    pipeline = st.radio("Select pipeline:", ["Classifier", "Regressor"], horizontal=True)

    if pipeline == "Classifier":
        cv_df = load_metrics_csv("classifier_cv_metrics.csv")
        test_df = load_metrics_csv("classifier_test_results.csv")

        st.markdown("### Cross-Validation Summary (3×5 = 15 folds)")
        if cv_df is not None:
            st.dataframe(cv_df, use_container_width=True)

        st.markdown("### Scaffold-Based Test Set Results")
        if test_df is not None:
            st.dataframe(test_df, use_container_width=True)

            # Bar chart: ROC AUC comparison
            st.markdown("### ROC AUC Comparison (Scaffold Test Set)")
            fig, ax = plt.subplots(figsize=(10, 5))
            auc_col = "ROC_AUC" if "ROC_AUC" in test_df.columns else test_df.columns[6]
            test_sorted = test_df.sort_values(auc_col, ascending=True)
            colors = [TEAL if v >= 0.9 else (AMBER if v >= 0.85 else SLATE_LIGHT) for v in test_sorted[auc_col]]
            ax.barh(test_sorted["Model"], test_sorted[auc_col], color=colors)
            ax.set_xlabel("ROC AUC", fontsize=11)
            ax.set_title("Classifier ROC AUC on Scaffold Test Set", fontsize=13)
            ax.axvline(x=0.9, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Validation results
        st.markdown("### Validation Tests")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(f"""
            <div class="info-card">
                <h3>Y-Scrambling Test</h3>
                <p style='margin:4px 0; font-size:0.9rem;'>Real AUC: <strong>0.946</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>Scrambled AUC: <strong>0.495 ± 0.015</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>Z-score: <strong>29.47</strong></p>
                <p style='margin:8px 0 0 0;'>{pass_fail_badge(True, "PASS")}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_v2:
            st.markdown(f"""
            <div class="info-card">
                <h3>Permutation Test</h3>
                <p style='margin:4px 0; font-size:0.9rem;'>Real AUC: <strong>0.946</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>Permuted AUC: <strong>0.500 ± 0.016</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>p-value: <strong>&lt; 0.0001</strong></p>
                <p style='margin:8px 0 0 0;'>{pass_fail_badge(True, "Significant")}</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        cv_df = load_metrics_csv("regressor_cv_metrics.csv")
        test_df = load_metrics_csv("regressor_test_results.csv")

        st.markdown("### Cross-Validation Summary (3×5 = 15 folds)")
        if cv_df is not None:
            st.dataframe(cv_df, use_container_width=True)

        st.markdown("### Scaffold-Based Test Set Results")
        if test_df is not None:
            st.dataframe(test_df, use_container_width=True)

            # Bar chart: R² comparison
            st.markdown("### R² Comparison (Scaffold Test Set)")
            fig, ax = plt.subplots(figsize=(10, 5))
            r2_col = "R2" if "R2" in test_df.columns else test_df.columns[1]
            test_sorted = test_df.sort_values(r2_col, ascending=True)
            colors = [TEAL if v >= 0.85 else (AMBER if v >= 0.75 else SLATE_LIGHT) for v in test_sorted[r2_col]]
            ax.barh(test_sorted["Model"], test_sorted[r2_col], color=colors)
            ax.set_xlabel("R²", fontsize=11)
            ax.set_title("Regressor R² on Scaffold Test Set", fontsize=13)
            ax.axvline(x=0.85, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Validation results
        st.markdown("### Validation Tests")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(f"""
            <div class="info-card">
                <h3>Y-Scrambling Test</h3>
                <p style='margin:4px 0; font-size:0.9rem;'>Real R²: <strong>0.871</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>Scrambled R²: <strong>-0.072 ± 0.009</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>Z-score: <strong>106.17</strong></p>
                <p style='margin:8px 0 0 0;'>{pass_fail_badge(True, "PASS")}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_v2:
            st.markdown(f"""
            <div class="info-card">
                <h3>Permutation Test</h3>
                <p style='margin:4px 0; font-size:0.9rem;'>Real R²: <strong>0.871</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>Permuted R²: <strong>-0.072 ± 0.011</strong></p>
                <p style='margin:4px 0; font-size:0.9rem;'>p-value: <strong>&lt; 0.0001</strong></p>
                <p style='margin:8px 0 0 0;'>{pass_fail_badge(True, "Significant")}</p>
            </div>
            """, unsafe_allow_html=True)

    # Dataset info
    st.markdown("### Dataset Information")
    st.markdown(f"""
    <div class="info-card">
        <p style='margin:4px 0; font-size:0.9rem;'><strong>Compounds:</strong> 4,399 unique (after dedup + standardization)</p>
        <p style='margin:4px 0; font-size:0.9rem;'><strong>Targets:</strong> 3 ChEMBL targets pooled (CHEMBL3580522, CHEMBL4523993, CHEMBL6066575)</p>
        <p style='margin:4px 0; font-size:0.9rem;'><strong>Features:</strong> 2048 Morgan FP + 17 physicochemical descriptors → VarianceThreshold</p>
        <p style='margin:4px 0; font-size:0.9rem;'><strong>Split:</strong> Scaffold-based (Bemis-Murcko) + repeated 5-fold CV (3 repeats)</p>
        <p style='margin:4px 0; font-size:0.9rem;'><strong>Active threshold:</strong> IC50 ≤ {ACTIVE_THRESHOLD} nM (classifier)</p>
    </div>
    """, unsafe_allow_html=True)


# ===== TUTORIAL TAB =====
with tab_tutorial:
    st.header("How to Use pPred")

    st.markdown("### Single Compound Prediction")
    st.markdown("""
    1. Navigate to the **Single Prediction** tab.
    2. Either select an example compound from the dropdown or type a SMILES string.
    3. The app will display:
       - 2D molecular structure
       - **Classification**: Active/Inactive with probability score
       - **Regression**: Predicted pIC50 value (higher = more potent)
       - **Applicability Domain**: Tanimoto similarity to training set
       - **Drug-likeness**: Lipinski, Veber, and QED assessment
       - **PAINS**: Pan-assay interference compound alert
    """)

    st.markdown("### Batch Screening")
    st.markdown("""
    1. Navigate to the **Batch Screening** tab.
    2. Choose an input method:
       - **CSV**: Upload a file with a `SMILES` column (case-insensitive)
       - **SDF**: Upload a standard SDF molecular file
       - **Paste**: Type or paste SMILES strings, one per line
    3. Click **Run Screening** to process all compounds.
    4. Review the summary statistics and results table.
    5. Download results as CSV for further analysis.
    """)

    st.markdown("### Interpreting Results")

    st.markdown("#### Classification (Active/Inactive)")
    st.markdown("""
    - **Active**: The model predicts IC50 ≤ 1000 nM (probability ≥ 0.5)
    - **Inactive**: The model predicts IC50 > 1000 nM
    - **Probability**: Confidence score (0–1). Values near 0.5 indicate uncertainty.
    """)

    st.markdown("#### Regression (pIC50)")
    st.markdown("""
    - pIC50 = -log₁₀(IC50 in molar). Higher values indicate greater potency.
    - pIC50 6 = 1 μM, pIC50 7 = 100 nM, pIC50 8 = 10 nM
    - The app also shows the approximate IC50 in nM for convenience.
    """)

    st.markdown("#### Applicability Domain")
    st.markdown("""
    - **In-domain** (Tanimoto ≥ 0.4): Prediction is reliable — the compound is similar to training data.
    - **Borderline** (0.3–0.4): Use with caution — moderate similarity to training data.
    - **Out-of-domain** (< 0.3): Prediction may be unreliable — compound is outside the training chemical space.
    """)

    st.markdown("#### Drug-likeness")
    st.markdown("""
    - **Lipinski Ro5**: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10. Passing suggests oral bioavailability.
    - **Veber**: Rotatable bonds ≤ 10, TPSA ≤ 140 Å². Passing suggests good oral bioavailability.
    - **QED**: Quantitative Estimate of Drug-likeness (0–1). > 0.5 is considered good.
    """)

    st.markdown("#### PAINS Alerts")
    st.markdown("""
    PAINS (Pan-Assay Interference Compounds) are substructures known to cause false positives
    in biological assays. If a PAINS alert is triggered, the prediction should be interpreted
    with extra caution.
    """)

    st.markdown("### SMILES Format")
    st.markdown("""
    SMILES (Simplified Molecular-Input Line-Entry System) is a text representation of molecular structures.
    Examples:
    - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
    - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
    - Ethanol: `CCO`
    """)


# ===== FAQ TAB =====
with tab_faq:
    st.header("Frequently Asked Questions")

    with st.expander("Q1: How does pPred work?"):
        st.write("""
        pPred uses machine learning models trained on known PD-1/PD-L1 inhibitors from the ChEMBL database.
        The classifier predicts whether a compound is active (IC50 ≤ 1000 nM) or inactive,
        while the regressor predicts the continuous pIC50 value. Both models use Morgan molecular
        fingerprints and physicochemical descriptors as input features.
        """)

    with st.expander("Q2: What models are used?"):
        st.write("""
        The classifier uses an ExtraTrees model (best of 10 models tested, including Random Forest,
        XGBoost, LightGBM, SVM, and a Stacking ensemble). The regressor uses a Stacking ensemble
        combining LightGBM, XGBoost, and Gradient Boosting with a Ridge meta-learner.
        """)

    with st.expander("Q3: Is pPred free to use?"):
        st.write("Yes. pPred is completely free and open to all users.")

    with st.expander("Q4: What kind of data do I need to provide?"):
        st.write("""
        No personal data is required. You only need to provide valid SMILES strings
        for the molecules you wish to analyze.
        """)

    with st.expander("Q5: How accurate are the predictions?"):
        st.write("""
        On a scaffold-based test set (compounds with novel chemical scaffolds):
        - Classifier: ROC AUC = 0.918, MCC = 0.624
        - Regressor: R² = 0.864, RMSE = 0.615 pIC50 units

        The applicability domain assessment helps gauge reliability for each specific compound.
        Predictions for out-of-domain compounds should be treated as rough estimates only.
        """)

    with st.expander("Q6: What is the applicability domain?"):
        st.write("""
        The applicability domain (AD) defines the chemical space where the model's predictions
        are considered reliable. pPred uses k-NN Tanimoto similarity to the training set:
        compounds with high similarity to training data receive more reliable predictions.
        """)

    with st.expander("Q7: What are PAINS alerts?"):
        st.write("""
        PAINS (Pan-Assay Interference Compounds) are chemical substructures known to produce
        false-positive results across many biological assays. pPred flags compounds containing
        these substructures so users can interpret predictions with appropriate caution.
        """)

    with st.expander("Q8: How can I provide feedback or report issues?"):
        st.write("Please contact Erica Akanko at **eakank001@gmail.com**.")

    st.markdown("---")
    st.subheader("Glossary")
    st.markdown("""
    - **PD-1/PD-L1**: Immune checkpoint proteins that regulate T-cell immune responses. Blocking this pathway is a major cancer immunotherapy strategy.
    - **SMILES**: Simplified Molecular-Input Line-Entry System — a text format for representing molecular structures.
    - **pIC50**: Negative log of IC50 in molar units. Higher values indicate greater potency.
    - **Morgan fingerprint**: A binary molecular fingerprint encoding structural fragments around each atom.
    - **ExtraTrees**: Extremely Randomized Trees — an ensemble machine learning method using randomized decision trees.
    - **Stacking ensemble**: A meta-learning approach that combines predictions from multiple base models.
    - **Tanimoto similarity**: A measure of molecular similarity between two fingerprints (0–1, higher = more similar).
    - **Lipinski Ro5**: Rule of Five — guidelines for oral drug-likeness (MW, LogP, HBD, HBA thresholds).
    - **Veber rules**: Drug-likeness criteria based on rotatable bonds and polar surface area.
    - **QED**: Quantitative Estimate of Drug-likeness — a score from 0 to 1.
    - **PAINS**: Pan-Assay Interference Compounds — substructures that cause false positives in assays.
    - **Y-scrambling**: A validation test where labels are randomly shuffled to verify the model isn't learning spurious patterns.
    - **Permutation test**: A statistical test comparing model performance against random label permutations.
    - **Scaffold split**: A train/test split based on Bemis-Murcko scaffolds, ensuring test compounds have novel core structures.
    """)
