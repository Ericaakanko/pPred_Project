import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io

# Load logo
logo = Image.open("ppred_logo.png")
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=80)
with col2:
    st.markdown("<h1 style='margin-top: 20px;'>pPred</h1>", unsafe_allow_html=True)

# Load feature selector and training/test sets
selector = joblib.load("selection.pkl")
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")

# Load models
models = {
    "Random Forest": joblib.load("rf.pkl"),
    "K-Nearest Neighbors": joblib.load("knn.pkl"),
    "AdaBoost": joblib.load("adaboost.pkl"),
    "Extra Trees": joblib.load("et.pkl"),
    "Gradient Boosting": joblib.load("gb.pkl")
}

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Predict", "Tutorial", "FAQs"])

# --- HOME TAB ---
with tab1:
    st.title("Welcome to pPred")
    st.write("""
    **pPred** is a machine learning-powered tool for predicting inhibitors of the **PD-1/PD-L1** immune checkpoint pathway.
    It accepts SMILES strings to assess molecular activity using trained models.

    **Developer**: Erica Akanko  
    **Email**: eakank001@gmail.com
    """)

# --- PREDICT TAB ---
with tab2:
    st.header("Make a Prediction")
    selected_model_name = st.selectbox("Select a prediction model:", list(models.keys()))
    model = models[selected_model_name]
    option = st.radio("Choose input method:", ["Input SMILES", "Upload SMILES File"])

    def get_prediction(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None, None, None, "Invalid SMILES"
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp)
        fp_selected = selector.transform([fp_array])
        prob = model.predict_proba(fp_selected)[0]
        pred = model.predict(fp_selected)[0]
        return pred, prob[pred], mol, fp_selected

    if option == "Input SMILES":
        smiles_input = st.text_input("Enter a SMILES string:")
        if smiles_input:
            pred, conf, mol, fp_selected = get_prediction(smiles_input)
            if mol is None:
                st.error("Invalid SMILES.")
            else:
                st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Chemical Structure")
                label = "Active" if pred == 1 else "Inactive"

                with st.container():
                    st.markdown("""
                    <div class='highlight-box'>
                        <h4>Prediction Result</h4>
                        <p><strong>Prediction:</strong> {}</p>
                        <p><strong>Confidence Score:</strong> {:.2f}</p>
                    </div>
                    """.format(label, conf), unsafe_allow_html=True)

                # Applicability Domain plot
                X_all = np.vstack([X_train, X_test, fp_selected])
                y_all = np.concatenate([['train'] * len(X_train), ['test'] * len(X_test), ['query']])

                scaler = StandardScaler()
                X_std = scaler.fit_transform(X_all)

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_std)

                fig, ax = plt.subplots()
                for group, color in zip(['train', 'test', 'query'], ['blue', 'orange', 'red']):
                    idx = y_all == group
                    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, label=group, alpha=0.6, edgecolors='k')

                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                ax.set_title("Applicability Domain: PCA of Descriptor Space")
                ax.legend()
                st.pyplot(fig)

    elif option == "Upload SMILES File":
        file = st.file_uploader("Upload a CSV or Excel file with a 'SMILES' column", type=["csv", "xls", "xlsx"])
        if file:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            if "SMILES" not in df.columns:
                st.error("File must contain a 'SMILES' column.")
            else:
                results = []
                for smi in df["SMILES"]:
                    pred, conf, mol, _ = get_prediction(smi)
                    if mol is None:
                        results.append((smi, "Invalid", None))
                    else:
                        label = "Active" if pred == 1 else "Inactive"
                        results.append((smi, label, round(conf, 2)))

                result_df = pd.DataFrame(results, columns=["SMILES", "Prediction", "Confidence"])
                st.markdown("### Batch Prediction Results")
                st.dataframe(result_df.style.set_properties(**{'text-align': 'left'}).set_table_styles([
                    {'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]}]))

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
# --- TUTORIAL TAB ---
with tab3:
    st.header("How to Use pPred")
    st.markdown("""
    1. Go to the **Predict** tab.
    2. Select a model and input method.
    3. Provide a SMILES string or upload a file.
    4. View predictions and download your results.
    """)
# --- FAQ TAB ---
with tab4:
    st.subheader("FAQs")

    st.markdown("**Q1: How does pPred work?**")
    st.write("pPred uses multiple machine learning models trained on known PD-1/PD-L1 inhibitors to predict the inhibitory activity of compounds based on their molecular fingerprints.")

    st.markdown("**Q2: Is pPred free to use?**")
    st.write("Yes. pPred is completely free and open to all users.")

    st.markdown("**Q3: What kind of data do I need to provide to use pPred?**")
    st.write("No personal data is required. You only need to provide valid SMILES strings for the molecules you wish to analyze.")

    st.markdown("**Q4: Is pPred designed for professionals only?**")
    st.write("Not at all. Anyone with SMILES data—students, researchers, or hobbyists—can use pPred to explore potential inhibitory properties.")

    st.markdown("**Q5: How accurate are the predictions made by pPred?**")
    st.write("The models demonstrate high performance on the validation set, with confidence scores typically ranging between 0.80 and 1.00 for known inhibitors.")
    st.markdown("**Q6: How can I provide feedback or report issues with pPred?**")
    st.write("If you have suggestions, feedback, or encounter issues, please contact Erica Akanko at **eakank001@gmail.com**.")

    st.markdown("---")
    st.subheader("Glossary of Terms")
    st.markdown("""
    - **PD-1/PD-L1**: Immune checkpoint proteins involved in regulating immune responses.
    - **SMILES**: A text-based format for representing molecular structures.
    - **Random Forest, KNN, AdaBoost, Extra Trees, Gradient Boosting**: Different machine learning algorithms used to predict activity.
    - **Applicability Domain**: The chemical space where the model's predictions are considered reliable.
    """)


