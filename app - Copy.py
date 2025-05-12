import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import numpy as np
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
model = joblib.load('rf.pkl')

# Load training fingerprints for applicability domain plot
train_fps = pd.read_csv('bioactivity_data_descriptors_morgan.csv')

# Function to generate fingerprint from SMILES
def featurize(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array(fp)

# Streamlit interface
st.title("SMILES Activity Predictor & Applicability Domain")

# Input SMILES string
smiles_input = st.text_input("Enter a SMILES string:")

# Process when user enters SMILES
if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        st.subheader("Chemical Structure")
        st.image(Draw.MolToImage(mol, size=(300, 300)))

        # Featurize and predict
        fp = featurize(smiles_input)
        if fp is not None:
            prob = model.predict_proba([fp])[0]
            pred_class = model.predict([fp])[0]
            st.subheader("Prediction Result")
            st.write(f"**Prediction:** {'Active' if pred_class == 1 else 'Inactive'}")
            st.write(f"**Confidence Score:** {np.max(prob) * 100:.2f}%")

            # Applicability domain
            st.subheader("Applicability Domain (PCA Projection)")
            pca = PCA(n_components=2)
            pca_train = pca.fit_transform(train_fps)
            pca_test = pca.transform([fp])

            plt.figure(figsize=(6, 5))
            plt.scatter(pca_train[:, 0], pca_train[:, 1], alpha=0.3, label='Training Set')
            plt.scatter(pca_test[0, 0], pca_test[0, 1], color='red', label='Query Compound')
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
            plt.title("Applicability Domain")
            plt.legend()
            st.pyplot(plt)
        else:
            st.error("Could not generate fingerprint from the SMILES.")
    else:
        st.error("Invalid SMILES string.")