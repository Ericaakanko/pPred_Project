import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# App title
st.title("Compound Activity Prediction")

# Input: SMILES string
smiles_input = st.text_input("Enter a SMILES string:")

# Load trained model and selector
model = joblib.load("rf.pkl")                 # RandomForestClassifier
selector = joblib.load("selection.pkl")       # VarianceThreshold

# Handle SMILES input
if smiles_input:
    try:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES string.")
        else:
            # Draw and display structure
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="Chemical Structure")

            # Generate 2048-bit Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = np.array(fp)

            # Apply feature selection
            fp_selected = selector.transform([fp_array])  # (1, 52)

            # Predict
            prob = model.predict_proba(fp_selected)[0]
            prediction = model.predict(fp_selected)[0]

            # Show prediction
            label = "Active" if prediction == 1 else "Inactive"
            st.markdown(f"**Prediction**: {label}")
            st.markdown(f"**Confidence Score**: {prob[prediction]:.2f}")

            # Optional: Applicability Domain Plot (placeholder)
            # You would need to define X_train and PCA/TSNE model used for AD
            # This is a mock example:
            # plt.scatter(...); st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
