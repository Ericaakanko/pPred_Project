# 🧬 pPred: A Machine Learning Tool for PD-1/PD-L1 Inhibitor Prediction

**pPred** is an interactive web application built with **Streamlit** that predicts the inhibitory activity of compounds targeting the **PD-1/PD-L1 immune checkpoint pathway**.  
It uses multiple trained machine learning models to evaluate **SMILES strings** and return predictions with associated confidence scores.

---

## 🚀 Features

- 🔹 Predicts PD-1/PD-L1 inhibitor activity from SMILES strings  
- 🔹 Supports both **single** and **batch predictions**  
- 🔹 Visualizes chemical structures and model confidence  
- 🔹 Includes PCA-based **applicability domain plots**  
- 🔹 Easy-to-use **web interface** for researchers and students  

---

## 🧠 Underlying Models

The following models are integrated into **pPred**:
- Random Forest  
- K-Nearest Neighbors (KNN)  
- AdaBoost  
- Extra Trees  
- Gradient Boosting  

Each model was trained on molecular descriptors (Morgan fingerprints) derived from curated bioactivity data.

---

## 📦 Project Structure

```
ppred/
│
├── app.py                          # Main Streamlit app
├── bioactivity_data_descriptors_morgan.csv  # Molecular descriptor dataset
├── selection.pkl                   # Feature selector (e.g., selected fingerprint bits)
├── rf.pkl, knn.pkl, adaboost.pkl, et.pkl, gb.pkl  # Trained ML models
├── X_train.pkl, X_test.pkl         # Training and test sets
├── ppred_logo.png                  # App logo
└── README.md                       # Project documentation
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ppred.git
cd ppred
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

Launch the Streamlit app locally:
```bash
streamlit run app.py
```

Once it starts, open your browser and navigate to:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧰 Usage Guide

### 🔹 Single Prediction
1. Navigate to the **Predict** tab.  
2. Select a model (e.g., Random Forest).  
3. Enter a valid SMILES string (e.g., `CC(=O)Nc1ccc(O)cc1`).  
4. View the prediction result, confidence score, and structure visualization.

### 🔹 Batch Prediction
1. Upload a `.csv` or `.xlsx` file containing a column named **`SMILES`**.  
2. The app will generate predictions for all molecules.  
3. Download the results as a `.csv` file.

---

## 📊 Applicability Domain

pPred visualizes the **applicability domain** of each query molecule using **PCA** plots of the descriptor space, helping users understand whether a new compound lies within the model’s reliable prediction region.

---

## 🧪 Example Input

| SMILES | Prediction | Confidence |
|--------|-------------|-------------|
| CC(=O)Nc1ccc(O)cc1 | Active | 0.91 |
| CCCC(=O)O | Inactive | 0.35 |

---

## 💡 Generate `requirements.txt`

If not included, create it using:
```bash
pip freeze > requirements.txt
```

Typical dependencies:
```text
streamlit
rdkit-pypi
numpy
pandas
matplotlib
scikit-learn
pillow
joblib
```

---

## 🧩 Future Improvements

- Integration of additional ML and deep learning models  
- Deployment on cloud platforms (e.g., Streamlit Cloud or Hugging Face Spaces)  
- Expanded dataset to include more PD-1/PD-L1 inhibitors  

---

## 👩‍🔬 Author

**Erica Azechum Akanko**  
📧 Email: [eakanko15@gmail.com](mailto:eakanko15@gmail.com)

---

## ⭐ Acknowledgment

This project leverages open-source cheminformatics tools and machine learning libraries to advance computational drug discovery for cancer immunotherapy.

---

### 📖 License
This project is open-source under the [MIT License](LICENSE).

---

> 💬 “Empowering research through accessible machine learning tools.”
