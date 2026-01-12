import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="Breast Cancer Model Evaluator", layout="wide")

st.title("📊 Breast Cancer Classification Evaluator")
st.markdown("Upload a test dataset to evaluate your pre-trained models.")

# --- 1. Path Resolution Logic ---
ROOT = os.getcwd()
SUBFOLDER = "breast_cancer_classification_models"
# Logic to handle both local and Streamlit Cloud directory structures
MODEL_DIR = os.path.join(ROOT, SUBFOLDER, "model") if os.path.exists(os.path.join(ROOT, SUBFOLDER)) else os.path.join(ROOT, "model")

# --- 2. Optimized Asset Loading ---
@st.cache_resource
def get_scaler():
    """Loads the universal scaler using absolute path resolution."""
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def get_model(filename):
    """Loads the specific model from the correct directory."""
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path) if os.path.exists(path) else None

# --- 3. Sidebar Configuration ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Test Data")
    st.dataframe(df.head())

    # Dynamically select the target column
    target_col = st.sidebar.selectbox("Select Target (Diagnosis)", df.columns, index=len(df.columns)-1)
    
    st.sidebar.header("2. Model Selection")
    model_map = {
        "Logistic Regression": "breast_cancer_model_lr.pkl",
        "Decision Tree": "breast_cancer_model_dt.pkl",
        "XG Boost": "breast_cancer_model_xg.pkl"
    }
    
    model_option = st.sidebar.selectbox("Choose Trained Model", options=list(model_map.keys()))
    selected_filename = model_map[model_option]

    if st.button("Run Evaluation"):
        model = get_model(selected_filename)
        scaler = get_scaler()

        if model is not None and scaler is not None:
            try:
                # --- 4. Data Preparation & Prediction ---
                # FIX: Explicitly drop 'id' and 'diagnosis' columns to avoid feature name mismatch
                cols_to_ignore = [col for col in df.columns if col.lower() in ['id', 'diagnosis']]
                X_raw = df.drop(columns=cols_to_ignore)
                
                # Check for label existence for metric calculation
                le = LabelEncoder()
                y_true = le.fit_transform(df[target_col].astype(str))
                
                # Apply scaling and generate predictions
                X_scaled = scaler.transform(X_raw) 
                y_pred = model.predict(X_scaled)
                
                # Attempt probability prediction for AUC
                try:
                    y_proba = model.predict_proba(X_scaled)[:, 1]
                except AttributeError:
                    y_proba = y_pred 

                # --- 5. Metrics Display ---
                st.subheader(f"Results for {model_option}")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")