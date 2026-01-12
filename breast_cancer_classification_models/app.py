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
st.markdown("Upload a test dataset to evaluate performance or predict outcomes.")

# Path Resolution for GitHub Codespaces / Streamlit Cloud
ROOT = os.getcwd()
SUB_FOLDER = "breast_cancer_classification_models"
MODEL_DIR = os.path.join(ROOT, SUB_FOLDER, "model") if os.path.exists(os.path.join(ROOT, SUB_FOLDER)) else os.path.join(ROOT, "model")

# --- Asset Loading (Cachable) ---
@st.cache_resource
def load_assets(model_file):
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    model_path = os.path.join(MODEL_DIR, model_file)
    if os.path.exists(scaler_path) and os.path.exists(model_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

# --- a. Dataset Upload Option [Requirement A] ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Test Data Preview")
    st.dataframe(df.head())

    # Check if Diagnosis exists for Evaluation Mode
    has_labels = "diagnosis" in [col.lower() for col in df.columns]
    
    # --- b. Model Selection Dropdown [Requirement B] ---
    st.sidebar.header("2. Model Selection")
    model_map = {
        "Logistic Regression": "breast_cancer_model_lr.pkl",
        "Decision Tree": "breast_cancer_model_dt.pkl",
        "XG Boost": "breast_cancer_model_xg.pkl"
    }
    model_option = st.sidebar.selectbox("Choose Model", options=list(model_map.keys()))

    if st.button("Run Model"):
        model, scaler = load_assets(model_map[model_option])

        if model is not None:
            try:
                # Data Prep: Handle features only (30 columns)
                # If diagnosis exists, drop it; otherwise, take all columns
                if has_labels:
                    target_col = [col for col in df.columns if col.lower() == "diagnosis"][0]
                    X = df.drop(columns=[target_col])
                    y_true = LabelEncoder().fit_transform(df[target_col].astype(str))
                else:
                    X = df
                
                # Inference
                X_scaled = scaler.transform(X)
                y_pred = model.predict(X_scaled)

                # --- c. Display Evaluation Metrics [Requirement C] ---
                if has_labels:
                    st.subheader(f"📈 Evaluation Metrics: {model_option}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
                    m2.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
                    m3.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
                    m4.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.2f}")

                    # --- d. Confusion Matrix & Report [Requirement D] ---
                    st.write("---")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("#### Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                    with c2:
                        st.write("#### Classification Report")
                        report = classification_report(y_true, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                else:
                    # Prediction Only Mode (if no diagnosis column exists)
                    st.success("Predictions Complete!")
                    results = pd.DataFrame({"Prediction": ["Malignant" if p == 1 else "Benign" for p in y_pred]})
                    st.dataframe(pd.concat([df, results], axis=1))

            except Exception as e:
                st.error(f"Computation Error: {e}")
        else:
            st.error(f"Model files not found in {MODEL_DIR}")
else:
    st.info("Upload a CSV file to begin.")