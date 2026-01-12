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
# This ensures it finds the model folder whether in local or cloud root
MODEL_DIR = os.path.join(ROOT, SUBFOLDER, "model") if os.path.exists(os.path.join(ROOT, SUBFOLDER)) else os.path.join(ROOT, "model")

# --- 2. Optimized Asset Loading ---
@st.cache_resource
def get_assets(model_filename):
    """Loads both model and scaler safely in one cached call."""
    m_path = os.path.join(MODEL_DIR, model_filename)
    s_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    if os.path.exists(m_path) and os.path.exists(s_path):
        return joblib.load(m_path), joblib.load(s_path)
    return None, None

# --- 3. Sidebar Configuration ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Test Data")
    st.dataframe(df.head())

    target_col = st.sidebar.selectbox("Select Target (Diagnosis)", df.columns, index=len(df.columns)-1)
    
    st.sidebar.header("2. Model Selection")
    model_map = {
        "Logistic Regression": "breast_cancer_model_lr.pkl",
        "Decision Tree": "breast_cancer_model_dt.pkl",
        "XG Boost": "breast_cancer_model_xg.pkl"
    }
    
    model_option = st.sidebar.selectbox("Choose Trained Model", options=list(model_map.keys()))
    
    if st.button("Run Evaluation"):
        model, scaler = get_assets(model_map[model_option])

        if model is not None and scaler is not None:
            # We wrap the core computation in ONE try block to ensure syntax remains clean
            try:
                # --- 4. Data Preparation ---
                # Drop ID and Diagnosis to get only the 30 features
                to_drop = [c for c in df.columns if c.lower() in ['id', 'diagnosis']]
                X_raw = df.drop(columns=to_drop)
                
                # Encode labels
                le = LabelEncoder()
                y_true = le.fit_transform(df[target_col].astype(str))
                
                # Scale and Predict
                X_scaled = scaler.transform(X_raw) 
                y_pred = model.predict(X_scaled)
                
                # Handle probability safely without nested try/except if possible
                y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

                # --- 5. Metrics Display ---
                st.subheader(f"Results for {model_option}")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
                m2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.2f}")
                m3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.2f}")
                m4.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
                m5.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
                m6.metric("F1", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

                # --- 6. Visualizations ---
                st.write("---")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("#### Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='g', cmap='Purples', ax=ax)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(fig)
                    
                with col_b:
                    st.write("#### Classification Report")
                    report = classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

            except Exception as e:
                # This properly closes the 'try' and prevents the SyntaxError
                st.error(f"Computation Error: {e}")
                st.info("Check if your CSV features match the 30 features used in training.")
        else:
            st.error(f"Asset Error: Files not found in {MODEL_DIR}")
else:
    st.info("Please upload a CSV file to begin.")