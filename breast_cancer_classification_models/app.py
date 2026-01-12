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
st.set_page_config(page_title="Breast Cancer Evaluator", layout="wide")

st.title("📊 Breast Cancer Classification Evaluator")

# --- 1. Smart Path Resolution [Seasoned Developer Fix] ---
def find_model_dir():
    """Recursively searches for the 'model' directory starting from root."""
    for root, dirs, files in os.walk(os.getcwd()):
        if 'model' in dirs:
            return os.path.join(root, 'model')
    return os.path.join(os.getcwd(), 'model') # Fallback

MODEL_DIR = find_model_dir()

# --- 2. Safe Asset Loading ---
@st.cache_resource
def get_verified_assets(model_filename):
    """Verifies files exist and are valid model objects."""
    m_path = os.path.join(MODEL_DIR, model_filename)
    s_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    try:
        if os.path.exists(m_path) and os.path.exists(s_path):
            model_obj = joblib.load(m_path)
            scaler_obj = joblib.load(s_path)
            # Ensure it's a model with a predict method, not a string
            if hasattr(model_obj, "predict"):
                return model_obj, scaler_obj
        return None, None
    except Exception:
        return None, None

# --- 3. Sidebar ---
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
    model_option = st.sidebar.selectbox("Choose Model", options=list(model_map.keys()))
    
    if st.button("Run Evaluation"):
        model, scaler = get_verified_assets(model_map[model_option])

        if model is not None and scaler is not None:
            try:
                # --- 4. Data Preparation ---
                # Drop ID and Diagnosis to isolate 30 features
                to_drop = [c for c in df.columns if c.lower() in ['id', 'diagnosis']]
                X_raw = df.drop(columns=to_drop)
                
                # Check for feature count mismatch
                if X_raw.shape[1] != 30:
                    st.error(f"Error: Expected 30 features, but found {X_raw.shape[1]}.")
                    st.stop()

                y_true = LabelEncoder().fit_transform(df[target_col].astype(str))
                X_scaled = scaler.transform(X_raw) 
                y_pred = model.predict(X_scaled)
                y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

                # --- 5. Display ---
                st.subheader(f"Results for {model_option}")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
                m2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.2f}")
                m3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.2f}")
                m4.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
                m5.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
                m6.metric("F1", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

                st.write("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("#### Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='g', cmap='Purples', ax=ax)
                    st.pyplot(fig)
                with col_b:
                    st.write("#### Classification Report")
                    report = classification_report(y_true, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

            except Exception as e:
                st.error(f"Computation Error: {e}")
        else:
            st.error(f"Asset Error: Could not find files in {MODEL_DIR}")
            st.write("Available folders here:", os.listdir(os.getcwd()))
else:
    st.info("Please upload a CSV file to begin.")