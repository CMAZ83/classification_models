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

# Page Configuration
st.set_page_config(page_title="Breast Cancer Model Evaluator", layout="wide")

st.title("📊 Breast Cancer Classification Evaluator")
st.markdown("Upload a test dataset to evaluate your pre-trained models.")

# --- 1. Optimized Asset Loading ---
@st.cache_resource
def get_scaler():
    """Loads the universal scaler once."""
    return joblib.load("model/scaler.pkl")

@st.cache_resource
def get_model(model_path):
    """Loads the specific model selected."""
    return joblib.load(model_path)

# --- 2. Sidebar Configuration ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Test Data")
    st.dataframe(df.head())

    target_col = st.sidebar.selectbox("Select Target (Diagnosis) Column", df.columns, index=len(df.columns)-1)
    
    st.sidebar.header("2. Model Selection")
    model_map = {
        "Logistic Regression": "model/breast_cancer_model_lr.pkl",
        "Decision Tree": "model/breast_cancer_model_dt.pkl",
        "XG Boost": "model/breast_cancer_model_xg.pkl"
    }
    
    # FIX: Changed .kevs() to .keys()
    model_option = st.sidebar.selectbox("Choose Trained Model", options=list(model_map.keys()))
    selected_filename = model_map[model_option]

    if st.button("Run Evaluation"):
        try:
            # Load assets using optimized functions
            model = get_model(selected_filename)
            scaler = get_scaler()

            # --- Data Preparation ---
            X_raw = df.drop(columns=[target_col])
            le = LabelEncoder()
            y_true = le.fit_transform(df[target_col].astype(str))
            
            # Scale features
            X_scaled = scaler.transform(X_raw)
            
            # Predictions
            y_pred = model.predict(X_scaled)
            
            try:
                y_proba = model.predict_proba(X_scaled)[:, 1]
            except:
                y_proba = y_pred 

            # --- Display Metrics ---
            st.subheader(f"Results for {model_option}")
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
            m2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.2f}")
            m3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.2f}")
            m4.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
            m5.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
            m6.metric("F1", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

            # --- Visualizations ---
            st.write("---")
            c1, c2 = st.columns(2)
            with c1:
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='g', cmap='Purples', ax=ax)
                st.pyplot(fig)
            with c2:
                st.write("#### Classification Report")
                report = classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

        except Exception as e:
            st.error(f"Error: {e}. Check if file paths and CSV features match.")
else:
    st.info("Please upload a CSV file to begin.")