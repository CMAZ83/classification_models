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

# --- 1. Load Models & Assets ---
# Using cache_resource to prevent reloading the model on every rerun
@st.cache_resource
def load_ml_assets(model_name):
    # Map selection to your filename
    model_path = f"{model_name}.pkl"
    scaler_path = "scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

# --- 2. Sidebar Configuration ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Test Data")
    st.dataframe(df.head())

    # Selection for ground truth and features
    target_col = st.sidebar.selectbox("Select Target (Diagnosis) Column", df.columns, index=len(df.columns)-1)
    
    st.sidebar.header("2. Model Selection")
    model_map = {
    "Logistic Regression": "breast_cancer_model_lr.pkl",
    "Decision Tree": "breast_cancer_model_dt.pkl",
    "XG Boost": "breast_cancer_model_xg.pkl"
}
    model_option = st.sidebar.selectbox(
        "Choose Trained Model",
        ["Logistic Regression", "Decision Tree", "XG Boost"]
    )

    # Load the actual model and scaler
    model, scaler = load_ml_assets(model_option)

    if st.button("Run Evaluation"):
        if model is not None:
            # --- Data Preparation ---
            X_raw = df.drop(columns=[target_col])
            
            # 1. Encode target labels (e.g., 'M'/'B' to 0/1)
            le = LabelEncoder()
            y_true = le.fit_transform(df[target_col].astype(str))
            
            # 2. Scale features (CRITICAL for kNN/Logistic Regression)
            X_scaled = scaler.transform(X_raw)
            
            # 3. Generate Actual Predictions
            y_pred = model.predict(X_scaled)
            
            # Get probabilities for AUC (if supported by model)
            try:
                y_proba = model.predict_proba(X_scaled)[:, 1]
            except AttributeError:
                y_proba = y_pred # Fallback if model doesn't support probabilities

            # --- 3. Display Metrics ---
            st.subheader(f"Results for {model_option}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
            m2.metric("AUC Score", f"{roc_auc_score(y_true, y_proba):.2f}")
            m3.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.2f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
            m5.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
            m6.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

            # --- 4. Visualizations ---
            st.write("---")
            c1, c2 = st.columns(2)

            with c1:
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='g', cmap='Purples', ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

            with c2:
                st.write("#### Classification Report")
                report = classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.error(f"Model file '{model_option}.pkl' or 'scaler.pkl' not found in repository.")

else:
    st.info("Please upload a CSV file to begin evaluation.")