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
@st.cache_resource
def load_ml_assets(model_filename):
    """Loads the model based on the mapped filename and the universal scaler."""
    scaler_path = "scaler.pkl"
    
    if os.path.exists(model_filename) and os.path.exists(scaler_path):
        model = joblib.load(model_filename)
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

    # Selection for ground truth
    target_col = st.sidebar.selectbox("Select Target (Diagnosis) Column", df.columns, index=len(df.columns)-1)
    
    st.sidebar.header("2. Model Selection")
    
    # Mapping friendly names to your specific .pkl filenames
    model_map = {
        "Logistic Regression": "breast_cancer_model_lr.pkl",
        "Decision Tree": "breast_cancer_model_dt.pkl",
        "XG Boost": "breast_cancer_model_xg.pkl"
    }
    
    model_option = st.sidebar.selectbox(
        "Choose Trained Model",
        options=list(model_map.keys())
    )

    # Get the actual filename from the map
    selected_filename = model_map[model_option]

    # Load the actual model using the filename
    model, scaler = load_ml_assets(selected_filename)

    if st.button("Run Evaluation"):
        if model is not None and scaler is not None:
            # --- Data Preparation ---
            X_raw = df.drop(columns=[target_col])
            
            # 1. Encode target labels
            le = LabelEncoder()
            y_true = le.fit_transform(df[target_col].astype(str))
            
            # 2. Scale features (Uses your scaler.pkl)
            try:
                X_scaled = scaler.transform(X_raw)
            except Exception as e:
                st.error(f"Scaling error: {e}. Ensure your CSV has the same 30 features used during training.")
                st.stop()
            
            # 3. Generate Predictions
            y_pred = model.predict(X_scaled)
            
            # Get probabilities for AUC
            try:
                y_proba = model.predict_proba(X_scaled)[:, 1]
            except:
                y_proba = y_pred 

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
            st.error(f"Files not found. Looking for: {selected_filename} and scaler.pkl")

else:
    st.info("Please upload a CSV file to begin evaluation.")