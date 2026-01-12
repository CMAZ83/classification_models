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
st.markdown("Upload a test dataset to evaluate your pre-trained models stored in the `model/` directory.")

# Define the Base Directory for absolute pathing
BASE_DIR = os.getcwd()

# --- 1. Optimized Asset Loading ---
@st.cache_resource
def get_scaler():
    """Loads the universal scaler using an absolute path."""
    scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

@st.cache_resource
def get_model(model_filename):
    """Loads a specific model using its mapped path."""
    model_path = os.path.join(BASE_DIR, model_filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# --- 2. Sidebar & Model Mapping ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Test Data")
    st.dataframe(df.head())

    # Selection for ground truth (Diagnosis)
    target_col = st.sidebar.selectbox("Select Target (Diagnosis) Column", df.columns, index=len(df.columns)-1)
    
    st.sidebar.header("2. Model Selection")
    
    # Mapping friendly names to your exact .pkl file paths
    model_map = {
        "Logistic Regression": "breast_cancer_classification_models/model/breast_cancer_model_lr.pkl",
        "Decision Tree": "breast_cancer_classification_models/model/breast_cancer_model_dt.pkl",
        "XG Boost": "breast_cancer_classification_models/model/breast_cancer_model_xg.pkl"
    }
    
    model_option = st.sidebar.selectbox("Choose Trained Model", options=list(model_map.keys()))
    selected_path = model_map[model_option]

    if st.button("Run Evaluation"):
        # Load assets
        model = get_model(selected_path)
        scaler = get_scaler()

        if model is not None and scaler is not None:
            try:
                # --- 3. Data Preparation ---
                # Remove target column to get feature set
                X_raw = df.drop(columns=[target_col])
                
                # Encode target labels (M/B to 0/1)
                le = LabelEncoder()
                y_true = le.fit_transform(df[target_col].astype(str))
                
                # Scale features exactly as done during training
                X_scaled = scaler.transform(X_raw)
                
                # Generate Predictions
                y_pred = model.predict(X_scaled)
                
                # Generate Probabilities for AUC
                try:
                    y_proba = model.predict_proba(X_scaled)[:, 1]
                except AttributeError:
                    y_proba = y_pred 

                # --- 4. Display Metrics ---
                st.subheader(f"Results for {model_option}")
                
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
                m2.metric("AUC Score", f"{roc_auc_score(y_true, y_proba):.2f}")
                m3.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.2f}")
                m4.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
                m5.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
                m6.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

                # --- 5. Visualizations ---
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

            except Exception as e:
                st.error(f"Computation Error: {e}")
        else:
            # Enhanced debugging error message
            st.error(f"File Error: Could not find '{selected_path}' or 'model/scaler.pkl'.")
            st.write("Debug - Current Working Directory:", BASE_DIR)
            st.write("Debug - Folder Contents:", os.listdir(os.path.join(BASE_DIR, "model") if os.path.exists("model") else "."))
else:
    st.info("Please upload a CSV file to begin evaluation.")