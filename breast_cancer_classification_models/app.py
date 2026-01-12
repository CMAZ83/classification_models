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
st.markdown("**ML Assignment 2 by 2024dc04022@wilp.bits-pilani.ac.in**")
st.markdown("Upload a test dataset to evaluate your pre-trained models.")

# --- 1. Path Resolution Logic ---
ROOT = os.getcwd()
SUBFOLDER = "breast_cancer_classification_models"
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

    # Default to 'diagnosis' column if it exists, otherwise use last column
    default_idx = df.columns.get_loc('diagnosis') if 'diagnosis' in df.columns else len(df.columns)-1
    target_col = st.sidebar.selectbox("Select Target (Diagnosis)", df.columns, index=default_idx)
    
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
        
        # Debug info
        st.write(f"Model type: {type(model)}")
        st.write(f"Scaler type: {type(scaler)}")

        if model is not None and scaler is not None:
            if isinstance(model, str):
                st.error(f"Error: Model loaded as string instead of model object. The file '{selected_filename}' may be corrupted.")
            else:
                try:
                    # --- 4. Data Preparation & Prediction ---
                    cols_to_drop = [target_col]
                    if 'id' in df.columns:
                        cols_to_drop.append('id')
                    X_raw = df.drop(columns=cols_to_drop)
                    
                    # Check if we have actual labels for evaluation
                    has_labels = not df[target_col].isna().all()
                    
                    if has_labels:
                        # Use standard breast cancer encoding: M=1 (Malignant), B=0 (Benign)
                        label_map = {'M': 1, 'B': 0}
                        y_true = df[target_col].map(label_map).values
                        
                        # Check for unmapped labels
                        if np.isnan(y_true).any():
                            unique_labels = df[target_col].unique()
                            st.warning(f"⚠️ Found unexpected labels: {unique_labels}. Expected 'M' or 'B'.")
                    
                    X_scaled = scaler.transform(X_raw) 
                    y_pred = model.predict(X_scaled)
                    
                    try:
                        y_proba = model.predict_proba(X_scaled)[:, 1]
                    except AttributeError:
                        y_proba = y_pred 

                    if has_labels:
                        # --- Debug Information (only in evaluation mode) ---
                        st.write("### 🔍 Debug Information")
                        d1, d2, d3 = st.columns(3)
                        d1.write(f"**Original labels:** {df[target_col].unique()}")
                        d1.write(f"**Label mapping:** M=1, B=0")
                        d2.write(f"**True labels (unique):** {np.unique(y_true)}")
                        d2.write(f"**True labels distribution:** {np.bincount(y_true.astype(int))}")
                        d3.write(f"**Predicted labels (unique):** {np.unique(y_pred)}")
                        d3.write(f"**Predicted labels distribution:** {np.bincount(y_pred)}")
                        st.write(f"**Feature shape:** {X_scaled.shape}")
                        
                        # Warning for single-class datasets
                        if len(np.unique(y_true)) == 1:
                            st.warning("⚠️ **Single-class dataset detected.** Your test data contains only one diagnosis type. For comprehensive evaluation (AUC, MCC), include both Benign (B) and Malignant (M) samples.")
                        
                        st.write("---")

                        # --- 5. Metrics Display ---
                        st.subheader(f"Results for {model_option}")
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
                        m2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.2f}")
                        m3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.2f}")
                        m4.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
                        m5.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
                        m6.metric("F1", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")
                    else:
                        st.info("ℹ️ **Prediction Mode**: No diagnosis labels found. Showing predictions only (no evaluation metrics).")
                        st.subheader(f"Predictions from {model_option}")

                    # --- 6. Predictions Table ---
                    st.write("---")
                    st.write("#### Predictions" + (" vs Actual" if has_labels else ""))
                    
                    # Create results dataframe
                    results_df = df.copy()
                    # Map predictions back to labels
                    pred_labels = ['B' if p == 0 else 'M' for p in y_pred]
                    results_df['Predicted_Diagnosis'] = pred_labels
                    results_df['Prediction_Probability'] = y_proba if hasattr(y_proba, '__iter__') else y_pred
                    
                    if has_labels:
                        results_df['Correct'] = (y_true == y_pred)
                        # Display with highlighting
                        st.dataframe(
                            results_df.style.applymap(
                                lambda x: 'background-color: #90EE90' if x == True else ('background-color: #FFB6C6' if x == False else ''),
                                subset=['Correct']
                            ),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(results_df, use_container_width=True)
                    
                    # --- 7. Visualizations (only in evaluation mode) ---
                    if has_labels:
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
                    else:
                        st.write("---")
                        st.write("#### Prediction Summary")
                        pred_counts = pd.Series(y_pred).map({0: 'Benign', 1: 'Malignant'}).value_counts()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Samples", len(y_pred))
                        if 'Benign' in pred_counts:
                            col2.metric("Benign (B)", pred_counts.get('Benign', 0))
                        if 'Malignant' in pred_counts:
                            col3.metric("Malignant (M)", pred_counts.get('Malignant', 0))

                except Exception as e:
                    st.error(f"❌ An error occurred during prediction. Please check your data format.")
                    st.exception(e)
        else:
            st.error(f"File Error: Could not find '{selected_filename}' in {MODEL_DIR}")
else:
    st.info("Please upload a CSV file to begin.")