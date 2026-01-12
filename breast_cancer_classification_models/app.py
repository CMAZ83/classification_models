import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Model Evaluator", layout="wide")

st.title("📊 Classification Model Evaluation App")
st.markdown("Upload your test dataset to evaluate model performance and view metrics.")

# --- 1. Dataset Upload Option ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Test Data")
    st.dataframe(df.head())

    # User selects the ground truth column
    target_col = st.sidebar.selectbox("Select Target (Ground Truth) Column", df.columns, index=len(df.columns)-1)
    
    # --- 2. Model Selection Dropdown ---
    st.sidebar.header("2. Model Selection")
    model_option = st.sidebar.selectbox(
        "Choose Model to Evaluate",
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    if st.button("Run Evaluation"):
        # --- Data Preparation ---
        # Ensure y_true is numeric for metric calculations
        le = LabelEncoder()
        y_true = le.fit_transform(df[target_col].astype(str))
        
        # Placeholder Prediction Logic
        # In production: y_pred = model.predict(df.drop(columns=[target_col]))
        # Generating random binary predictions (0 or 1) to match encoded y_true
        y_pred = np.random.randint(0, 2, size=len(y_true)) 
        
        # Generating random probabilities for AUC score
        y_proba = np.random.uniform(0, 1, size=len(y_true))

        # --- 3. Display Evaluation Metrics ---
        st.subheader(f"Results for {model_option}")
        
        # First row of metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
        m2.metric("AUC Score", f"{roc_auc_score(y_true, y_proba):.2f}")
        m3.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.2f}")

        # Second row of metrics
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
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

        with c2:
            st.write("#### Classification Report")
            # We use target_names so the report shows 'Benign/Malignant' instead of '0/1'
            report = classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload a CSV file in the sidebar to get started.")