import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Configuration
st.set_page_config(page_title="Model Evaluator", layout="wide")

st.title("📊 Classification Model Evaluation App")
st.markdown("Upload your test dataset to evaluate model performance and view metrics.")

# --- a. Dataset Upload Option ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Test Data")
    st.dataframe(df.head())

    # Assume the last column is the target/ground truth
    target_col = st.sidebar.selectbox("Select Target (Ground Truth) Column", df.columns, index=len(df.columns)-1)
    feature_cols = [col for col in df.columns if col != target_col]

    # --- b. Model Selection Dropdown ---
    st.sidebar.header("2. Model Selection")
    model_option = st.sidebar.selectbox(
        "Choose Model to Evaluate",
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    # In a real app, you would load your trained .pkl file here:
    # model = joblib.load(f"{model_option}.pkl")
    
    st.warning("⚠️ Note: This demo requires a pre-trained model or a function to generate predictions.")

    if st.button("Run Evaluation"):
        # Placeholder: Generate random predictions for demonstration
        # Replace y_pred with: model.predict(df[feature_cols])
        y_true = df[target_col]
        y_pred = np.random.choice(y_true.unique(), size=len(y_true)) 

        # --- c. Display Evaluation Metrics ---
        st.subheader(f"Results for {model_option}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.2f}")
        col3.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.2f}")
        col4.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

        # --- d. Confusion Matrix & Classification Report ---
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
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload a CSV file in the sidebar to get started.")