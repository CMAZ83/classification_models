import streamlit as st
import pandas as pd

st.title("Classification Model Comparison")

# To display your performance table
data = {
    "ML Model Name": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
    "Accuracy": [0.85, 0.82, 0.89, 0.91],
    "F1 Score": [0.84, 0.81, 0.88, 0.90]
}
df = pd.DataFrame(data)

st.subheader("Model Performance Metrics")
st.table(df) # Static table or use st.dataframe(df) for interactive