# breast_cancer_classification_models/app.py
# Streamlit app for evaluating breast cancer classification models with enhanced EDA and visualizations
# Author: 2024dc04022
import streamlit as st
import pandas as pd
import os
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- Page Configuration ---
st.set_page_config(page_title="Breast Cancer Model Evaluator", layout="wide")

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

st.title("📊 Breast Cancer Classification Evaluator")
st.markdown("**ML Assignment 2 by 2024dc04022@wilp.bits-pilani.ac.in**")
st.markdown("Upload a test dataset to evaluate your pre-trained models.")

# --- Download Test Data Section ---
st.sidebar.header("📥 Download Test Data")
test_data_folder = os.path.join(ROOT, SUBFOLDER, "data", "test_data") if os.path.exists(os.path.join(ROOT, SUBFOLDER)) else os.path.join(ROOT, "data", "test_data")

if os.path.exists(test_data_folder):
    test_files = [f for f in os.listdir(test_data_folder) if os.path.isfile(os.path.join(test_data_folder, f)) and f.endswith('.csv')]
    if test_files:
        for file in sorted(test_files):
            file_path = os.path.join(test_data_folder, file)
            with open(file_path, "rb") as f:
                file_data = f.read()
                st.sidebar.download_button(
                    label=f"📄 {file}",
                    data=file_data,
                    file_name=file,
                    mime="text/csv",
                    key=f"download_{file}"
                )
    else:
        st.sidebar.info("No test data files available")
else:
    st.sidebar.info("Test data folder not found")

st.sidebar.markdown("---")

# --- 3. Sidebar Configuration ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- Exploratory Data Analysis Tab ---
    tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔍 Exploratory Analytics", "📈 Advanced Visualizations"])
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Data Summary
        st.subheader("1. Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(df))
        col2.metric("Total Features", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        if 'diagnosis' in df.columns:
            diagnosis_counts = df['diagnosis'].value_counts()
            col4.metric("Diagnosis Distribution", f"M:{diagnosis_counts.get('M', 0)} | B:{diagnosis_counts.get('B', 0)}")
        
        # Statistical Summary
        st.subheader("2. Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Correlation Matrix
        st.subheader("3. Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax_corr)
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close()
        
        # Top Correlations
        st.subheader("4. Top Feature Correlations")
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(10)
        st.dataframe(corr_df, use_container_width=True)
        
        # Outlier Detection
        st.subheader("5. Outlier Detection (IQR Method)")
        outlier_info = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_info.append({
                    'Feature': col,
                    'Outlier Count': len(outliers),
                    'Percentage': f"{(len(outliers)/len(df)*100):.2f}%"
                })
        
        if outlier_info:
            outlier_df = pd.DataFrame(outlier_info).sort_values('Outlier Count', ascending=False)
            st.dataframe(outlier_df, use_container_width=True)
            
            # Box plots for top features with outliers
            st.write("#### Box Plots (Top 6 Features with Outliers)")
            top_outlier_features = outlier_df.head(6)['Feature'].tolist()
            fig_box, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.ravel()
            for idx, feature in enumerate(top_outlier_features):
                sns.boxplot(y=df[feature], ax=axes[idx], color='skyblue')
                axes[idx].set_title(feature, fontsize=10)
                axes[idx].set_ylabel('Value')
            plt.tight_layout()
            st.pyplot(fig_box)
            plt.close()
        else:
            st.info("No outliers detected using IQR method.")
        
        # Feature Distributions (if diagnosis column exists)
        if 'diagnosis' in df.columns:
            st.subheader("6. Feature Distributions by Diagnosis")
            selected_features = st.multiselect(
                "Select features to visualize:",
                numeric_cols[:10],  # Show first 10 features by default
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_features:
                num_features = len(selected_features)
                cols_per_row = 2
                rows = (num_features + cols_per_row - 1) // cols_per_row
                
                fig_dist, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows*4))
                if rows == 1:
                    axes = axes.reshape(1, -1)
                axes = axes.ravel()
                
                for idx, feature in enumerate(selected_features):
                    for diagnosis in df['diagnosis'].unique():
                        if pd.notna(diagnosis):
                            subset = df[df['diagnosis'] == diagnosis][feature]
                            axes[idx].hist(subset, alpha=0.6, label=f'{diagnosis}', bins=20)
                    axes[idx].set_title(feature, fontsize=11)
                    axes[idx].set_xlabel('Value')
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].legend()
                
                # Hide empty subplots
                for idx in range(num_features, len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig_dist)
                plt.close()
    
    with tab1:
        # Original content moved to tab1
        st.write("### Preview of Test Data")
        st.dataframe(df.head())

    with tab1:
        # Original content moved to tab1
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
                            sns.heatmap(cm, annot=True, fmt='g', cmap='Purples', ax=ax,
                                      xticklabels=['Benign', 'Malignant'],
                                      yticklabels=['Benign', 'Malignant'])
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            plt.close()
                        with col_b:
                            st.write("#### Classification Report")
                            report = classification_report(y_true, y_pred, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose())
                    
                    # Move to Advanced Visualizations tab
                    with tab3:
                        st.header("Advanced Visualizations")
                        
                        if has_labels:
                            # ROC Curve
                            st.subheader("1. ROC Curve")
                            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, 
                                       label=f'ROC curve (AUC = {roc_auc:.3f})')
                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                                       label='Random Classifier')
                            ax_roc.set_xlim([0.0, 1.0])
                            ax_roc.set_ylim([0.0, 1.05])
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.set_title(f'Receiver Operating Characteristic - {model_option}')
                            ax_roc.legend(loc="lower right")
                            ax_roc.grid(alpha=0.3)
                            st.pyplot(fig_roc)
                            plt.close()
                            
                            # Prediction Probability Distribution
                            st.subheader("2. Prediction Probability Distribution")
                            fig_prob, ax_prob = plt.subplots(figsize=(10, 5))
                            
                            # Separate probabilities by actual class
                            benign_probs = y_proba[y_true == 0]
                            malignant_probs = y_proba[y_true == 1]
                            
                            ax_prob.hist(benign_probs, bins=30, alpha=0.6, label='Benign (Actual)', color='green')
                            ax_prob.hist(malignant_probs, bins=30, alpha=0.6, label='Malignant (Actual)', color='red')
                            ax_prob.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
                            ax_prob.set_xlabel('Predicted Probability (Malignant)')
                            ax_prob.set_ylabel('Count')
                            ax_prob.set_title('Distribution of Prediction Probabilities by Actual Class')
                            ax_prob.legend()
                            ax_prob.grid(alpha=0.3)
                            st.pyplot(fig_prob)
                            plt.close()
                            
                            # Feature Importance (for tree-based models)
                            if hasattr(model, 'feature_importances_'):
                                st.subheader("3. Feature Importance")
                                feature_importance = pd.DataFrame({
                                    'Feature': X_raw.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False).head(15)
                                
                                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                                sns.barplot(data=feature_importance, y='Feature', x='Importance', 
                                          palette='viridis', ax=ax_imp)
                                ax_imp.set_title('Top 15 Feature Importances')
                                ax_imp.set_xlabel('Importance Score')
                                st.pyplot(fig_imp)
                                plt.close()
                            
                            # Error Analysis
                            st.subheader("4. Error Analysis")
                            misclassified = results_df[results_df['Correct'] == False]
                            
                            if len(misclassified) > 0:
                                st.write(f"**Total Misclassified Samples:** {len(misclassified)} ({len(misclassified)/len(results_df)*100:.2f}%)")
                                
                                col_err1, col_err2 = st.columns(2)
                                
                                with col_err1:
                                    st.write("##### False Positives (Predicted M, Actually B)")
                                    false_positives = len(misclassified[(misclassified[target_col] == 'B') & 
                                                                       (misclassified['Predicted_Diagnosis'] == 'M')])
                                    st.metric("Count", false_positives)
                                
                                with col_err2:
                                    st.write("##### False Negatives (Predicted B, Actually M)")
                                    false_negatives = len(misclassified[(misclassified[target_col] == 'M') & 
                                                                       (misclassified['Predicted_Diagnosis'] == 'B')])
                                    st.metric("Count", false_negatives)
                                
                                st.write("##### Misclassified Samples Details")
                                st.dataframe(misclassified, use_container_width=True)
                            else:
                                st.success("🎉 Perfect classification! No errors detected.")
                            
                            # Confidence Analysis
                            st.subheader("5. Prediction Confidence Analysis")
                            results_df['Confidence'] = np.maximum(y_proba, 1 - y_proba)
                            
                            fig_conf, axes_conf = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # Confidence distribution
                            axes_conf[0].hist(results_df['Confidence'], bins=30, color='steelblue', edgecolor='black')
                            axes_conf[0].axvline(x=results_df['Confidence'].mean(), color='red', 
                                               linestyle='--', linewidth=2, label=f'Mean: {results_df["Confidence"].mean():.3f}')
                            axes_conf[0].set_xlabel('Prediction Confidence')
                            axes_conf[0].set_ylabel('Count')
                            axes_conf[0].set_title('Distribution of Prediction Confidence')
                            axes_conf[0].legend()
                            axes_conf[0].grid(alpha=0.3)
                            
                            # Confidence vs Correctness
                            correct_conf = results_df[results_df['Correct'] == True]['Confidence']
                            incorrect_conf = results_df[results_df['Correct'] == False]['Confidence']
                            
                            axes_conf[1].hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green')
                            if len(incorrect_conf) > 0:
                                axes_conf[1].hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red')
                            axes_conf[1].set_xlabel('Prediction Confidence')
                            axes_conf[1].set_ylabel('Count')
                            axes_conf[1].set_title('Confidence by Prediction Correctness')
                            axes_conf[1].legend()
                            axes_conf[1].grid(alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig_conf)
                            plt.close()
                            
                        else:
                            st.info("ℹ️ Advanced visualizations are only available when actual diagnosis labels are provided in the dataset.")
                    
                    if not has_labels:
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