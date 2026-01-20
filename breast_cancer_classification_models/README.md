# Breast Cancer Classification Models
# Assignment 2 submission by 2024dc04022

## 1. Problem Statement

The goal of this project is to evaluate and compare the performance of various machine learning algorithms to predict whether a patient has breast cancer. By analyzing multiple evaluation metrics, we aim to identify the most reliable and effective model for this binary classification task.

---

## 2. Dataset Description

- **Source:** UCI Machine Learning Repository  
- **Dataset Size:** 569 samples × 30 features  
- **Feature Type:** Core nuclear characteristics derived from digitized images of Fine Needle Aspirate (FNA) samples of breast masses  

### 2.1 Feature Details

Each feature describes properties of the cell nucleus:

- **Radius:** Mean distance from the center to points on the perimeter  
- **Texture:** Standard deviation of gray-scale values (surface roughness)  
- **Perimeter:** Length around the nucleus  
- **Area:** Total surface area of the nucleus  
- **Smoothness:** Local variation in radius lengths  
- **Compactness:**  (perimeter² / area) − 1.0
- **Concavity:** Severity of concave portions of the contour  
- **Concave Points:** Number of concave portions of the contour  
- **Symmetry:** Symmetry of the nucleus shape  
- **Fractal Dimension:** Coastline approximation measurement minus one  

> Note: These 10 features are computed as mean, standard error, and worst (largest) values, resulting in a total of 30 input features.

### 2.2 Target Variable

- **Diagnosis**
  - `0` → Benign
  - `1` → Malignant

---

## 3. Models Used

The following machine learning models were evaluated:

1. **Logistic Regression**  
   - Linear baseline model for binary classification  

2. **Decision Tree**  
   - Non-linear model using recursive feature splits  

3. **k-Nearest Neighbors (kNN)**  
   - Instance-based classifier using distance metrics  

4. **Naive Bayes**  
   - Probabilistic model based on Bayes’ theorem and feature independence  

5. **Random Forest (Ensemble)**  
   - Bagging-based ensemble of decision trees to reduce overfitting  

6. **XGBoost (Ensemble)**  
   - Gradient boosting framework optimized for performance and scalability  

---

## 4. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | **0.9737** | **0.9974** | **0.9762** | **0.9535** | **0.9647** | **0.9439** |
| **Decision Tree** | 0.9386 | 0.9369 | 0.9091 | 0.9302 | 0.9195 | 0.8701 |
| **kNN** | 0.9474 | 0.9820 | 0.9302 | 0.9302 | 0.9302 | 0.8880 |
| **Naive Bayes** | 0.9649 | **0.9974** | 0.9756 | 0.9302 | 0.9524 | 0.9253 |
| **Random Forest (Ensemble)** | 0.9649 | 0.9959 | 0.9756 | 0.9302 | 0.9524 | 0.9253 |
| **XGBoost (Ensemble)** | 0.9561 | 0.9908 | 0.9524 | 0.9302 | 0.9412 | 0.9064 |

**Evaluation Metrics Used**
- **Accuracy**
- **ROC-AUC**
- **Precision**
- **Recall**
- **F1-Score**
- **Matthews Correlation Coefficient (MCC)**

---

## 5. Qualitative Observations

| ML Model Name | Performance Summary |
|---------------|---------------------|
| **Logistic Regression** | **Top Performer** – Achieved the highest overall scores across most metrics, indicating strong linear separability in the dataset. |
| **Decision Tree** | **Baseline Performer** – Lowest overall performance; high interpretability but weaker generalization. |
| **kNN** | **Moderate Stability** – Solid accuracy but sensitive to local noise and feature scaling. |
| **Naive Bayes** | **Exceptional Class Separation** – Tied for the highest AUC, despite its strong independence assumption. |
| **Random Forest (Ensemble)** | **Robust & Balanced** – Significant improvement over a single decision tree with stable predictions. |
| **XGBoost (Ensemble)** | **High Complexity Model** – Strong performance, though slightly outperformed by simpler linear models for this dataset. |

---

## 6. Conclusion

- The dataset exhibits strong linear separability, making **Logistic Regression** an excellent and interpretable choice.
- Ensemble methods provide robustness but do not significantly outperform simpler models in this case.
- Probabilistic approaches like **Naive Bayes** perform exceptionally well due to well-separated feature distributions.

---

## 7. Future Work

- Hyperparameter tuning and cross-validation  
- Feature selection and dimensionality reduction (PCA)  
- Model explainability using SHAP / LIME  
- Cost-sensitive learning to reduce false negatives  

---

