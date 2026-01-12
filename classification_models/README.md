# classification_models

### a. Problem Statement
The goal of this project is to evaluate and compare the performance of various machine learning algorithms to predict **[Insert Target variable here]**. By analyzing different metrics, we aim to identify the most reliable model for this specific classification task.

### b. Dataset Description
* **Source:** UCI Machine Learning Repository
* **Size:** 569 rows and 30 columns.
* **Features:** Includes numerical and categorical data such as [Feature 1], [Feature 2], and [Feature 3].
* **Target Variable:** Diagnosis, representing whether breast cancer is detected or not.

### c. Models Used
1.  **Logistic Regression:** A linear model used as a baseline for binary classification.
2.  **Decision Tree:** A non-linear model that splits data based on feature thresholds.
3.  **kNN (k-Nearest Neighbors):** An instance-based learner that classifies data points based on proximity.
4.  **Naive Bayes:** A probabilistic classifier based on Bayes' Theorem.
5.  **Random Forest (Ensemble):** A bagging technique that combines multiple decision trees to reduce overfitting.
6.  **XGBoost (Ensemble):** A gradient boosting framework designed for speed and performance.

---

### Model Performance Comparison



| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.973684   | 0.997380    | 0.976190   | 0.953488 | 0.964706 | 0.943898 |
| **Decision Tree** | 0.938596 | 0.936947 | 0.909091 | 0.930233 | 0.919540 |  0.870056|
| **kNN** | 0.947368 | 0.981985  | 0.930233 |  0.930233 | 0.930233 | 0.887979 |
| **Naive Bayes** | 0.964912 | 0.997380 | 0.975610  | 0.930233  | 0.952381 |   0.925285 |
| **Random Forest (Ensemble)** | 0.964912 | 0.995906 | 0.975610 | 0.930233  | 0.952381 |  0.925285 |
| **XGBoost (Ensemble)** | 0.956140 | 0.990829 | 0.952381 | 0.930233 | 0.941176 | 0.906379 |

---

### Qualitative Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed well with linear boundaries; low computational cost. |
| **Decision Tree** | High interpretability but showed signs of high variance/overfitting. |
| **kNN** | Performance heavily dependent on the choice of 'k' and feature scaling. |
| **Naive Bayes** | Fast and effective for high-dimensional data, despite independence assumption. |
| **Random Forest (Ensemble)** | Significant improvement over single trees; very stable across folds. |
| **XGBoost (Ensemble)** | Best overall performance; handled non-linear relationships most effectively. |