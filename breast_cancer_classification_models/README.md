# Breast Cancer Classification Models

### a. Problem Statement
The goal of this project is to evaluate and compare the performance of various machine learning algorithms to predict whether a patient might have breast cancer. By analyzing different metrics, we aim to identify the most reliable model for this specific classification task.

### b. Dataset Description
* **Source:** UCI Machine Learning Repository
* **Size:** 569 rows and 30 columns.
* **Features:** Core Nuclear Characteristics (10)

These features are computed from digitized images of fine needle aspirate (FNA) samples of breast masses.

Radius: Average of distances from the center to points on the perimeter.

Texture: Standard deviation of gray-scale values, indicating surface roughness.

Perimeter: The measured distance around the nucleus.

Area: The total surface area of the cell nucleus.

Smoothness: Local variation in radius lengths.Compactness: Calculated as $\frac{\text{perimeter}^2}{\text{area}} - 1.0$.

Concavity: Severity of concave portions (indentations) of the contour.

Concave Points: The actual number of concave portions on the contour.

Symmetry: Measured by comparing chords perpendicular to the major axis of the cell.

Fractal Dimension: A "coastline approximation" measurement minus one.

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
| **Logistic Regression** | Top Performer: Achieved the highest scores across all metrics (Accuracy: 0.973, MCC: 0.943), suggesting the dataset has strong linear separability. |
| **Decision Tree** | Baseline Performer: Recorded the lowest scores in the group; its high interpretability is offset by lower generalization compared to ensemble methods. |
| **kNN** | Moderate Stability: Performance is solid (0.947 Accuracy) but slightly trails the probabilistic and ensemble models, likely due to its sensitivity to local data outliers. |
| **Naive Bayes** | Exceptional Separability: Tied for the highest AUC (0.997), proving extremely effective at class separation despite the feature independence assumption. |
| **Random Forest (Ensemble)** | Robust & Balanced: Showed a significant accuracy boost (0.964) over the single Decision Tree, providing very stable and reliable predictions. |
| **XGBoost (Ensemble)** | High Complexity: Performed very well (0.956 Accuracy), though in this specific instance, it was slightly outperformed by the simpler Logistic Regression model. |
