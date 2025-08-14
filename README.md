# Breast Cancer Classification with Machine Learning

This project applies classic machine learning models to classify breast tumors as benign or malignant based on selected medical imaging features.

---

## Objective

Compare multiple classification algorithms and identify the best-performing model for breast cancer diagnosis, using a simplified subset of the UCI Breast Cancer Wisconsin dataset.

---

## Dataset

- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- Target variable: `Subtype` (0 = Benign, 1 = Malignant)
- Used 5 key features for simplicity and interpretability:
  - `mean radius`
  - `mean texture`
  - `mean perimeter`
  - `mean area`
  - `mean smoothness`

---

## Methods

The project follows a standard ML workflow:

1. **Data Preprocessing**
   - Train/validation/test split (60/20/20)
   - Feature scaling using `StandardScaler`
   - Class balancing with `RandomOverSampler`

2. **Model Training**
   - Trained 5 models:
     - K-Nearest Neighbors (KNN)
     - Naive Bayes
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest

3. **Evaluation**
   - Accuracy, precision, recall, F1-score via `classification_report`
   - 5-fold cross-validation for robust accuracy comparison
   - ROC curve and feature analysis were conducted using Logistic Regression, due to its interpretability

---

## Results

| Model              | Mean Accuracy (5-Fold CV) |
|-------------------|---------------------------|
| KNN               | 0.887                     |
| Naive Bayes       | 0.902                     |
| Logistic Regression | 0.926                   |
| **SVM**           | **0.928**                |
| Random Forest     | 0.923                     |

- **SVM** achieved the highest cross-validation accuracy.
- **Logistic Regression** was used to interpret feature importance.

---

## Visualizations

- Pairplot of feature distribution by tumor type  
- Histograms of individual features (benign vs. malignant)  
- Feature importance bar plot (Logistic Regression)  
- ROC curve (Logistic Regression)  
- Bar chart: Model Comparison – 5-Fold Cross-Validation Accuracy

---

## Project Structure

breast-cancer-project/
│
├── .venv/                 # Virtual environment (excluded from version control)
├── data/                  # Dataset (CSV file)
├── models/                # Reserved for saved models (currently empty)
├── notebook/              # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── README.md              # Project overview and documentation


---

## Future Work

- Feature engineering and testing full dataset
- Hyperparameter tuning (e.g. GridSearchCV)
- Model deployment with Streamlit or Flask
- Explainable AI (e.g. SHAP, LIME)

---

## Author

Afrakoma Julia Ouattara  
Bioinformatics Student | Passionate about Machine Learning in Precision Medicine  
[LinkedIn](https://www.linkedin.com/in/afrakoma-julia-ouattara-25a138307) · [GitHub](https://github.com/afrakoma-dev)

