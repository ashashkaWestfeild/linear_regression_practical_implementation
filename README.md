# Supervised Machine Learning: Regression & Classification Pipelines

A professional machine learning project implementing and comparing multiple supervised learning algorithms. The repository contains end-to-end pipelines for both regression (predicting continuous variables) and classification (predicting categorical labels) tasks using `scikit-learn`.

---

## 📌 Project Overview
This project serves as a practical demonstration of training, tuning, and evaluating classical supervised machine learning models. It is divided into two primary pipelines:

1. **Regression Pipeline**: Implements **Ordinary Least Squares (OLS) Linear Regression**, **Ridge Regression (L2 Regularization)**, and **Lasso Regression (L1 Regularization)**. It compares the effects of regularization and hyperparameter tuning (using cross-validation grids) on predicting housing prices.
2. **Classification Pipeline**: Implements **Logistic Regression** to classify tumors as malignant or benign. It covers feature standardization, binary classification metrics, and ROC-AUC curve analysis.

---

## 📊 Performance Summary

### 1. Regression Models Comparison (Boston Housing Dataset)
All models were tested using 5-fold cross-validation, 10-fold cross-validation, and an independent 77/23 train-test split.

| Model | 5-Fold CV Mean MSE | 10-Fold CV Mean MSE | Test MSE | Test $R^2$ Score | Test Adjusted $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | -37.1318 | -34.7053 | **20.7240** | **0.7262** | **0.7029** |
| **Ridge Regression ($\alpha=100$)** | **-29.9057** | **-29.6152** | 22.9321 | 0.6970 | 0.6712 |
| **Lasso Regression ($\alpha=1$)** | -35.5316 | -34.4555 | 25.3723 | 0.6647 | 0.6362 |

* **Key Takeaway**: Ordinary Linear Regression achieved the lowest MSE and highest R-squared on the test set, indicating a strong fit. Ridge Regression ($\alpha=100$) performed best during cross-validation, effectively combating variance and stabilizing predictions across folds.

---

### 2. Classification Model Results (Breast Cancer Dataset)
The Logistic Regression model was trained on standardized features with a 67/33 train-test split.

* **Test Accuracy**: `97.37%`
* **ROC-AUC Score**: `0.9942`

#### Classification Report:
| Class | Label | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Malignant | 0.98 | 0.95 | 0.96 | 43 |
| **1** | Benign | 0.97 | 0.99 | 0.98 | 71 |
| **Accuracy** | | | | **0.97** | 114 |

#### Confusion Matrix:
* **True Negatives (Malignant)**: 41
* **False Positives**: 2
* **False Negatives**: 1
* **True Positives (Benign)**: 70

---

## 📂 Project Structure
```text
linear_regression_practical_implementation/
│
├── .gitignore                                  # Git exclusion file
├── README.md                                   # Project summary & metrics (this file)
├── requirements.txt                            # Python dependencies list
└── linear_regression_practical_implementation.ipynb # End-to-end Jupyter Notebook
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ashashkaWestfeild/linear_regression_practical_implementation.git
   cd linear_regression_practical_implementation
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**:
   ```bash
   jupyter notebook
   ```
   Open `linear_regression_practical_implementation.ipynb` to run the experiments.

---

## 🧠 Technologies Used
* **Languages**: Python
* **Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyter`
