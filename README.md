# Fairness and Explainability in Machine Learning (Adult Income Dataset)

## Project Overview

This project demonstrates how to build a machine learning classification model using the Adult Income dataset from OpenML and evaluate it not only for performance but also for fairness and explainability.

A Logistic Regression model is trained to predict whether an individual's income is >50K or <=50K. Beyond accuracy, the project analyzes:

* **Fairness across gender groups (male/female)** using Fairlearn metrics
* **Global and local interpretability** using SHAP
* **Instance-level explanations** using LIME

This helps ensure the model is both accurate and ethically responsible.

---

## Dataset

* Source: OpenML Adult Income dataset
* Target variable: `class` (>50K or <=50K)
* Sensitive attribute: `sex` (used for fairness evaluation)

---

## Features

* Data cleaning (handling missing values and duplicates)
* One-hot encoding of categorical variables
* Label encoding of sensitive attribute
* Feature scaling using StandardScaler
* Logistic Regression model training

---

## Model Performance

* Algorithm: Logistic Regression
* Accuracy: ~0.8427

### Evaluation Metrics

* Confusion Matrix
* Precision, Recall, F1-score

---

## Fairness Analysis

Using **Fairlearn MetricFrame**, the following metrics are evaluated across gender groups:

* Accuracy
* Selection Rate
* False Positive Rate (FPR)
* True Positive Rate (TPR)

This helps identify disparities in model predictions between groups.

---

## Explainability

### SHAP (Global + Local Explainability)

* Explains overall feature importance
* Shows how each feature contributes to predictions
* Waterfall plot for individual predictions

### LIME (Local Explainability)

* Explains a single prediction instance
* Shows which features influenced the model decision locally

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/parika8ec-hub/AI_Assignment14.git
cd AI_Assignment14
```

### 2. Install dependencies

---

## Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib shap lime fairlearn openml
```

---

## How to Run

Run the Jupyter Notebook:

```bash
Assignment14.ipynb
```

---

## Project Structure

```
├── main.py / Assignment14.ipynb
├── README.md
```

---

## Key Takeaways

* Model achieves good accuracy (~84%) but shows **performance differences across gender groups**
* Fairness metrics reveal bias in selection rate and TPR
* SHAP and LIME improve interpretability of predictions

---

## Future Improvements

* Try fairness-aware models (e.g., Reweighing, Exponentiated Gradient)
* Reduce bias using preprocessing or postprocessing techniques
* Experiment with advanced models (Random Forest, XGBoost)

---
