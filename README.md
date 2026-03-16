# Heart Disease Prediction using Machine Learning

## Overview

This project develops a machine learning model to predict the presence of heart disease using clinical data from the **UCI Heart Disease dataset**.

The goal is to explore the dataset, build predictive models, and evaluate their performance for potential use in clinical decision support systems.

## Dataset

Dataset source: UCI Machine Learning Repository.

The dataset contains clinical attributes related to cardiovascular health such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Fasting blood sugar
* Resting ECG results
* Maximum heart rate achieved
* Exercise induced angina
* ST depression induced by exercise
* Number of major vessels
* Thalassemia

Target variable:

* **1 → Heart disease present**
* **0 → No heart disease**

## Project Structure

```
heart-disease-ml
│
├── data
│
├── notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models
│
├── app
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

## Exploratory Data Analysis

EDA includes:

* Feature distribution analysis
* Correlation analysis
* Detection of potential outliers
* Class balance evaluation

Key observations:

* Some features show strong correlation with heart disease
* Age, chest pain type and maximum heart rate show predictive power
* The dataset shows mild class imbalance

## Feature Engineering

The preprocessing pipeline includes:

* Handling missing values
* Feature scaling
* Encoding categorical variables

## Models Implemented

The following machine learning models were trained and evaluated:

* Logistic Regression
* Random Forest
* Support Vector Machine
* Gradient Boosting

## Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Results

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression |    XX    |    XX   |
| Random Forest       |    XX    |    XX   |
| SVM                 |    XX    |    XX   |

Random Forest achieved the best overall performance.

## Model Interpretation

Feature importance analysis was performed to understand which variables contribute most to predictions.

Important predictors include:

* Chest pain type
* Maximum heart rate
* ST depression
* Number of major vessels

## Future Work

Potential improvements:

* Hyperparameter optimization
* Model interpretability with SHAP
* Deployment as a web application
* Testing on larger cardiovascular datasets

## How to Run the Project

Clone the repository:

```
git clone https://github.com/yourusername/heart-disease-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the notebooks to reproduce the analysis.

## Author

Ramiro Cervetto
Biomedical Engineering
Machine Learning for Healthcare
