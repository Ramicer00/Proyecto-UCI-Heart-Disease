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

## Preprocessing Decisions

### 1. Medically Impossible Zeros
Zeros were identified in variables where they are physiologically impossible.
These values were converted to NaN for subsequent imputation.

| Variable | Zeros detected | % | Justification |
|---|---|---|---|
| `chol` | 202 | 22% | Cholesterol cannot be 0 mg/dl |
| `trestbps` | 60 | 6.5% | Blood pressure cannot be 0 mmHg |

### 2. Null Imputation
All continuous numeric variables were analyzed for physiologically impossible
values. `age` and `thalach` were verified with no zeros found, requiring no action.

#### `chol` and `trestbps` → mean imputation
Both variables are continuous and approximately symmetric according to the EDA
histograms, making the mean a reasonable estimate for missing values.

Although `chol` showed a weak correlation with `target` (−0.23), the column was
kept since cholesterol is a clinically established cardiovascular risk factor.
The low correlation is explained by the heterogeneity of the 4 data sources and
the high proportion of incorrectly recorded values.

#### Remaining variables → mode imputation

| Variable | Nulls | % | Type |
|---|---|---|---|
| `restecg` | 2 | 0.2% | Categorical |
| `thalach` | 55 | 6.0% | Numerical |
| `exang` | 55 | 6.0% | Categorical |
| `fbs` | 90 | 9.8% | Categorical |
| `oldpeak` | 62 | 6.7% | Numerical |
| `slope` | 309 | 33.6% | Categorical |
| `thal` | 486 | 52.8% | Categorical |
| `ca` | 611 | 66.4% | Categorical |

Mode imputation was chosen for all variables for the following reasons:
- Most affected variables are categorical, where mode is the most appropriate
measure of central tendency
- For numerical variables (`thalach`, `oldpeak`), mode reduces introduced bias
compared to mean when nulls exceed 6%

Alternative strategies were considered for `ca`, `thal` and `slope` given their
high percentage of nulls:
- **Drop columns**: discarded because `ca` (0.46) and `thal` (0.50) have the
highest correlation with `target`
- **KNN Imputer**: discarded because with 66% nulls in `ca`, many neighbors also
lack the registered value, limiting its effectiveness
- **Indicator variables**: discarded because it may introduce dataset biases
(which hospitals performed which studies) rather than real clinical patterns

It is acknowledged that mode imputation on variables with high null percentages
introduces bias, a limitation that should be considered when interpreting
model results.

### 3. Outlier Treatment

Extreme values were analyzed across all continuous numeric variables:

| Variable | Min | Max | Outliers? | Action |
|---|---|---|---|---|
| `age` | 28 | 77 | ❌ | No action |
| `trestbps` | 80 | 200 | ❌ | No action |
| `chol` | 85 | 603 | ⚠️ | No action (clinically possible) |
| `thalach` | 60 | 202 | ❌ | No action |
| `oldpeak` | -2.6 | 6.2 | ✅ | Fix negative values |

#### `chol` → no action
The maximum value of 603 mg/dl is extreme but clinically possible
(severe hypercholesterolemia). It is not considered a recording error.

#### `oldpeak` → clipping to 0
12 records (1.3%) with negative `oldpeak` values were detected.
Physiologically, `oldpeak` represents ST segment depression induced by exercise,
whose minimum possible value is 0 (absence of depression).
Negative values are recording errors and were replaced with 0 via clipping.

### 4. Categorical Variable Encoding

**One-Hot Encoding** (`pd.get_dummies`) was applied to nominal categorical
variables. Label Encoding was discarded because it would imply a hierarchical
order between categories that does not exist (e.g. chest pain types).

| Variable | Categories | Reason |
|---|---|---|
| `cp` | 4 | Chest pain type, no hierarchical order |
| `restecg` | 3 | Resting ECG result, no hierarchical order |
| `slope` | 3 | ST segment slope, no hierarchical order |
| `ca` | 4 | Number of colored vessels, no hierarchical order |
| `thal` | 3 | Thalassemia type, no hierarchical order |

`drop_first=True` was used to remove the first dummy category of each variable,
avoiding multicollinearity, which is especially important for Logistic Regression.

Variables `sex`, `fbs` and `exang` were not encoded as they are already binary.

### 5. Numeric Variable Scaling

**StandardScaler** was applied to continuous numeric variables, transforming
each variable to have mean 0 and standard deviation 1.

| Variable | Scale? | Reason |
|---|---|---|
| `age` | ✅ | Continuous, different scale from others |
| `trestbps` | ✅ | Continuous, values between 80 and 200 |
| `chol` | ✅ | Continuous, values between 85 and 603 |
| `thalach` | ✅ | Continuous, values between 60 and 202 |
| `oldpeak` | ✅ | Continuous, values between 0 and 6.2 |
| `sex`, `fbs`, `exang` | ❌ | Already binary (0 and 1) |
| One-Hot dummies | ❌ | Already binary (0 and 1) |
| `target` | ❌ | Target variable |

Random Forest and XGBoost are not sensitive to scale, but Logistic Regression
requires it. Since all 3 models share the same processed dataset, StandardScaler
is applied to all continuous numeric variables.

The scaler is saved to `models/scaler.pkl` for reuse in production without
retraining, ensuring new data is transformed with the same parameters as the
training data.

## Models Implemented

The following machine learning models were trained and evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

## Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Results

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression |    XX    |    XX     |   XX   |    XX    |   XX    |
| Random Forest       |    XX    |    XX     |   XX   |    XX    |   XX    |
| XGBoost             |    XX    |    XX     |   XX   |    XX    |   XX    |



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
