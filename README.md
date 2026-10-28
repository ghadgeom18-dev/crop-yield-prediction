# Crop Yield Prediction Using Machine Learning

## Overview
A complete Machine Learning regression system that predicts Indian agricultural crop yield
(in tonnes per hectare) using historical district-level agricultural data from
the Ministry of Agriculture & Farmers Welfare, Government of India.

## Dataset
- **Source:** India Data Portal (indiadataportal.com)
- **Underlying Source:** Ministry of Agriculture & Farmers Welfare, GoI
- **File:** crop-wise-area-production-yield.csv
- **Records:** 455,359 rows | 16 columns
- **Period:** 1997-1998 to 2022-2023

## Project Structure
`
FA 2/
|-- crop-wise-area-production-yield.csv   # Dataset
|-- crop.ipynb                            # Jupyter Notebook (full project)
|-- crop_yield_model.py                   # Complete ML pipeline script
|-- app.py                                # Streamlit web application
|-- requirements.txt                      # Python dependencies
|-- README.md                             # This file
|-- models/                               # Saved trained model pipelines
|   |-- linear_regression_pipeline.pkl
|   |-- decision_tree_pipeline.pkl
|   |-- random_forest_pipeline.pkl
|   |-- linearsvr_pipeline.pkl
|   -- best_model.pkl
-- results/                              # Results and charts
    |-- model_results.csv
    |-- best_model_info.json
    -- model_comparison.png
`

## Machine Learning Models
| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline — simple linear model |
| Decision Tree | Captures non-linear relationships |
| Random Forest | Ensemble — best performer |
| LinearSVR | SVR variant practical for large datasets |

## Key Design Decisions
- **Target Transformation:** TransformedTargetRegressor with log1p/expm1
  handles extreme right-skewness (skewness ≈ 16.33) and zero-yield records
- **Encoding:** OneHotEncoder for categorical features (no ordinal assumption)
- **Scaling:** StandardScaler for Linear/SVR models; passthrough for tree models
- **No production feature:** Excluded to prevent target leakage
- **GridSearchCV:** Run on 10,000-row representative subset to avoid MemoryError

## How to Run

### 1. Train Models
`ash
python crop_yield_model.py
`
This trains all 4 models, runs GridSearchCV, saves models and results.
Expected time: 30-60 minutes depending on hardware.

### 2. Launch Streamlit App
`ash
streamlit run app.py
`

## Evaluation Metrics
- MAE  — Mean Absolute Error
- MSE  — Mean Squared Error
- RMSE — Root Mean Squared Error
- R2   — Coefficient of Determination

## Target Variable
**yield** — measured in Tonnes per Hectare (T/ha)

## Features Used
| Feature | Type |
|---------|------|
| state_name | Categorical |
| district_name | Categorical |
| crop_name | Categorical |
| crop_type | Categorical |
| season | Categorical |
| area | Numerical |
| year_start | Numerical |
