# 🌾 Crop Yield Prediction Using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crop-yield-prediction.streamlit.app)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Formative Assessment 2 (FA 2) — Advanced Data Science (ADS)**  
> **Author:** Shravani Ghadge (125M1H018)  
> **GitHub Repository:** [https://github.com/ghadgeom18-dev/crop-yield-prediction](https://github.com/ghadgeom18-dev/crop-yield-prediction)  
> **Live Web Application:** [https://crop-yield-prediction.streamlit.app](https://crop-yield-prediction.streamlit.app)

---

## 📌 Project Overview
A complete Machine Learning regression system that predicts Indian agricultural crop yield
(in tonnes per hectare) using historical district-level agricultural data from
the Ministry of Agriculture & Farmers Welfare, Government of India.

- **Live Streamlit App:** [https://crop-yield-prediction.streamlit.app](https://crop-yield-prediction.streamlit.app)
- **GitHub Repository:** [https://github.com/ghadgeom18-dev/crop-yield-prediction](https://github.com/ghadgeom18-dev/crop-yield-prediction)
- **Target Variable:** `yield` (Tonnes per Hectare — T/ha)
- **Dataset:** 455,359 records spanning 1997–2023 across 34 Indian States & UTs

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

## Machine Learning Models & Final Performance
| Model | MAE (T/ha) | RMSE (T/ha) | R² Score | Rank / Notes |
|:---|:---:|:---:|:---:|:---|
| **Random Forest Regressor** | **15.8604** | **275.459** | **0.8944** | 🏆 **Best Performer (Selected)** |
| **Decision Tree Regressor** | 14.1409 | 303.510 | 0.8718 | Fast tree baseline |
| **LinearSVR** | 25.9360 | 382.075 | 0.7969 | Support Vector Regressor |
| **Linear Regression** | 27.3285 | 419.031 | 0.7557 | Baseline linear model |

*All models evaluated on the held-out 20% test set (91,072 unseen records) using `TransformedTargetRegressor` with `log1p`/`expm1`.*

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
