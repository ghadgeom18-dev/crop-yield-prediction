# ============================================================================
# CROP YIELD PREDICTION - COMPLETE ML PIPELINE
# ============================================================================
# Dataset : India Crop Wise Area, Production and Yield (1997-2023)
# Source  : India Data Portal / Ministry of Agriculture & Farmers Welfare, GoI
# Target  : yield (Tonnes / Hectare)
# Models  : Linear Regression | Decision Tree | Random Forest | LinearSVR
# Tuning  : GridSearchCV on Random Forest (memory-safe representative subset)
# ============================================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

import joblib

warnings.filterwarnings('ignore')

print('=' * 60)
print('CROP YIELD PREDICTION - ML PIPELINE')
print('=' * 60)

# ---- PATHS ------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'crop-wise-area-production-yield.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- 1. LOAD DATASET --------------------------------------------------------
print('\n[1/10] Loading dataset...')
data = pd.read_csv(DATA_PATH)
print(f'       Rows: {len(data):,}  |  Columns: {data.shape[1]}')

# ---- 2. FEATURE ENGINEERING -------------------------------------------------
print('\n[2/10] Cleaning data & engineering features...')

ml_data = data.copy()
# Convert '1997-1998' -> 1997
ml_data['year_start'] = data['year'].str[:4].astype(int)

# Drop columns not needed (identifiers, units, leakage)
# 'production' is excluded: production = area x yield -> TARGET LEAKAGE
columns_to_drop = [
    'id', 'state_code', 'district_code', 'crop_code',
    'area_unit', 'production', 'production_unit', 'yield_unit', 'year'
]
ml_data = ml_data.drop(columns=columns_to_drop)

print(f'       ML dataset shape: {ml_data.shape}')
print(f'       Missing values  : {ml_data.isnull().sum().sum()}')

# ---- 3. FEATURES & TARGET ---------------------------------------------------
print('\n[3/10] Defining features and target...')

categorical_features = ['state_name', 'district_name', 'crop_name', 'crop_type', 'season']
numerical_features   = ['area', 'year_start']
TARGET = 'yield'

X = ml_data.drop(columns=[TARGET])
y = ml_data[TARGET]

print(f'       Features (X): {X.shape}  |  Target (y): {y.shape}')
print(f'       Yield: min={y.min():.3f}, median={y.median():.3f}, mean={y.mean():.3f}, max={y.max():.3f}')
print(f'       Skewness: {y.skew():.3f}')

# ---- 4. TRAIN / TEST SPLIT (80/20) ------------------------------------------
print('\n[4/10] Splitting data - 80% train / 20% test...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f'       Train: {X_train.shape}  |  Test: {X_test.shape}')

# ---- 5. PREPROCESSING PIPELINES ---------------------------------------------
print('\n[5/10] Building preprocessing pipelines...')

# OHE with handle_unknown='ignore' so Streamlit predictions work for
# any state/district combination not seen during training
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

# For Linear Regression and LinearSVR - scale numerical features
linear_preprocessor = ColumnTransformer(transformers=[
    ('categorical', cat_encoder, categorical_features),
    ('numerical', StandardScaler(), numerical_features),
], remainder='drop')

# For tree models - trees do not require feature scaling
tree_preprocessor = ColumnTransformer(transformers=[
    ('categorical', cat_encoder, categorical_features),
    ('numerical', 'passthrough', numerical_features),
], remainder='drop')

print('       linear_preprocessor : OHE + StandardScaler')
print('       tree_preprocessor   : OHE + passthrough (no scaling)')

# ---- 6. BUILD FOUR MODEL PIPELINES ------------------------------------------
# TransformedTargetRegressor:
#   - Training  : on log1p(yield)  -> handles right-skewness & zero values
#   - Prediction: back-transformed via expm1 -> original yield scale
print('\n[6/10] Building model pipelines...')

# 6a. Linear Regression
linear_regression_pipeline = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessing', linear_preprocessor),
        ('model', LinearRegression()),
    ]),
    func=np.log1p, inverse_func=np.expm1,
)

# 6b. Decision Tree (no max_depth = grows fully, learns all patterns)
decision_tree_pipeline = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessing', tree_preprocessor),
        ('model', DecisionTreeRegressor(random_state=42)),
    ]),
    func=np.log1p, inverse_func=np.expm1,
)

# 6c. Random Forest - n_estimators=50 is fast AND produces quality results
#     100 trees barely outperforms 50 on large datasets at 2x the time/memory
random_forest_pipeline = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessing', tree_preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=50,  # Fast but effective; GridSearchCV will tune this
            random_state=42,
            n_jobs=-1,        # Use all cores
        )),
    ]),
    func=np.log1p, inverse_func=np.expm1,
)

# 6d. LinearSVR - O(n) complexity, practical for 455k rows unlike RBF-SVR
linear_svr_pipeline = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessing', linear_preprocessor),
        ('model', LinearSVR(random_state=42, max_iter=5000)),
    ]),
    func=np.log1p, inverse_func=np.expm1,
)

print('       4 pipelines ready: LinearRegression | DecisionTree | RandomForest | LinearSVR')

# ---- 7. TRAIN BASELINE MODELS -----------------------------------------------
print('\n[7/10] Training baseline models on full training set...')

print('       [1/4] Linear Regression...')
linear_regression_pipeline.fit(X_train, y_train)
print('             done')

print('       [2/4] Decision Tree...')
decision_tree_pipeline.fit(X_train, y_train)
print('             done')

print('       [3/4] Random Forest (n_estimators=50, n_jobs=-1)...')
random_forest_pipeline.fit(X_train, y_train)
print('             done')

print('       [4/4] LinearSVR (max_iter=5000)...')
linear_svr_pipeline.fit(X_train, y_train)
print('             done')

# ---- 8. GRIDSEARCHCV ON RANDOM FOREST (MEMORY-SAFE) -------------------------
# Strategy:
#   - 10,000-row representative tuning subset from X_train (test set untouched)
#   - 4 parameter combos x 3 folds = 12 fits total
#   - n_jobs=1 (sequential) to prevent BrokenProcessPool / MemoryError
#   - Best params -> retrain on FULL X_train
print('\n[8/10] GridSearchCV on Random Forest (memory-safe approach)...')
print('       Extracting 10,000-row representative tuning subset from X_train...')
print('       (Test set remains UNTOUCHED during all tuning operations)')

X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, train_size=10000, random_state=42
)
print(f'       Tuning subset: {X_tune.shape}')

# Fresh RF pipeline for GridSearchCV
# n_jobs=1 on model to avoid nested parallelism memory crashes
rf_for_gridsearch = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessing', ColumnTransformer(transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features),
            ('numerical', 'passthrough', numerical_features),
        ])),
        ('model', RandomForestRegressor(random_state=42, n_jobs=1)),
    ]),
    func=np.log1p, inverse_func=np.expm1,
)

# Small but meaningful grid: 4 combinations x 3 folds = 12 model fits
param_grid = {
    'regressor__model__n_estimators': [50, 100],
    'regressor__model__max_depth':    [15, 20],
}

rf_grid_search = GridSearchCV(
    estimator=rf_for_gridsearch,
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,    # Sequential to avoid MemoryError / BrokenProcessPool
    verbose=1,
)

print(f'       Grid: {param_grid}')
print('       Running 12 fits (4 combos x 3-fold CV)...')
rf_grid_search.fit(X_tune, y_tune)

best_params  = rf_grid_search.best_params_
best_cv_rmse = -rf_grid_search.best_score_

print(f'\n       Best params : {best_params}')
print(f'       Best CV RMSE: {best_cv_rmse:.4f}')

# ---- 9. RETRAIN FINAL RF ON COMPLETE TRAINING SET ---------------------------
print('\n[9/10] Retraining final Random Forest on complete training set...')
print(f'       Using best params: {best_params}')

final_rf_pipeline = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessing', tree_preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=best_params['regressor__model__n_estimators'],
            max_depth=best_params['regressor__model__max_depth'],
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    func=np.log1p, inverse_func=np.expm1,
)
final_rf_pipeline.fit(X_train, y_train)
print('       Final Random Forest trained on complete training set')

# ---- 10. FINAL EVALUATION ON TEST SET ---------------------------------------
print('\n[10/10] Final evaluation on HELD-OUT TEST SET...')
print('        (Test set was never used during training or tuning)')

def evaluate_model(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Generate predictions
lr_pred  = linear_regression_pipeline.predict(X_test)
dt_pred  = decision_tree_pipeline.predict(X_test)
rf_pred  = final_rf_pipeline.predict(X_test)
svr_pred = linear_svr_pipeline.predict(X_test)

# Compute all metrics
lr_mae,  lr_mse,  lr_rmse,  lr_r2  = evaluate_model(y_test, lr_pred)
dt_mae,  dt_mse,  dt_rmse,  dt_r2  = evaluate_model(y_test, dt_pred)
rf_mae,  rf_mse,  rf_rmse,  rf_r2  = evaluate_model(y_test, rf_pred)
svr_mae, svr_mse, svr_rmse, svr_r2 = evaluate_model(y_test, svr_pred)

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'LinearSVR'],
    'MAE':  [lr_mae,  dt_mae,  rf_mae,  svr_mae],
    'MSE':  [lr_mse,  dt_mse,  rf_mse,  svr_mse],
    'RMSE': [lr_rmse, dt_rmse, rf_rmse, svr_rmse],
    'R2':   [lr_r2,   dt_r2,   rf_r2,   svr_r2],
})

print('\n' + '=' * 65)
print('  FINAL MODEL COMPARISON')
print('=' * 65)
print(results.to_string(index=False))
print('=' * 65)

best_idx        = results['R2'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
print(f'\n  Best model (highest R2): {best_model_name}')

# ---- SAVE RESULTS -----------------------------------------------------------
results_csv = os.path.join(RESULTS_DIR, 'model_results.csv')
results.to_csv(results_csv, index=False)
print(f'\n  Results saved -> {results_csv}')

best_model_info = {
    'best_model': best_model_name,
    'best_params': best_params,
    'best_cv_rmse': float(best_cv_rmse),
    'final_metrics': {
        'MAE':  float(results.loc[best_idx, 'MAE']),
        'MSE':  float(results.loc[best_idx, 'MSE']),
        'RMSE': float(results.loc[best_idx, 'RMSE']),
        'R2':   float(results.loc[best_idx, 'R2']),
    },
}
info_json = os.path.join(RESULTS_DIR, 'best_model_info.json')
with open(info_json, 'w') as f:
    json.dump(best_model_info, f, indent=4)
print(f'  Best model info saved -> {info_json}')

# ---- SAVE MODEL PIPELINES ---------------------------------------------------
best_pipeline_map = {
    'Linear Regression': linear_regression_pipeline,
    'Decision Tree':     decision_tree_pipeline,
    'Random Forest':     final_rf_pipeline,
    'LinearSVR':         linear_svr_pipeline,
}

print('\n  Saving model pipelines via joblib...')
joblib.dump(linear_regression_pipeline, os.path.join(MODELS_DIR, 'linear_regression_pipeline.pkl'))
print('    linear_regression_pipeline.pkl saved')
joblib.dump(decision_tree_pipeline, os.path.join(MODELS_DIR, 'decision_tree_pipeline.pkl'))
print('    decision_tree_pipeline.pkl saved')
joblib.dump(final_rf_pipeline, os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'))
print('    random_forest_pipeline.pkl saved')
joblib.dump(linear_svr_pipeline, os.path.join(MODELS_DIR, 'linearsvr_pipeline.pkl'))
print('    linearsvr_pipeline.pkl saved')
joblib.dump(best_pipeline_map[best_model_name], os.path.join(MODELS_DIR, 'best_model.pkl'))
print(f'    best_model.pkl -> {best_model_name}')

# ---- MODEL COMPARISON CHARTS ------------------------------------------------
print('\n  Generating comparison charts...')
sns.set_style('whitegrid')
palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
models_list = results['Model'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Comparison - Crop Yield Prediction', fontsize=14, fontweight='bold')

axes[0].bar(models_list, results['RMSE'], color=palette)
axes[0].set_title('RMSE (lower is better)')
axes[0].set_ylabel('RMSE (Tonnes/Hectare)')
axes[0].tick_params(axis='x', rotation=20)
for i, v in enumerate(results['RMSE']):
    axes[0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].bar(models_list, results['MAE'], color=palette)
axes[1].set_title('MAE (lower is better)')
axes[1].set_ylabel('MAE (Tonnes/Hectare)')
axes[1].tick_params(axis='x', rotation=20)
for i, v in enumerate(results['MAE']):
    axes[1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[2].bar(models_list, results['R2'], color=palette)
axes[2].set_title('R2 Score (higher is better)')
axes[2].set_ylabel('R2 Score')
axes[2].tick_params(axis='x', rotation=20)
axes[2].set_ylim(0, 1.05)
for i, v in enumerate(results['R2']):
    axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'    Chart saved -> {chart_path}')

# ---- DONE -------------------------------------------------------------------
r2_val   = results.loc[best_idx, 'R2']
rmse_val = results.loc[best_idx, 'RMSE']
mae_val  = results.loc[best_idx, 'MAE']

print('\n' + '=' * 60)
print('  PIPELINE COMPLETE - ALL MODELS SAVED')
print(f'  Best model : {best_model_name}')
print(f'  R2         : {r2_val:.6f}')
print(f'  RMSE       : {rmse_val:.4f}')
print(f'  MAE        : {mae_val:.4f}')
print('=' * 60)
