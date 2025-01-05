import sys
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import joblib

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Add the parent directory to system path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

#Import the data loader
from data_exploration.data_loader import load_and_clean_data

# Define the base path
base_path = os.getcwd()

#Specify the file path
input_file_path = os.path.join(
    base_path, 
    "water_data", 
    "output", 
    "water_consumption_2015_2023_normalized.csv"
)

df_cleaned = load_and_clean_data(
    file_path=input_file_path, 
    with_lag_features=True, 
    lag_days=7,
    add_time_features=True,
    add_weather_features=True
)

#  features
feature_cols = [
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_std_3',
    'is_saturday', 'is_sunday', 'month', 'weekday', 
    'day_of_year', 'week_of_year', 
    'sin_day_of_year', 'cos_day_of_year',
    'RainDur_min', 'T_C', 'StrGlo_W/m2',
    'T_C_rolling3', 'rained_today',
    'year', 'Wegzüge' ,'Zuzüge','Geburte','Todesfälle'
]

print(df_cleaned.head())

# ensure index is a DateTimeIndex
df_cleaned.index = pd.to_datetime(df_cleaned.index)

target_col = 'Wasserverbrauch'
relevant_columns = feature_cols + [target_col]
df_relevant = df_cleaned[relevant_columns]


processed_data_path = os.path.join(base_path, 'water_data', 'output', 'processed_data_with_lags_exo_time_pop_relevant.csv')
df_relevant.to_csv(processed_data_path)

train_data = df_cleaned[df_cleaned.index.year <= 2022]
test_data = df_cleaned[df_cleaned.index.year == 2023]

X_train = train_data[feature_cols]
y_train = train_data['Wasserverbrauch']

X_test  = test_data[feature_cols]
y_test  = test_data['Wasserverbrauch']

print(f"Training set: {X_train.shape[0]} rows ({X_train.index.min()} to {X_train.index.max()})")
print(f"Test set:     {X_test.shape[0]} rows ({X_test.index.min()} to {X_test.index.max()})")

tscv = TimeSeriesSplit(n_splits=5)

# hyperparameter grid
param_grid = {
    'n_estimators':      [100, 200, 300, 400],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'num_leaves':        [20, 31, 40, 50],
    'max_depth':         [-1, 10, 20, 30],
    'min_child_samples': [5, 10, 20]
}

lgbm_model = LGBMRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=lgbm_model, 
    param_grid=param_grid, 
    cv=tscv, 
    n_jobs=-1, 
    verbose=2
)
grid_search.fit(X_train, y_train)

best_lgbm_model = grid_search.best_estimator_

y_pred = best_lgbm_model.predict(X_test)

# metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mse**0.5
r2  = r2_score(y_test, y_pred)

print(f"Test MSE:    {mse:.2f}")
print(f"Test MAE:    {mae:.2f}")
print(f"Test RMSE:   {rmse:.2f}")
print(f"Test R^2:    {r2:.2f}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
print("\nSample of Actual vs Predicted:")
print(comparison.head())

# Plot Observed vs. Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Observed')
plt.plot(y_test.index, y_pred, label='Predicted', color='red')
plt.legend()
plt.title('LightGBM Predictions vs Observed Values (Test Set)')
plt.show()

# Plot Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, residuals, label='Residuals', color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.legend()
plt.title('Residuals (Observed - Predicted)')
plt.show()

# Feature importances
importances = best_lgbm_model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_names = [feature_cols[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), feat_names)
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()  # so the top feature is at the top
plt.show()

# Save the model
model_save_path = os.path.join(base_path, 'models', 'lightgbm_model_v1.5_exo_time.pkl')
joblib.dump(best_lgbm_model, model_save_path)
print(f"Model saved to {model_save_path}")
