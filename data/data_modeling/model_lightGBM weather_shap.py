import sys
import os
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap

# Add the parent directory to system path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import the data loader
from data_exploration.data_loader import load_and_clean_data

# Define the base path
base_path = os.getcwd()

# Specify the file path
input_file_path = os.path.join(base_path, "water_data", "output", "water_consumption_2015_2023_normalized.csv")

# Load and clean the data
df_cleaned = load_and_clean_data(input_file_path, with_lag_features=True, lag_days=7)

# Create 'days_since_rain' feature
def calculate_days_since_rain(df):
    days_since_rain = []
    count = 0
    for rain in df['RainDur_min']:
        if rain > 0:
            count = 0
        else:
            count += 1
        days_since_rain.append(count)
    df['days_since_rain'] = days_since_rain
    return df

df_cleaned = calculate_days_since_rain(df_cleaned)

# Create lag features for weather columns
weather_columns = ['RainDur_min', 'StrGlo_W/m2', 'T_C', 'T_max_h1_C', 'p_hPa']
for col in weather_columns:
    for lag in range(1, 8):
        df_cleaned[f'{col}_lag_{lag}'] = df_cleaned[col].shift(lag)

# Drop NaN values created by lagging
df_cleaned.dropna(inplace=True)

# Ensure the date column is parsed and used for splitting
df_cleaned.index = pd.to_datetime(df_cleaned.index)

# Split data into training and testing
train_data = df_cleaned[df_cleaned.index.year <= 2022]
test_data = df_cleaned[df_cleaned.index.year == 2023]

# Define the training and testing sets
feature_columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                   'is_saturday', 'is_sunday', 'month', 'weekday',
                   'rolling_mean_3', 'rolling_mean_7', 'rolling_std_3',
                   'days_since_rain'] + [f'{col}_lag_{i}' for col in weather_columns for i in range(1, 8)]

X_train = train_data[feature_columns]
y_train = train_data['Wasserverbrauch']

X_test = test_data[feature_columns]
y_test = test_data['Wasserverbrauch']

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter tuning for LightGBM model
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 40, 50],
    'max_depth': [-1, 10, 20, 30],
    'min_child_samples': [5, 10, 20]
}

lgbm_model = LGBMRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model after tuning
best_lgbm_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_lgbm_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R²): {r2}')

# SHAP Analysis
explainer = shap.Explainer(best_lgbm_model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)

# Path for saving the model
model_save_path = os.path.join(base_path, 'models', 'best_lightgbm_model_weather_v1.pkl')

# Save the trained model using joblib
joblib.dump(best_lgbm_model, model_save_path)

print(f"Model saved to {model_save_path}")
