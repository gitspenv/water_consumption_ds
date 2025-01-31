import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# Import plotting utilities
from Eval_functions import plot_residuals_vs_predicted, plot_feature_importance, plot_mean_error_by_feature, plot_variance_by_category

# Load the model
model_save_path = r"models\best_lightgbm_model_weather_v1.pkl"
loaded_model = joblib.load(model_save_path)

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

# Preprocess test data (repeat necessary preprocessing steps)
feature_columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                   'is_saturday', 'is_sunday', 'month', 'weekday',
                   'rolling_mean_3', 'rolling_mean_7', 'rolling_std_3',
                   'days_since_rain'] + [f'{col}_lag_{i}' for col in weather_columns for i in range(1, 8)]
X_test = df_cleaned[feature_columns]
y_test = df_cleaned['Wasserverbrauch']

# Predict
y_pred = loaded_model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals vs predicted values
plot_residuals_vs_predicted(y_pred, residuals)

# Plot feature importance
feature_importances = loaded_model.feature_importances_
plot_feature_importance(feature_importances, X_test.columns)

# Calculate residuals
test_data['Prediction_Error'] = residuals

plot_mean_error_by_feature(test_data, 'weekday')

plot_mean_error_by_feature(test_data, 'month')

plot_variance_by_category(
    data=df_cleaned, 
    category_feature='weekday', 
    target_feature='Wasserverbrauch', 
    title="Variance of Water Consumption by Month"
)

# Improvement: Introduce holidays as a feature.