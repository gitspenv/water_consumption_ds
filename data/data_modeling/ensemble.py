import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# Import plotting utilities
from Eval_functions import plot_residuals_vs_predicted, plot_feature_importance, plot_mean_error_by_feature, plot_variance_by_category

# Load and preprocess the data
def preprocess_data():
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

    return df_cleaned

df_cleaned = preprocess_data()

# Split data into training and testing
train_data = df_cleaned[df_cleaned.index.year <= 2022]
test_data = df_cleaned[df_cleaned.index.year == 2023]

# Define features and target
feature_columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                   'is_saturday', 'is_sunday', 'month', 'weekday',
                   'rolling_mean_3', 'rolling_mean_7', 'rolling_std_3',
                   'days_since_rain'] + [f'{col}_lag_{i}' for col in ['RainDur_min', 'StrGlo_W/m2', 'T_C', 'T_max_h1_C', 'p_hPa'] for i in range(1, 8)]
X_train = train_data[feature_columns]
y_train = train_data['Wasserverbrauch']
X_test = test_data[feature_columns]
y_test = test_data['Wasserverbrauch']

# Define base models for ensemble
base_models = [
    ('lightgbm', LGBMRegressor(random_state=42)),
    ('random_forest', RandomForestRegressor(random_state=42)),
    ('linear_regression', LinearRegression())
]

# Define stacking regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5  # 5-fold cross-validation
)

# Train the ensemble model
stacking_model.fit(X_train, y_train)

# Save the ensemble model for reuse
ensemble_model_path = "models/ensemble_model.pkl"
joblib.dump(stacking_model, ensemble_model_path)

# Predict on the test set
y_pred = stacking_model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals vs predicted values
plot_residuals_vs_predicted(y_pred, residuals)

# Plot feature importance (LightGBM importance)
lightgbm_model = stacking_model.named_estimators_['lightgbm']
plot_feature_importance(lightgbm_model.feature_importances_, X_train.columns)

# Analyze errors
test_data['Prediction_Error'] = residuals
plot_mean_error_by_feature(test_data, 'weekday')
plot_mean_error_by_feature(test_data, 'month')

# Evaluate the ensemble model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R²): {r2:.4f}')

# Plot Predicted vs Actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Identity line
plt.title('Predicted vs Actual Water Consumption')
plt.xlabel('Actual Water Consumption')
plt.ylabel('Predicted Water Consumption')
plt.grid(True)
plt.show()
