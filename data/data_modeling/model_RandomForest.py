import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import datetime

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

# Prepare the data for training
X = df_cleaned[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
               'is_saturday', 'is_sunday', 'month', 'weekday',
               'rolling_mean_3', 'rolling_mean_7', 'rolling_std_3']]
y = df_cleaned['Wasserverbrauch']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter tuning for RandomForest model
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model after tuning
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R²): {r2}')

# Print a few actual vs predicted values
print("\nActual vs Predicted Values:")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# Plot observed vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Observed')
plt.plot(y_test.index, y_pred, label='Predicted', color='red')
plt.legend()
plt.title('Random Forest Predictions vs Observed Values')
plt.show()

# Plot residuals (Actual - Predicted)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, residuals, label='Residuals', color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.legend()
plt.title('Residuals (Observed - Predicted)')
plt.show()

# Feature importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), features[indices])
plt.xlabel("Relative Importance")
plt.show()

# Path for saving the model
model_save_path = os.path.join(base_path, 'models', 'best_random_forest_model.pkl')

# Save the trained model using joblib
joblib.dump(best_rf_model, model_save_path)

print(f"Model saved to {model_save_path}")
