import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Add parent directory to system path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import the data loading function, exploration functions, and plot functions
from data_exploration.data_loader import load_and_clean_data
from data_exploration.data_exploration import perform_data_exploration
from data_exploration.plot_functions import plot_forecast

# Define the base path
base_path = os.getcwd()

# Specify the file path (can be modified to load different files)
input_file_path = os.path.join(base_path, "water_data", "output", "water_consumption_2015_2023_monthly_normalized.csv")

# Load and clean the data
df_cleaned = load_and_clean_data(input_file_path)

# Split the data into training and testing
train = df_cleaned[:round(len(df_cleaned)*0.7)]
test = df_cleaned[round(len(df_cleaned)*0.7):]

# Plot ACF and PACF on the original data
plot_acf(train['Wasserverbrauch'], lags=20)
plt.show()
plot_pacf(train['Wasserverbrauch'], lags=20)
plt.show()

# Fit the ARIMA model on training data
model_arima = ARIMA(train['Wasserverbrauch'], order=(2, 0, 2))  # No differencing
model_arima_fit = model_arima.fit()

# Create Prediction for ARIMA
prediction_arima = model_arima_fit.predict(start=test.index[0], end=test.index[-1])
df_cleaned['arimaPred'] = np.nan
df_cleaned.loc[test.index, 'arimaPred'] = prediction_arima

# Print MAE for ARIMA
print("ARIMA Model MAE:", mean_absolute_error(test['Wasserverbrauch'], prediction_arima))

# Fit the SARIMAX model on training data
model_sarimax = SARIMAX(train['Wasserverbrauch'], order=(2, 0, 2), seasonal_order=(2, 0, 2, 12))
model_sarimax_fit = model_sarimax.fit()

# Create Prediction for SARIMAX
prediction_sarimax = model_sarimax_fit.predict(start=test.index[0], end=test.index[-1])
df_cleaned['sarimaxPred'] = np.nan
df_cleaned.loc[test.index, 'sarimaxPred'] = prediction_sarimax

# Print MAE for SARIMAX
print("SARIMAX Model MAE:", mean_absolute_error(test['Wasserverbrauch'], prediction_sarimax))

# Plot the predictions
plt.figure(figsize=(10,6))
plt.plot(df_cleaned['Wasserverbrauch'], label='Actual')
plt.plot(df_cleaned['arimaPred'], label='ARIMA Prediction', linestyle='--')
plt.plot(df_cleaned['sarimaxPred'], label='SARIMAX Prediction', linestyle='--')
plt.legend()
plt.show()
