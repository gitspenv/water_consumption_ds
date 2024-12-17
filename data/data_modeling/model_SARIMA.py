import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
input_file_path = os.path.join(base_path, "water_data", "output" , "water_consumption_2015_2023_monthly_normalized_join_population_monthly.csv")

# Load and clean the data
df_cleaned = load_and_clean_data(input_file_path)

# Perform data exploration
perform_data_exploration(df_cleaned)

# Set the index to a period (monthly) for time series analysis
df_cleaned.index = pd.DatetimeIndex(df_cleaned.index).to_period('M')

# Split the data into training and testing
train_end = pd.Period('2020-12', freq='M')
train_data = df_cleaned[df_cleaned.index <= train_end]
test_data = df_cleaned[df_cleaned.index > train_end]

# Define target and features for training
endog_train = train_data['Wasserverbrauch']
relevant_columns = ['Todesfälle', 'T_C']
exog_train = train_data[relevant_columns]

# Clean the exogenous variables for training
exog_train_clean = exog_train.replace([np.inf, -np.inf], np.nan).dropna()

# Ensure target variable has no missing values
endog_train_clean = endog_train[exog_train_clean.index]

# Fit the SARIMA model on training data
model = sm.tsa.SARIMAX(endog_train_clean, exog=exog_train_clean,
                       order=(3, 1, 0),  # AR, I, MA
                       seasonal_order=(3, 1, 0, 12),  # AR, I, MA, seasonality
                       enforce_stationarity=False,
                       enforce_invertibility=False)
results = model.fit(disp=False)

# Display the results summary
print(results.summary())

# Plot diagnostics to check the residuals
results.plot_diagnostics(figsize=(15, 10))
plt.show()

# Forecast for the test period (2023)
future_months = len(test_data)

# Create exogenous variables for the forecast period (using test data features)
future_exog = pd.DataFrame(
    {col: test_data[col] for col in exog_train_clean.columns},
    index=test_data.index
)

# Forecasting
forecast = results.get_forecast(steps=future_months, exog=future_exog)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot the observed data and the forecast
plot_forecast(train_data, forecast_mean, forecast_ci, observed_column='Wasserverbrauch')

# Evaluate the model performance on the test set
y_true = test_data['Wasserverbrauch']
mae = mean_absolute_error(y_true, forecast_mean)
mse = mean_squared_error(y_true, forecast_mean)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
