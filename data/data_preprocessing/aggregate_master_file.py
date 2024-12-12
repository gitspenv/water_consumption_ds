import pandas as pd
import os

def aggregate_data(input_path, output_path, frequency='ME'):
    # Read the data
    df = pd.read_csv(input_path, delimiter=";")

    # Convert 'Datum' column to datetime
    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')

    # Set 'Datum' as the index
    df.set_index('Datum', inplace=True)

    # Convert all columns except 'Datum' to numeric
    df = df.apply(pd.to_numeric, errors='coerce', axis=1)

    # Perform the resampling and aggregation
    df_resampled = df.resample(frequency).agg({
        'Wasserverbrauch': 'sum',
        'Wegzüge': 'sum',
        'Zuzüge': 'sum',
        'Geburte': 'sum',
        'Todesfälle': 'sum',
        'RainDur_min': 'sum',
        'StrGlo_W/m2': 'sum',
        'T_C': 'mean',
        'T_max_h1_C': 'mean',
        'p_hPa': 'mean'
    })

    # Round to 2 decimal places
    df_resampled[['T_C', 'T_max_h1_C', 'p_hPa']] = df_resampled[['T_C', 'T_max_h1_C', 'p_hPa']].round(2)

    # Save
    df_resampled.to_csv(output_path, sep=";", index=True)

# Define the base path
base_path = os.getcwd()

input_path = os.path.join(base_path, "water_data", "output" , "water_consumption_2015_2023_normalized.csv")
output_dir = os.path.join(base_path, "water_data", "output")

# Output file paths
output_path_monthly = os.path.join(output_dir, "water_consumption_2015_2023_monthly_normalized.csv")
output_path_weekly = os.path.join(output_dir, "water_consumption_2015_2023_weekly_normalized.csv")
output_path_daily = os.path.join(output_dir, "water_consumption_2015_2023_daily_normalized.csv")

# Perform aggregations
aggregate_data(input_path, output_path_monthly, frequency='ME') # Monthly aggregation
aggregate_data(input_path, output_path_weekly, frequency='W') # Weekly aggregation
aggregate_data(input_path, output_path_daily, frequency='D') # Daily aggregation
