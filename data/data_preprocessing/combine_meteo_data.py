import pandas as pd
import glob
import os

# Define the base path
base_path = os.getcwd()

# Specify the path with *. to read all csv files
path = os.path.join(base_path, "meteo_data", "*.csv")
all_files = glob.glob(path)

# Concatenate all csv files to df
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Pivot the df
df_pivoted = df.pivot_table(index=['Datum', 'Standort', 'Intervall'], 
                            columns=['Parameter', 'Einheit'], 
                            values='Wert').reset_index()

# Flatten the multi-index columns
df_pivoted.columns = [f"{i}_{j}" if j else i for i, j in df_pivoted.columns]

# Convert 'Datum' to datetime
df_pivoted['Datum'] = pd.to_datetime(df_pivoted['Datum']).dt.strftime('%d.%m.%Y')

# Filter out specific 'Standort' values
excluded_locations = ['Zch_Rosengartenstrasse', 'Zch_Schimmelstrasse']
df_filtered = df_pivoted[~df_pivoted['Standort'].isin(excluded_locations)]

# Drop columns
columns_to_drop = ['Intervall', 'Standort']  # Replace with the actual column names you want to drop
df_filtered = df_filtered.drop(columns=columns_to_drop, errors='ignore')

# Save the filtered df
output_path = os.path.join(base_path, 'water_data', 'input', 'filtered_meteo_data.csv')
df_filtered.to_csv(output_path, index=False)
