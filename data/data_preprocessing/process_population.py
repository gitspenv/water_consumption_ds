import pandas as pd
import os

# Define the base path
base_path = os.getcwd()

# Define the input and output paths
input_path = os.path.join(base_path, 'water_data', 'input', 'bevölkerung_monatlich.csv')
output_path = os.path.join(base_path, 'water_data', 'output', 'population_monthly.csv')

# Read the data
df = pd.read_csv(input_path, delimiter=";")

# Convert the 'Datum' column to datetime
df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')

# Extract the month from the 'Datum' column
df['Monat'] = df['Datum'].dt.to_period('M')

# Pivot the data to separate Swiss and foreign populations
pivoted_data = (
    df.pivot_table(
        index='Monat', 
        columns='Herkunft', 
        values='Bevölkerung', 
        aggfunc='sum'
    )
    .reset_index()
)

# Rename columns
pivoted_data.columns = ['Monat', 'Bevölkerung_Swiss', 'Bevölkerung_Foreign']

# Calculate the total population
pivoted_data['Bevölkerung_total'] = pivoted_data['Bevölkerung_Swiss'] + pivoted_data['Bevölkerung_Foreign']

# Convert the 'Monat' to the last day of each month
pivoted_data['Datum'] = pivoted_data['Monat'].dt.to_timestamp('M')

# Drop the 'Monat' column
pivoted_data.drop(columns=['Monat'], inplace=True)

# Reorder columns
pivoted_data = pivoted_data[['Datum', 'Bevölkerung_Swiss', 'Bevölkerung_Foreign', 'Bevölkerung_total']]

# Sort the data by 'Datum'
pivoted_data.sort_values(by='Datum', ascending=True, inplace=True)

# Save the aggregated data
pivoted_data.to_csv(output_path, index=False, sep=";")

print("Aggregated data saved to:", output_path)
