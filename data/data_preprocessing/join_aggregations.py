import pandas as pd
import os

def merge_files(file1_path, file2_path, output_path):
    # Read the files
    df1 = pd.read_csv(file1_path, delimiter=";")
    df2 = pd.read_csv(file2_path, delimiter=";")
    
    # Convert 'Datum' column to datetime
    df1['Datum'] = pd.to_datetime(df1['Datum'], format='%Y-%m-%d')
    df2['Datum'] = pd.to_datetime(df2['Datum'], format='%Y-%m-%d')

    # Merge the two df on 'Datum' column
    merged_df = pd.merge(df1, df2, on='Datum', how='outer')

    # Generate the output filenames
    file1_name = os.path.basename(file1_path).split('.')[0]
    file2_name = os.path.basename(file2_path).split('.')[0]
    output_filename = f"{file1_name}_join_{file2_name}.csv"

    # Define the full output path
    output_path = os.path.join(output_folder, output_filename)

    # Save the merged data to the new CSV file
    merged_df.to_csv(output_path, sep=";", index=False)

    print(f"Merged file saved to: {output_path}")

# Define the base path
base_path = os.getcwd()

# Define the input and output folder paths
input_folder = os.path.join(base_path, 'water_data', 'output')
output_folder = os.path.join(base_path, 'water_data', 'output')

# Define the input file paths
file1_path = os.path.join(output_folder, 'water_consumption_2015_2023_monthly_normalized.csv')
file2_path = os.path.join(output_folder, 'population_monthly.csv')

# Call function
merge_files(file1_path, file2_path, output_folder)
