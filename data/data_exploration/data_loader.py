import pandas as pd
import numpy as np

def load_and_clean_data(file_path, with_lag_features=False, lag_days=7):

    # Load the data
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['Datum'], dayfirst=True)
    
    # Calculate IQR and filter out outliers
    Q1 = df['Wasserverbrauch'].quantile(0.25)
    Q3 = df['Wasserverbrauch'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df['Wasserverbrauch'] < lower_bound) | (df['Wasserverbrauch'] > upper_bound)
    df['Wasserverbrauch'] = df['Wasserverbrauch'].where(~outliers).interpolate(method='linear')
    
    # Ensure the 'Datum' is in datetime format
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    
    # Set 'Datum' as index
    df.set_index('Datum', inplace=True)
    
    # Convert the rest of the columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Create dummy variables for weekdays
    df['is_saturday'] = (df.index.dayofweek == 5).astype(int)
    df['is_sunday'] = (df.index.dayofweek == 6).astype(int)
    
    # If lag features is true
    if with_lag_features:
        for i in range(1, lag_days + 1):
            df[f'lag_{i}'] = df['Wasserverbrauch'].shift(i)
        
        # Add rolling statistics
        df['rolling_mean_3'] = df['Wasserverbrauch'].rolling(window=3).mean()
        df['rolling_mean_7'] = df['Wasserverbrauch'].rolling(window=7).mean()
        df['rolling_std_3'] = df['Wasserverbrauch'].rolling(window=3).std()
        
        # Add month and weekday features
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        
        # Drop rows with NaN values after creating lag features
        df = df.dropna()
    
    return df

def load_streamlit_data(file_path, with_lag_features=False, lag_days=7):

    # Load the data
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['Datum'], dayfirst=True)
    
    # Calculate IQR and filter out outliers
    Q1 = df['Wasserverbrauch'].quantile(0.25)
    Q3 = df['Wasserverbrauch'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df['Wasserverbrauch'] < lower_bound) | (df['Wasserverbrauch'] > upper_bound)
    df['Wasserverbrauch'] = df['Wasserverbrauch'].where(~outliers).interpolate(method='linear')
    
    # Ensure the 'Datum' column is in datetime format
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    
    # Sort data by date
    df = df.sort_values('Datum')
    
    # Convert the rest of the columns to numeric
    for col in df.columns:
        if col not in ['Datum', 'Date_Column']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create dummy variables for weekdays
    df['is_saturday'] = (df['Datum'].dt.dayofweek == 5).astype(int)
    df['is_sunday'] = (df['Datum'].dt.dayofweek == 6).astype(int)
    
    # If lag features is true
    if with_lag_features:
        for i in range(1, lag_days + 1):
            df[f'lag_{i}'] = df['Wasserverbrauch'].shift(i)
        
        # Add rolling statistics
        df['rolling_mean_3'] = df['Wasserverbrauch'].rolling(window=3).mean()
        df['rolling_mean_7'] = df['Wasserverbrauch'].rolling(window=7).mean()
        df['rolling_std_3'] = df['Wasserverbrauch'].rolling(window=3).std()
        
        # Add month and weekday features
        df['month'] = df['Datum'].dt.month
        df['weekday'] = df['Datum'].dt.weekday
        
        # Drop rows with NaN values
        df = df.dropna()

    return df


