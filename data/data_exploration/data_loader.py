import pandas as pd
import numpy as np

def load_and_clean_data(file_path, 
                        with_lag_features=False, 
                        lag_days=7,
                        add_time_features=True,
                        add_weather_features=True
                       ):
    """
    Loads daily water consumption data from CSV, filters outliers,
    optionally adds time features, weather features, and lag features.

    1. Reads the CSV with ';' delimiter and parse 'Datum' as dates.
    2. Filters out outliers in 'Wasserverbrauch' via IQR.
    3. Ensures a daily DateTimeIndex (freq='D') by filling missing dates.
    4. Optionally adds time-based columns (month, weekday, year, etc.).
    5. Optionally adds weather-based columns (rolling T_C, rained_today).
    6. Optionally adds lag features for Wasserverbrauch.

    Returns:
        df (pd.DataFrame): A DataFrame with daily rows, cleaned and ready for modeling.
    """

    # 1) Load the data
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['Datum'], dayfirst=False)

    # 2) Filter out outliers in Wasserverbrauch
    Q1 = df['Wasserverbrauch'].quantile(0.25)
    Q3 = df['Wasserverbrauch'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df['Wasserverbrauch'] < lower_bound) | (df['Wasserverbrauch'] > upper_bound)
    # Interpolate outlier values
    df['Wasserverbrauch'] = df['Wasserverbrauch'].where(~outliers).interpolate(method='linear')

    # 3) Set DateTimeIndex
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    df.set_index('Datum', inplace=True)

    # Ensure numeric columns
    df = df.apply(pd.to_numeric, errors='coerce')

    # Sort by date and fill missing daily dates
    df = df.sort_index()
    # Convert to daily frequency, filling missing days with NaNs
    df = df.asfreq('D')

    # Interpolate or fill any newly created NaNs for Wasserverbrauch
    # (for days that weren't in the CSV but now exist in asfreq('D'))
    if df['Wasserverbrauch'].isna().any():
        df['Wasserverbrauch'] = df['Wasserverbrauch'].interpolate(method='linear')

    # 4) Basic day-of-week dummies
    df['is_saturday'] = (df.index.dayofweek == 5).astype(int)
    df['is_sunday']   = (df.index.dayofweek == 6).astype(int)

    # 5) Optionally add extended time features
    if add_time_features:
        df['month']   = df.index.month
        df['weekday'] = df.index.weekday
        df['year']    = df.index.year

        df['day_of_year']  = df.index.dayofyear
        # Some daily data might cross leap years, so day_ofyear can be up to 366
        df['week_of_year'] = df.index.isocalendar().week.astype(int)

        # cyclical transforms
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)

    # 6) Optionally add weather features
    if add_weather_features:
        if 'T_C' in df.columns:
            df['T_C_rolling3'] = df['T_C'].rolling(window=3).mean()
        if 'RainDur_min' in df.columns:
            df['rained_today'] = (df['RainDur_min'] > 0).astype(int)

    # 7) Optionally add lag features for Wasserverbrauch
    if with_lag_features:
        for i in range(1, lag_days + 1):
            df[f'lag_{i}'] = df['Wasserverbrauch'].shift(i)
        
        df['rolling_mean_3'] = df['Wasserverbrauch'].rolling(window=3).mean()
        df['rolling_mean_7'] = df['Wasserverbrauch'].rolling(window=7).mean()
        df['rolling_std_3']  = df['Wasserverbrauch'].rolling(window=3).std()

        # Drop rows that have become NaN after shifting or rolling
        df = df.dropna()

    return df
