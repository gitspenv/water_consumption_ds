import pandas as pd
import joblib
import yaml
import os

def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, index_col=0, parse_dates=True, encoding='utf-8')
    return df

def load_model(model_path: str):
    return joblib.load(model_path)

def load_model_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config
