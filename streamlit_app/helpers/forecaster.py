import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_scenario_forecast(model, scenario_df, expected_features, target_col="Wasserverbrauch"):

    X_scenario = scenario_df[expected_features]
    y_scenario_actual = scenario_df[target_col].copy()

    y_scenario_pred = model.predict(X_scenario)

    # Build final DataFrame
    results_df = pd.DataFrame({
        "Actual": y_scenario_actual,
        "Forecast": y_scenario_pred
    }, index=scenario_df.index)

    # Compute metrics
    mse = mean_squared_error(y_scenario_actual, y_scenario_pred)
    mae = mean_absolute_error(y_scenario_actual, y_scenario_pred)
    rmse= mse**0.5
    r2  = r2_score(y_scenario_actual, y_scenario_pred)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    return results_df, metrics
