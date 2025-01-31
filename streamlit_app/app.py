import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from helpers.data_utils import load_model_config, load_model, load_data
from helpers.scenario_utils import apply_scenario_modifications
from helpers.forecaster import run_scenario_forecast
from helpers.multistep_forecaster import forecast_multi_step_lightgbm, forecast_multi_step_sarima



st.set_page_config(page_title="Water Consumption Dashboard", layout="wide")

config_path = os.path.join("streamlit_app", "config", "model_config.yaml")
config = load_model_config(config_path)
model_options = config["model_options"]

model_names = [m["name"] for m in model_options]
selected_model_name = st.sidebar.selectbox("Select Model", model_names)

selected_model_info = next((m for m in model_options if m["name"] == selected_model_name), None)
if not selected_model_info:
    st.error("No model info found in config.")
    st.stop()

model_path = selected_model_info["model_path"]
data_path  = selected_model_info["data_path"]
expected_features = selected_model_info["expected_features"]

@st.cache_data
def get_data(path: str):
    return load_data(path) 

@st.cache_resource
def get_model(path: str):
    return load_model(path)

df = get_data(data_path)
model = get_model(model_path)

### TAB SETUP ###

tab1, tab2, tab3 = st.tabs(["Overview", "Forecast Scenarios", "Multi-Step-Forecaster"])

### TAB 1 ###

with tab1:
    st.header(f"Overview - {selected_model_name}")
    st.dataframe(df.head())  # Just to preview

    # Basic checks
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("DataFrame index is not a DateTimeIndex. Ensure 'Datum' is parsed.")
        st.stop()
    if "Wasserverbrauch" not in df.columns:
        st.error("Column 'Wasserverbrauch' not found.")
        st.stop()

    # Train: up to 2022, Test: 2023
    train_data = df[df.index.year <= 2022]
    test_data  = df[df.index.year == 2023]

    X_train = train_data[expected_features]
    y_train = train_data["Wasserverbrauch"]
    X_test  = test_data[expected_features]
    y_test  = test_data["Wasserverbrauch"]

    # Baseline model prediction on real 2023 data
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse= mse**0.5
    r2  = r2_score(y_test, y_pred)

    st.subheader("Test set metrics (2023)")
    st.write(f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f} | **R²:** {r2:.2f}")

    # Plot entire historical
    fig_all = px.line(
        df, x=df.index, y="Wasserverbrauch", 
        title="Water Consumption Over Time"
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Plot Observed vs Pred (2023)
    fig_test = px.line(
        test_data, x=test_data.index, y="Wasserverbrauch",
        title="Test 2023: Observed vs. Predicted"
    )
    fig_test.add_scatter(
        x=test_data.index, y=y_pred, mode='lines',
        name='Predicted', line=dict(color='red')
    )
    st.plotly_chart(fig_test, use_container_width=True)


### TAB 2 ###

from helpers.scenario_utils import apply_scenario_modifications
from helpers.forecaster import run_scenario_forecast

with tab2:
    st.header("Scenario-Based Forecast (2023)")

    # split data
    train_data_scenario = df[df.index.year <= 2022].copy()
    test_data_scenario  = df[df.index.year == 2023].copy()

    if test_data_scenario.empty:
        st.warning("No data found for 2023. Cannot do scenario analysis.")
        st.stop()

    #Sliders for Weather
    st.markdown("#### Weather Scenario")
    temp_offset = st.slider("Temperature Offset (°C)", -5.0, 5.0, 0.0, 0.5)
    rain_factor = st.slider("Rain Factor", 0.0, 2.0, 1.0, 0.1)
    rad_factor  = st.slider("Radiation Factor", 0.5, 2.0, 1.0, 0.1)
    temp_noise  = st.slider("Daily Temp Noise (°C)", 0.0, 5.0, 0.0, 0.1)
    rain_noise  = st.slider("Daily Rain Noise (min)", 0.0, 30.0, 0.0, 1.0)
    rad_noise   = st.slider("Daily Radiation Noise (W/m²)", 0.0, 50.0, 0.0, 1.0)

    # sliders for Population
    st.markdown("#### Population Scenario")
    c1, c2 = st.columns(2)
    with c1:
        geburten_change = st.slider("Geburten +/-", -10.0, 10.0, 0.0, 1.0)
        todesfaelle_change = st.slider("Todesfälle +/-", -10.0, 10.0, 0.0, 1.0)
    with c2:
        zuzuege_change = st.slider("Zuzüge +/-", -20.0, 20.0, 0.0, 1.0)
        wegzuege_change = st.slider("Wegzüge +/-", -20.0, 20.0, 0.0, 1.0)

    st.markdown("#### Noise for Population")
    c3, c4 = st.columns(2)
    with c3:
        wegzuege_noise = st.slider("Wegzüge Noise", 0.0, 5.0, 0.0, 0.1)
        zuzuege_noise  = st.slider("Zuzüge Noise", 0.0, 5.0, 0.0, 0.1)
    with c4:
        geburten_noise = st.slider("Geburten Noise", 0.0, 2.0, 0.0, 0.1)
        todesfaelle_noise = st.slider("Todesfälle Noise", 0.0, 2.0, 0.0, 0.1)

    # Build scenario dicts
    scenario_params = {
        "temp_offset": temp_offset,
        "rain_factor": rain_factor,
        "rad_factor":  rad_factor,
        "temp_noise":  temp_noise,
        "rain_noise":  rain_noise,
        "rad_noise":   rad_noise
    }
    population_scenarios = {
        "wegzuege_offset": wegzuege_change,
        "zuzuege_offset":  zuzuege_change,
        "geburten_offset": geburten_change,
        "todesfaelle_offset": todesfaelle_change,
        "wegzuege_noise": wegzuege_noise,
        "zuzuege_noise":  zuzuege_noise,
        "geburten_noise": geburten_noise,
        "todesfaelle_noise": todesfaelle_noise
    }

    # Apply scenario modifications
    scenario_test_df = apply_scenario_modifications(
        test_data_scenario,
        scenario_params,
        population_scenarios,
        weather_cols=["T_C", "RainDur_min", "StrGlo_W/m2"],
        population_cols=["Geburten","Todesfälle","Zuzüge","Wegzüge"]
    )

    # Predict
    from helpers.forecaster import run_scenario_forecast
    scenario_results_df, scenario_metrics = run_scenario_forecast(
        model, 
        scenario_test_df, 
        expected_features, 
        target_col="Wasserverbrauch"
    )

    st.subheader("Scenario-based Forecast Metrics (2023)")
    st.write(f"**MAE:** {scenario_metrics['mae']:.2f}")
    st.write(f"**RMSE:** {scenario_metrics['rmse']:.2f}")
    st.write(f"**R²:** {scenario_metrics['r2']:.2f}")

    # Plot scenario vs actual
    fig_scenario = px.line(
        scenario_results_df,
        x=scenario_results_df.index,
        y="Actual",
        title="Scenario Forecast vs Actual (2023)"
    )
    fig_scenario.add_scatter(
        x=scenario_results_df.index,
        y=scenario_results_df["Forecast"],
        mode='lines',
        name='Scenario Forecast',
        line=dict(color='red')
    )
    st.plotly_chart(fig_scenario, use_container_width=True)

    # Show final table
    st.write("**Scenario-based Forecast Table (2023)**")
    st.dataframe(scenario_results_df)

### TAB 3 ### TESTING
    
with tab3:
    st.header("Short-Horizon Multi-Step Forecast")
    st.write("We do a short forecast so errors don't accumulate too heavily.")

    # Let user pick how many days to forecast
    forecast_days = st.slider("Days to forecast", 1, 30, 7)

    # If the user selected a LightGBM model:
    if "lightgbm" in selected_model_name.lower():
        st.write("**Using multi-step approach for LightGBM**")
        full_df = df.copy()  
        multi_forecast_df = forecast_multi_step_lightgbm(
            model=model,
            df=full_df,  
            expected_features=expected_features,
            forecast_days=forecast_days
        )

    # If the user selected a SARIMA model:
    elif "sarima" in selected_model_name.lower():
        st.write("**Using multi-step SARIMA approach**")
        multi_forecast_df = forecast_multi_step_sarima(
            model=model,
            df=df,
            forecast_days=forecast_days,
            exog=None  
        )
    else:
        st.warning("No multi-step approach defined for this model.")
        st.stop()

    st.subheader("Multi-step Forecast Results")
    st.write(multi_forecast_df.head())

    fig_multi = px.line(
        multi_forecast_df, 
        x=multi_forecast_df.index, 
        y="Forecast",
        title=f"Short-Horizon {forecast_days}-Day Forecast"
    )
    st.plotly_chart(fig_multi, use_container_width=True)