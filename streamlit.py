from pygwalker.api.streamlit import StreamlitRenderer
import streamlit as st
import pandas as pd
import pygwalker as pyg
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import data loading and exploration functions
from data.data_exploration.data_loader import load_streamlit_data
from data.data_exploration.plot_functions import plot_aggregated_data

st.set_page_config(
    page_title="Water Consumption",
    layout="wide"
)

@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    df_manual = load_streamlit_data("water_data/input/water_consumption_2015_2023.csv")
    return StreamlitRenderer(df_manual, spec="./gw_config.json", spec_io_mode="rw")

renderer = get_pyg_renderer()

tab1, tab2, tab3 = st.tabs(["Manual Exploration", "Python Visualization", "Prediction"])

with tab1:
    renderer.explorer()

with tab2:
    st.header("Python Visualization")

with tab3:
    st.header("Prediction")
