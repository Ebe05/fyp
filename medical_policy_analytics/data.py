"""Data loading for CDC population health dataset."""

import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """Load the CDC population health dataset."""
    cdc = pd.read_csv("fyp_data/data/diabetes_012_health_indicators_BRFSS2015_cleaned.csv")

    diabetes_map = {0.0: "No Diabetes", 1.0: "Prediabetes", 2.0: "Diabetes"}
    cdc['Diabetes_Status'] = cdc['Diabetes_012'].map(diabetes_map)

    return cdc
