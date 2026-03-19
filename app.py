import streamlit as st

from medical_policy_analytics.data import load_data
from medical_policy_analytics.ui.views import render_population_health

# --- PAGE CONFIG ---
st.set_page_config(page_title="Population Health Analytics", layout="wide")

# --- LOAD DATA ---
cdc_df = load_data()

# --- RENDER MAIN VIEW ---
render_population_health(cdc_df)
