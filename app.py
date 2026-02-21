import streamlit as st

from medical_policy_analytics.data import load_data
from medical_policy_analytics.ui.views import render_hospital_operations_view, view_1_population_health

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Policy Analytics", layout="wide")

# --- LOAD DATA ---
cdc_df, hosp_df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Policy Navigation")
view = st.sidebar.radio("Go to:", ["Population Health (CDC)", "Hospital Operations (UCI)"])

# --- VIEW 1: POPULATION HEALTH ---
if view == "Population Health (CDC)":
    view_1_population_health(cdc_df)

# --- VIEW 2: HOSPITAL OPERATIONS ---
elif view == "Hospital Operations (UCI)":
    render_hospital_operations_view(hosp_df)
