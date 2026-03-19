"""View wiring: Population Health analytics."""

import streamlit as st

from medical_policy_analytics.ui.disease_analysis import render_disease_analysis_tab
from medical_policy_analytics.ui.overview import render_overview_tab
from medical_policy_analytics.ui.policy_rules import render_policy_rules_tab
from medical_policy_analytics.ui.target import render_target_tab


def render_population_health(df):
    """Population Health view: tabs Overview, Disease Analysis, Policy Rules, Find Your Target."""
    st.header("Population Health Analytics")
    st.caption("Analyze disease prevalence, risk factors, and identify target populations for policy interventions.")
    st.divider()

    tab_overview, tab_disease, tab_policy, tab_target = st.tabs([
        "Overview", "Disease Analysis", "Policy Rules", "Find Your Target"
    ])

    with tab_overview:
        render_overview_tab(df.copy())

    with tab_disease:
        render_disease_analysis_tab(df.copy())

    with tab_policy:
        render_policy_rules_tab(df.copy())

    with tab_target:
        render_target_tab(df.copy())
