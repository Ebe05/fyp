"""View wiring: Population Health and Hospital Operations."""

import pandas as pd
import plotly.express as px
import streamlit as st

from medical_policy_analytics.ui.budget_priority import render_budget_priority_tab
from medical_policy_analytics.ui.disease_analysis import render_disease_analysis_tab
from medical_policy_analytics.ui.overview import render_overview_tab
from medical_policy_analytics.ui.policy_rules import render_policy_rules_tab
from medical_policy_analytics.ui.target import render_target_tab


def view_1_population_health(df):
    """Population Health view: tabs Overview, Disease Analysis, Policy Rules, Find Your Target."""
    st.header("📍 View 1: Comparative Population Analytics")
    st.markdown("---")

    tab_overview, tab_disease, tab_policy, tab_target = st.tabs([
        "📊 Overview", "🔬 Disease Analysis", "🔍 Policy Rules", "🎯 Find Your Target"
    ])

    with tab_overview:
        render_overview_tab(df.copy())

    with tab_disease:
        render_disease_analysis_tab(df.copy())

    with tab_policy:
        render_policy_rules_tab(df.copy())

    with tab_target:
        render_target_tab(df.copy())


def render_hospital_operations_view(hosp_df):
    """Hospital Operations view: Overview tab + Budget Priority tab."""
    st.header("📍 View 2: Hospital Efficiency & Resource Allocation")
    st.markdown("---")

    tab_overview, tab_budget = st.tabs(["📊 Overview", "💰 Budget Priority"])

    with tab_overview:
        st.markdown("### Hospital Operations Overview")
        st.markdown("Analyzing clinical data to optimize hospital support and spending.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Encounters", f"{len(hosp_df):,}")
        with col2:
            avg_stay = hosp_df['time_in_hospital'].mean()
            st.metric("Avg. Stay Duration", f"{avg_stay:.1f} days")
        with col3:
            readmit_rate = (hosp_df['readmitted'] != 'NO').sum() / len(hosp_df) * 100
            st.metric("Readmission Rate", f"{readmit_rate:.1f}%")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(hosp_df, x="time_in_hospital", color="admission_type",
                               title="Distribution of Stay Duration by Admission Type",
                               labels={"time_in_hospital": "Days in Hospital",
                                      "admission_type": "Admission Type",
                                      "count": "Number of Encounters"},
                               barmode="group",
                               nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            readmit_by_type = pd.crosstab(hosp_df['admission_type'],
                                          hosp_df['readmitted'],
                                          normalize='index') * 100
            fig2 = px.bar(readmit_by_type, barmode='group',
                          title="Readmission Rate by Admission Type (%)",
                          labels={"value": "Percentage", "readmitted": "Readmission Status"})
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Resource Utilization Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.box(hosp_df, x="readmitted", y="num_medications", color="readmitted",
                          title="Number of Medications by Readmission Status",
                          labels={"num_medications": "Number of Medications",
                                 "readmitted": "Readmission Status"})
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            fig4 = px.box(hosp_df, x="readmitted", y="num_lab_procedures", color="readmitted",
                          title="Number of Lab Procedures by Readmission Status",
                          labels={"num_lab_procedures": "Number of Lab Procedures",
                                 "readmitted": "Readmission Status"})
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Stay Duration Analysis")
        fig5 = px.histogram(hosp_df, x="time_in_hospital",
                            title="Overall Distribution of Hospital Stay Duration",
                            labels={"time_in_hospital": "Days in Hospital", "count": "Number of Encounters"},
                            nbins=30)
        st.plotly_chart(fig5, use_container_width=True)

    with tab_budget:
        render_budget_priority_tab(hosp_df)
