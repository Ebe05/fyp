"""Disease Analysis tab: disease-specific analysis and what-if simulation."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from medical_policy_analytics.analytics.risk import (
    calculate_disease_risk_score,
    calculate_high_risk,
    simulate_combined_intervention,
)
from medical_policy_analytics.config import DISEASE_CONFIG, DISEASE_SCORE_CUTOFFS


def render_disease_analysis_tab(df):
    """Render the Disease Analysis tab with disease-specific features"""

    df['HighBP'] = pd.to_numeric(df['HighBP'], errors='coerce')
    df['Diabetes_012'] = pd.to_numeric(df['Diabetes_012'], errors='coerce')
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['HighChol'] = pd.to_numeric(df['HighChol'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # SECTION 1: Compact Filter Section (2-column layout)
    filter_col1, filter_col2 = st.columns([1, 2])
    
    with filter_col1:
        selected_disease = st.selectbox(
            "Disease to Analyze",
            options=list(DISEASE_CONFIG.keys()),
            help="Choose which health condition to analyze"
        )
    
    with filter_col2:
        income_labels = {
            1: '1: <$10k', 2: '2: $10-15k', 3: '3: $15-20k', 4: '4: $20-25k',
            5: '5: $25-35k', 6: '6: $35-50k', 7: '7: $50-75k', 8: '8: $75k+'
        }
        income_options = sorted(df['Income'].unique())
        income_display = [income_labels.get(int(i), str(int(i))) for i in income_options]
        
        income_filter = st.multiselect(
            "Target Income Brackets (select all that apply)",
            options=income_options,
            default=income_options,
            format_func=lambda x: income_labels.get(int(x), str(int(x))),
            key="income_filter",
            help="Filter population by income level"
        )

    disease_info = DISEASE_CONFIG[selected_disease]
    target_col = disease_info["column"]

    mask = df['Income'].isin(income_filter)
    filtered_df = df[mask].copy()

    if len(filtered_df) == 0:
        st.warning("No data available for the selected income brackets. Please select at least one income bracket.")
        return

    # Calculate metrics
    cutoff = float(DISEASE_SCORE_CUTOFFS.get(selected_disease, disease_info["risk_threshold"]))
    filtered_df["disease_risk_score"] = calculate_disease_risk_score(
        filtered_df, selected_disease, target_col, calibrate_intercept=True
    )
    
    secondary_high_risk = calculate_high_risk(filtered_df, selected_disease, target_col, mode="secondary")
    primary_high_risk = calculate_high_risk(filtered_df, selected_disease, target_col, mode="primary")
    secondary_count = secondary_high_risk.sum()
    primary_count = primary_high_risk.sum()
    total_high_risk = secondary_count + primary_count
    total_count = len(filtered_df)
    disease_cases = (filtered_df[target_col] > 0).sum()
    prevalence = (filtered_df[target_col] > 0).mean() * 100

    st.divider()

    # SECTION 2: Population Overview (in container)
    with st.container():
        st.markdown(f"##### Population Overview: {selected_disease}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Population", f"{total_count:,}")
        with col2:
            st.metric(f"{selected_disease} Cases", f"{disease_cases:,}")
        with col3:
            st.metric("Prevalence Rate", f"{prevalence:.1f}%")

    st.divider()

    # SECTION 3: High-Risk Targeting (in container)
    with st.container():
        st.markdown(f"##### High-Risk Targeting (Score ≥ {cutoff:.0f})")
        target_col1, target_col2, target_col3 = st.columns(3)
        with target_col1:
            st.metric(
                "Primary Prevention",
                f"{primary_count:,}",
                help="At-risk individuals WITHOUT the disease (prevent onset)"
            )
        with target_col2:
            st.metric(
                "Secondary Prevention",
                f"{secondary_count:,}",
                help="High-risk individuals WITH the disease (prevent complications)"
            )
        with target_col3:
            st.metric(
                "Total Targetable",
                f"{total_high_risk:,}",
                help="Combined intervention pool"
            )
        
        st.caption(
            f"**Primary:** No disease + score ≥ {cutoff:.0f} (prevent onset) | "
            f"**Secondary:** Has disease + score ≥ {cutoff:.0f} (prevent complications)"
        )

    st.divider()

    # SECTION 4: What-If Policy Simulation (collapsible with presets)
    st.markdown("### What-If Policy Simulation")
    st.caption("Simulate the combined impact of multiple policy interventions on the high-risk population.")

    disease_relevance = {
        "Diabetes": {"bmi": "high", "exercise": "high", "diet": "high", "smoking": "medium", "cholesterol": "medium", "alcohol": "low", "bp": "medium"},
        "Heart Disease": {"bmi": "high", "exercise": "high", "diet": "high", "smoking": "high", "cholesterol": "high", "alcohol": "medium", "bp": "high"},
        "Hypertension": {"bmi": "high", "exercise": "medium", "diet": "medium", "smoking": "high", "cholesterol": "high", "alcohol": "high", "bp": "high"},
        "Stroke": {"bmi": "medium", "exercise": "medium", "diet": "medium", "smoking": "high", "cholesterol": "high", "alcohol": "medium", "bp": "high"}
    }
    relevance = disease_relevance.get(selected_disease, {})

    with st.expander("Configure Policy Interventions", expanded=True):
        st.caption("Adjust intervention coverage levels. ⭐ indicates high relevance for selected disease.")
        
        # Quick Presets
        st.markdown("**Quick Presets:**")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        preset_applied = None
        with preset_col1:
            if st.button("Conservative (10-20%)", use_container_width=True, key="preset_conservative"):
                preset_applied = "conservative"
        with preset_col2:
            if st.button("Moderate (20-35%)", use_container_width=True, key="preset_moderate"):
                preset_applied = "moderate"
        with preset_col3:
            if st.button("Aggressive (35-50%)", use_container_width=True, key="preset_aggressive"):
                preset_applied = "aggressive"
        with preset_col4:
            if st.button("Reset All", use_container_width=True, key="preset_reset"):
                preset_applied = "reset"

        st.divider()
        st.markdown("**Custom Adjustments:**")
        
        # Apply presets by setting session state
        preset_values = {
            "conservative": {"bmi": 10, "bmi_cov": 15, "smoking": 15, "exercise": 15, "cholesterol": 15, "diet": 15, "alcohol": 10, "bp": 20},
            "moderate": {"bmi": 8, "bmi_cov": 25, "smoking": 25, "exercise": 30, "cholesterol": 25, "diet": 25, "alcohol": 20, "bp": 30},
            "aggressive": {"bmi": 12, "bmi_cov": 40, "smoking": 40, "exercise": 45, "cholesterol": 40, "diet": 40, "alcohol": 35, "bp": 45},
            "reset": {"bmi": 0, "bmi_cov": 0, "smoking": 0, "exercise": 0, "cholesterol": 0, "diet": 0, "alcohol": 0, "bp": 0}
        }
        
        if preset_applied and preset_applied in preset_values:
            preset = preset_values[preset_applied]
            bmi_reduction = preset["bmi"]
            bmi_coverage = preset["bmi_cov"]
            smoking_cessation = preset["smoking"]
            exercise_increase = preset["exercise"]
            cholesterol_control = preset["cholesterol"]
            diet_improvement = preset["diet"]
            alcohol_reduction = preset["alcohol"]
            bp_control = preset["bp"]
        else:
            # 3-column layout for sliders
            slider_col1, slider_col2, slider_col3 = st.columns(3)
            
            with slider_col1:
                bmi_reduction = st.slider(
                    f"BMI Reduction {'⭐' if relevance.get('bmi') == 'high' else ''}",
                    0, 15, 0, key="sim_bmi",
                    help="% reduction in BMI for treated individuals (high BMI only)"
                )
                smoking_cessation = st.slider(
                    f"Smoking Cessation {'⭐' if relevance.get('smoking') == 'high' else ''}",
                    0, 50, 0, key="sim_smoking",
                    help="% of smokers who quit"
                )
                cholesterol_control = st.slider(
                    f"Cholesterol Control {'⭐' if relevance.get('cholesterol') == 'high' else ''}",
                    0, 50, 0, key="sim_cholesterol",
                    help="% of high cholesterol cases controlled"
                )
            
            with slider_col2:
                bmi_coverage = st.slider(
                    "BMI Coverage (BMI ≥ 25)",
                    0, 100, 0, key="sim_bmi_coverage",
                    help="% of people with high BMI (≥25) who receive the reduction"
                )
                exercise_increase = st.slider(
                    f"Exercise Increase {'⭐' if relevance.get('exercise') == 'high' else ''}",
                    0, 50, 0, key="sim_exercise",
                    help="% of inactive people who start exercising"
                )
                diet_improvement = st.slider(
                    f"Diet Improvement {'⭐' if relevance.get('diet') == 'high' else ''}",
                    0, 50, 0, key="sim_diet",
                    help="% of poor diet individuals who improve"
                )
            
            with slider_col3:
                bp_control = st.slider(
                    f"BP Control {'⭐' if relevance.get('bp') == 'high' else ''}",
                    0, 50, 0, key="sim_bp",
                    help="% of high BP cases brought under control"
                )
                alcohol_reduction = st.slider(
                    f"Alcohol Reduction {'⭐' if relevance.get('alcohol') == 'high' else ''}",
                    0, 50, 0, key="sim_alcohol",
                    help="% of heavy drinkers who reduce consumption"
                )

    interventions = {
        "bmi": bmi_reduction,
        "bmi_coverage": bmi_coverage,
        "smoking": smoking_cessation,
        "exercise": exercise_increase,
        "cholesterol": cholesterol_control,
        "diet": diet_improvement,
        "alcohol": alcohol_reduction,
        "bp": bp_control
    }

    any_intervention = (
        (bmi_reduction > 0 and bmi_coverage > 0)
        or smoking_cessation > 0 or exercise_increase > 0 or cholesterol_control > 0
        or diet_improvement > 0 or alcohol_reduction > 0 or bp_control > 0
    )

    # SECTION 5: Results (tabbed interface)
    if any_intervention:
        results = simulate_combined_intervention(
            filtered_df, selected_disease, target_col, secondary_count, interventions
        )

        st.divider()
        st.markdown("### Simulation Results")

        # Calculate metrics
        primary = results["primary"]
        primary_pct = (primary["reduction"] / primary["baseline"] * 100) if primary["baseline"] > 0 else 0
        
        secondary = results["secondary"]
        secondary_pct = (secondary["reduction"] / secondary["baseline"] * 100) if secondary["baseline"] > 0 else 0
        
        total_before = primary["baseline"] + secondary["baseline"]
        total_after = primary["after"] + secondary["after"]
        total_reduction = primary["reduction"] + secondary["reduction"]
        total_pct = (total_reduction / total_before * 100) if total_before > 0 else 0

        # Tabbed results
        result_tabs = st.tabs(["Combined Impact", "Primary Prevention", "Secondary Prevention", "Impact Breakdown"])
        
        with result_tabs[0]:
            st.markdown("##### Combined Impact Across Both Prevention Types")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("Total Targetable Before", f"{total_before:,}")
            with t_col2:
                st.metric(
                    "Total Targetable After",
                    f"{total_after:,}",
                    delta=f"-{total_reduction:,}",
                    delta_color="normal"
                )
            with t_col3:
                st.metric("Overall Reduction", f"{total_pct:.1f}%")
            
            # Add comparison chart
            if total_before > 0:
                fig = go.Figure(data=[
                    go.Bar(name='Before', x=['High-Risk Population'], y=[total_before], marker_color='#EF553B'),
                    go.Bar(name='After', x=['High-Risk Population'], y=[total_after], marker_color='#00CC96')
                ])
                fig.update_layout(
                    title="Before vs After Intervention",
                    yaxis_title="Number of People",
                    barmode='group',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with result_tabs[1]:
            st.markdown("##### Primary Prevention (Prevent Disease Onset)")
            st.caption("Target: At-risk individuals who do NOT currently have the disease")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.metric("At-Risk Before", f"{primary['baseline']:,}")
            with p_col2:
                st.metric(
                    "At-Risk After",
                    f"{primary['after']:,}",
                    delta=f"-{primary['reduction']:,}",
                    delta_color="normal"
                )
            with p_col3:
                st.metric("Reduction", f"{primary_pct:.1f}%")
            
            if primary["baseline"] > 0:
                fig = px.pie(
                    values=[primary['after'], primary['reduction']],
                    names=['Still At-Risk', 'Moved Out of Risk'],
                    title="Primary Prevention Impact",
                    color_discrete_sequence=['#EF553B', '#00CC96']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with result_tabs[2]:
            st.markdown("##### Secondary Prevention (Prevent Complications)")
            st.caption("Target: High-risk individuals who already HAVE the disease")
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("High-Risk Before", f"{secondary['baseline']:,}")
            with s_col2:
                st.metric(
                    "High-Risk After",
                    f"{secondary['after']:,}",
                    delta=f"-{secondary['reduction']:,}",
                    delta_color="normal"
                )
            with s_col3:
                st.metric("Reduction", f"{secondary_pct:.1f}%")
            
            if secondary["baseline"] > 0:
                fig = px.pie(
                    values=[secondary['after'], secondary['reduction']],
                    names=['Still High-Risk', 'Moved Out of Risk'],
                    title="Secondary Prevention Impact",
                    color_discrete_sequence=['#EF553B', '#00CC96']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with result_tabs[3]:
            if results['individual_impacts']:
                st.markdown("##### Individual Intervention Impact (Secondary Prevention)")
                st.caption("Each bar shows the impact if that intervention was applied alone")
                
                impact_data = [
                    {"Intervention": name, "People Removed from High-Risk": impact}
                    for name, impact in results['individual_impacts'].items()
                ]
                impact_df = pd.DataFrame(impact_data).sort_values("People Removed from High-Risk", ascending=True)
                
                fig_impact = px.bar(
                    impact_df,
                    x="People Removed from High-Risk",
                    y="Intervention",
                    orientation='h',
                    title="Individual Intervention Effectiveness",
                    color="People Removed from High-Risk",
                    color_continuous_scale="Greens"
                )
                fig_impact.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
                st.plotly_chart(fig_impact, use_container_width=True)

                best_intervention = max(results['individual_impacts'].items(), key=lambda x: x[1])
                st.success(
                    f"**Most Effective Single Intervention:** {best_intervention[0]} "
                    f"(removes {best_intervention[1]:,} from secondary high-risk)\n\n"
                    f"**Combined Effect:** All interventions together remove {total_reduction:,} people "
                    f"from high-risk targeting ({total_pct:.1f}% reduction)"
                )
            else:
                st.info("No individual impact data available for the current intervention settings.")
    else:
        st.info("Adjust the intervention sliders above to simulate policy impacts and see the results.")
