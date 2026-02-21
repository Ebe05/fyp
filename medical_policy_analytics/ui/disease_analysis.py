"""Disease Analysis tab: disease-specific analysis and what-if simulation."""

import pandas as pd
import plotly.express as px
import streamlit as st

from medical_policy_analytics.analytics.risk import calculate_high_risk, simulate_combined_intervention
from medical_policy_analytics.config import DISEASE_CONFIG


def render_disease_analysis_tab(df):
    """Render the Disease Analysis tab with disease-specific features"""

    df['HighBP'] = pd.to_numeric(df['HighBP'], errors='coerce')
    df['Diabetes_012'] = pd.to_numeric(df['Diabetes_012'], errors='coerce')
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['HighChol'] = pd.to_numeric(df['HighChol'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    selected_disease = st.selectbox(
        "Select Disease to Analyze",
        options=list(DISEASE_CONFIG.keys()),
        help="Choose which health condition to analyze"
    )

    disease_info = DISEASE_CONFIG[selected_disease]
    target_col = disease_info["column"]

    st.subheader(f"Analyzing: {selected_disease}")

    st.markdown("### Policy Intervention Filters")
    income_filter = st.multiselect(
        "Select Target Income Brackets",
        options=sorted(df['Income'].unique()),
        default=sorted(df['Income'].unique()),
        key="income_filter"
    )

    mask = df['Income'].isin(income_filter)
    filtered_df = df[mask].copy()

    filtered_df['Disease_Status'] = filtered_df[target_col].map(disease_info["labels"])

    high_risk_condition = calculate_high_risk(filtered_df, selected_disease, target_col)
    high_risk_count = high_risk_condition.sum()
    total_count = len(filtered_df)
    risk_percentage = (high_risk_count / total_count) * 100 if total_count > 0 else 0
    prevalence = (filtered_df[target_col] > 0).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Population", f"{total_count:,}")
    with col2:
        st.metric(f"{selected_disease} Cases", f"{(filtered_df[target_col] > 0).sum():,}")
    with col3:
        st.metric("Prevalence Rate", f"{prevalence:.1f}%")
    with col4:
        st.metric("High-Risk Individuals", f"{high_risk_count:,}", delta_color="inverse")

    st.markdown("---")

    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.subheader(f"{selected_disease} by Income Bracket")
        fig_income = px.histogram(
            filtered_df, x="Income", color="Disease_Status",
            barmode="group",
            title=f"Income Distribution vs {selected_disease}",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_income, use_container_width=True)

    with right_chart:
        st.subheader("BMI Correlation")
        fig_bmi = px.box(
            filtered_df, x="Disease_Status", y="BMI",
            color="Disease_Status",
            title=f"BMI Distribution: {selected_disease}"
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    st.info("### 🤖 Automated Policy Recommendation")
    threshold = disease_info["risk_threshold"]
    if risk_percentage > threshold:
        st.error(f"**CRITICAL:** {selected_disease} high-risk population is at {risk_percentage:.1f}%. "
                 f"Recommend immediate intervention programs targeting these income brackets.")
    elif risk_percentage > threshold * 0.6:
        st.warning(f"**MODERATE:** Elevated {selected_disease} risk detected. Increase screening programs.")
    else:
        st.success(f"**STABLE:** {selected_disease} metrics within acceptable ranges.")

    st.markdown("### 🧪 'What-If' Policy Simulation")
    st.markdown("Simulate the combined impact of multiple policy interventions on the high-risk population.")

    disease_relevance = {
        "Diabetes": {"bmi": "high", "exercise": "high", "diet": "high", "smoking": "medium", "cholesterol": "medium", "alcohol": "low", "bp": "medium"},
        "Heart Disease": {"bmi": "high", "exercise": "high", "diet": "high", "smoking": "high", "cholesterol": "high", "alcohol": "medium", "bp": "high"},
        "Hypertension": {"bmi": "high", "exercise": "medium", "diet": "medium", "smoking": "high", "cholesterol": "high", "alcohol": "high", "bp": "high"},
        "Stroke": {"bmi": "medium", "exercise": "medium", "diet": "medium", "smoking": "high", "cholesterol": "high", "alcohol": "medium", "bp": "high"}
    }
    relevance = disease_relevance.get(selected_disease, {})

    st.markdown("#### Adjust Intervention Levels")
    slider_col1, slider_col2 = st.columns(2)

    with slider_col1:
        bmi_reduction = st.slider(
            f"🏋️ BMI Reduction ({'⭐' if relevance.get('bmi') == 'high' else ''})",
            0, 15, 0, key="sim_bmi",
            help="% reduction in BMI for treated individuals (high BMI only)"
        )
        bmi_coverage = st.slider(
            "BMI treatment coverage (BMI ≥ 25)",
            0, 100, 0, key="sim_bmi_coverage",
            help="% of people with high BMI (≥25) who receive the reduction"
        )
        smoking_cessation = st.slider(
            f"🚭 Smoking Cessation ({'⭐' if relevance.get('smoking') == 'high' else ''})",
            0, 50, 0, key="sim_smoking",
            help="% of smokers who quit"
        )
        exercise_increase = st.slider(
            f"🏃 Exercise Increase ({'⭐' if relevance.get('exercise') == 'high' else ''})",
            0, 50, 0, key="sim_exercise",
            help="% of inactive people who start exercising"
        )

    with slider_col2:
        cholesterol_control = st.slider(
            f"💊 Cholesterol Control ({'⭐' if relevance.get('cholesterol') == 'high' else ''})",
            0, 50, 0, key="sim_cholesterol",
            help="% of high cholesterol cases controlled"
        )
        diet_improvement = st.slider(
            f"🥗 Diet Improvement ({'⭐' if relevance.get('diet') == 'high' else ''})",
            0, 50, 0, key="sim_diet",
            help="% of poor diet individuals who improve"
        )
        alcohol_reduction = st.slider(
            f"🍺 Alcohol Reduction ({'⭐' if relevance.get('alcohol') == 'high' else ''})",
            0, 50, 0, key="sim_alcohol",
            help="% of heavy drinkers who reduce consumption"
        )

    bp_control = st.slider(
        f"❤️ Blood Pressure Control ({'⭐' if relevance.get('bp') == 'high' else ''})",
        0, 50, 0, key="sim_bp",
        help="% of high BP cases brought under control"
    )

    st.caption("⭐ = High relevance for selected disease")

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

    if any_intervention:
        results = simulate_combined_intervention(
            filtered_df, selected_disease, target_col, high_risk_count, interventions
        )

        st.markdown("---")
        st.markdown("#### 📊 Simulation Results")

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric("High-Risk Before", f"{high_risk_count:,}")
        with result_col2:
            st.metric(
                "High-Risk After",
                f"{results['new_high_risk']:,}",
                delta=f"-{results['combined_reduction']:,}",
                delta_color="normal"
            )
        with result_col3:
            reduction_pct = (results['combined_reduction'] / high_risk_count * 100) if high_risk_count > 0 else 0
            st.metric("Reduction %", f"{reduction_pct:.1f}%")

        if results['individual_impacts']:
            st.markdown("#### 📈 Impact Breakdown by Intervention")
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
                title="Individual Intervention Impact (if applied alone)",
                color="People Removed from High-Risk",
                color_continuous_scale="Greens"
            )
            fig_impact.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_impact, use_container_width=True)

            best_intervention = max(results['individual_impacts'].items(), key=lambda x: x[1])
            st.success(
                f"**Most Effective Single Intervention:** {best_intervention[0]} "
                f"(removes {best_intervention[1]:,} from high-risk)\n\n"
                f"**Combined Effect:** All interventions together remove {results['combined_reduction']:,} people "
                f"({reduction_pct:.1f}% of high-risk population)"
            )
    else:
        st.info("👆 Adjust the sliders above to simulate policy interventions and see their impact.")
