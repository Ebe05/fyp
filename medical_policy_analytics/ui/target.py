"""Find Your Target tab: risk-based population targeting."""

import pandas as pd
import plotly.express as px
import streamlit as st

from medical_policy_analytics.analytics.risk import calculate_risk_score
from medical_policy_analytics.config import POLICY_DETAILS


def render_target_tab(df):
    """Render the Find Your Target tab for risk-based population targeting"""

    st.markdown("### Find Your Target Population")
    st.markdown("Identify high-risk individuals for targeted policy interventions based on a composite risk score.")

    df = calculate_risk_score(df)

    st.markdown("#### Population Risk Score Distribution")
    col_hist, col_stats = st.columns([3, 1])
    with col_hist:
        fig_dist = px.histogram(
            df, x='risk_score',
            nbins=50,
            title="Risk Score Distribution Across Population",
            labels={'risk_score': 'Risk Score (0-100)', 'count': 'Number of Individuals'},
            color_discrete_sequence=['#636EFA']
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_stats:
        st.metric("Mean Score", f"{df['risk_score'].mean():.1f}")
        st.metric("Median Score", f"{df['risk_score'].median():.1f}")
        st.metric("Std Dev", f"{df['risk_score'].std():.1f}")
        st.metric("Max Score", f"{df['risk_score'].max():.1f}")

    st.divider()
    st.markdown("#### Set Intervention Threshold")
    st.markdown("Select the risk percentile to define your target group. Higher percentiles = smaller, higher-risk groups.")

    threshold_percentile = st.slider(
        "Risk Threshold (Percentile)",
        min_value=50, max_value=99, value=90, step=1,
        help="Select the minimum percentile for intervention. 90 = top 10% highest risk."
    )

    threshold_value = df['risk_score'].quantile(threshold_percentile / 100)
    high_risk_df = df[df['risk_score'] >= threshold_value]

    high_risk_count = len(high_risk_df)
    high_risk_pct = (high_risk_count / len(df)) * 100

    col_gauge, col_info = st.columns([2, 2])
    with col_gauge:
        fig_gauge = px.pie(
            values=[high_risk_count, len(df) - high_risk_count],
            names=['High-Risk (Target)', 'Below Threshold'],
            title=f"Target Group: Top {100 - threshold_percentile}%",
            color_discrete_sequence=['#EF553B', '#E8E8E8']
        )
        fig_gauge.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_info:
        st.markdown("##### Target Group Summary")
        st.metric("Individuals in Target Group", f"{high_risk_count:,}")
        st.metric("Percentage of Population", f"{high_risk_pct:.1f}%")
        st.metric("Minimum Risk Score", f"{threshold_value:.1f}")
        st.metric("Avg Risk Score (Target)", f"{high_risk_df['risk_score'].mean():.1f}")
        st.success(f"**{high_risk_count:,} people ({high_risk_pct:.1f}%)** qualify for intervention based on your threshold.")

    st.divider()
    st.markdown("#### Demographic Profile of Target Group")
    st.markdown("Understand WHO your high-risk individuals are to design targeted outreach.")

    income_labels = {
        1: '<$10k', 2: '$10-15k', 3: '$15-20k', 4: '$20-25k',
        5: '$25-35k', 6: '$35-50k', 7: '$50-75k', 8: '$75k+'
    }
    high_risk_df = high_risk_df.copy()
    high_risk_df['Income_Label'] = high_risk_df['Income'].map(income_labels)

    age_labels = {
        1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
        6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
        11: '70-74', 12: '75-79', 13: '80+'
    }
    high_risk_df['Age_Label'] = high_risk_df['Age'].map(age_labels)

    edu_labels = {
        1: 'Never attended', 2: 'Elementary', 3: 'Some high school',
        4: 'High school grad', 5: 'Some college', 6: 'College grad'
    }
    high_risk_df['Education_Label'] = high_risk_df['Education'].map(edu_labels)
    high_risk_df['Gender'] = high_risk_df['Sex'].map({0: 'Female', 1: 'Male'})

    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        income_counts = high_risk_df['Income_Label'].value_counts().sort_index()
        fig_income = px.bar(
            x=list(income_labels.values()),
            y=[income_counts.get(label, 0) for label in income_labels.values()],
            title="Income Distribution of Target Group",
            labels={'x': 'Income Bracket', 'y': 'Count'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_income, use_container_width=True)
        age_counts = high_risk_df['Age_Label'].value_counts()
        fig_age = px.bar(
            x=list(age_labels.values()),
            y=[age_counts.get(label, 0) for label in age_labels.values()],
            title="Age Distribution of Target Group",
            labels={'x': 'Age Group', 'y': 'Count'},
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig_age, use_container_width=True)
    with demo_col2:
        edu_counts = high_risk_df['Education_Label'].value_counts()
        fig_edu = px.bar(
            x=list(edu_labels.values()),
            y=[edu_counts.get(label, 0) for label in edu_labels.values()],
            title="Education Level of Target Group",
            labels={'x': 'Education', 'y': 'Count'},
            color_discrete_sequence=['#AB63FA']
        )
        fig_edu.update_xaxes(tickangle=45)
        st.plotly_chart(fig_edu, use_container_width=True)
        gender_counts = high_risk_df['Gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Gender Split of Target Group",
            color_discrete_sequence=['#FF6692', '#19D3F3']
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    st.divider()
    st.markdown("#### Key Insights About Your Target Group")

    low_income_pct = (high_risk_df['Income'] <= 4).mean() * 100
    elderly_pct = (high_risk_df['Age'] >= 9).mean() * 100
    low_edu_pct = (high_risk_df['Education'] <= 3).mean() * 100
    male_pct = (high_risk_df['Sex'] == 1).mean() * 100

    smoker_pct = high_risk_df['Smoker'].mean() * 100
    no_exercise_pct = (high_risk_df['PhysActivity'] == 0).mean() * 100
    obese_pct = (high_risk_df['BMI'] >= 30).mean() * 100
    high_bp_pct = high_risk_df['HighBP'].mean() * 100
    high_chol_pct = high_risk_df['HighChol'].mean() * 100

    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        st.markdown("##### Demographics")
        st.info(f"**{low_income_pct:.0f}%** are in low-income brackets (<$25k)")
        st.info(f"**{elderly_pct:.0f}%** are elderly (60+ years)")
        st.info(f"**{low_edu_pct:.0f}%** have no high school diploma")
        st.info(f"**{male_pct:.0f}%** are male")
    with insight_col2:
        st.markdown("##### Primary Intervention Needs")
        interventions = [
            (high_bp_pct, "High Blood Pressure", "BP management programs"),
            (obese_pct, "Obesity", "Weight management initiatives"),
            (high_chol_pct, "High Cholesterol", "Cholesterol screening & treatment"),
            (no_exercise_pct, "Physical Inactivity", "Community fitness programs"),
            (smoker_pct, "Smoking", "Tobacco cessation programs")
        ]
        interventions.sort(reverse=True, key=lambda x: x[0])
        for pct, condition, program in interventions[:3]:
            st.warning(f"**{pct:.0f}%** have {condition} → {program}")

    st.success(
        f"**Recommended Targeting Strategy:** Focus outreach on "
        f"{'low-income' if low_income_pct > 50 else 'middle-income'} "
        f"{'elderly' if elderly_pct > 40 else 'adult'} populations. "
        f"Primary intervention: {interventions[0][2]}."
    )

    st.divider()
    st.markdown("#### Target Group vs. National Baseline")
    st.markdown("Compare disease prevalence in your target group against the general population to quantify the urgency.")

    diseases_comparison = {
        "Diabetes": {
            "national": (df['Diabetes_012'] > 0).mean() * 100,
            "target": (high_risk_df['Diabetes_012'] > 0).mean() * 100
        },
        "Heart Disease": {
            "national": df['HeartDiseaseorAttack'].mean() * 100,
            "target": high_risk_df['HeartDiseaseorAttack'].mean() * 100
        },
        "Hypertension": {
            "national": df['HighBP'].mean() * 100,
            "target": high_risk_df['HighBP'].mean() * 100
        },
        "Stroke": {
            "national": df['Stroke'].mean() * 100,
            "target": high_risk_df['Stroke'].mean() * 100
        }
    }

    comparison_data = []
    for disease, values in diseases_comparison.items():
        comparison_data.append({"Disease": disease, "Group": "National Average", "Prevalence (%)": values["national"]})
        comparison_data.append({"Disease": disease, "Group": "Target Group", "Prevalence (%)": values["target"]})
    comparison_df = pd.DataFrame(comparison_data)

    fig_comparison = px.bar(
        comparison_df,
        x="Disease",
        y="Prevalence (%)",
        color="Group",
        barmode="group",
        title="Disease Prevalence: Target Group vs. National Average",
        color_discrete_map={"National Average": "#636EFA", "Target Group": "#EF553B"}
    )
    fig_comparison.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("##### Risk Multipliers (Target vs. National)")
    mult_cols = st.columns(4)
    for i, (disease, values) in enumerate(diseases_comparison.items()):
        multiplier = values["target"] / values["national"] if values["national"] > 0 else 0
        with mult_cols[i]:
            delta_pct = values["target"] - values["national"]
            st.metric(
                disease,
                f"{multiplier:.1f}x",
                delta=f"+{delta_pct:.1f}%",
                delta_color="inverse"
            )

    st.warning(
        f"**Urgency Indicator:** Your target group has significantly elevated disease rates. "
        f"On average, they are **{sum(v['target']/v['national'] for v in diseases_comparison.values() if v['national'] > 0)/4:.1f}x more likely** "
        f"to have these conditions compared to the general population."
    )

    st.divider()
    st.markdown("#### Strategic Policy Recommendations")
    st.markdown("Based on your target group's profile, here are the prioritized policy interventions:")

    lever_prevalence = {
        "Smoker": high_risk_df['Smoker'].mean() * 100,
        "No_Exercise": (high_risk_df['PhysActivity'] == 0).mean() * 100,
        "Heavy_Alcohol": high_risk_df['HvyAlcoholConsump'].mean() * 100,
        "Poor_Diet": ((high_risk_df['Fruits'] == 0) & (high_risk_df['Veggies'] == 0)).mean() * 100,
        "Obese": (high_risk_df['BMI'] >= 30).mean() * 100,
        "High_Cholesterol": high_risk_df['HighChol'].mean() * 100
    }
    sorted_levers = sorted(lever_prevalence.items(), key=lambda x: -x[1])

    priority_badges = ["🥇", "🥈", "🥉"]
    for rank, (lever, prevalence) in enumerate(sorted_levers[:3]):
        detail = POLICY_DETAILS[lever]
        affected_count = int(high_risk_count * prevalence / 100)
        with st.container():
            col_badge, col_content = st.columns([1, 9])
            with col_badge:
                st.markdown(f"## {priority_badges[rank]}")
                st.caption(f"Priority {rank + 1}")
            with col_content:
                st.markdown(f"**{detail['title']}**")
                st.caption(f"Affects **{prevalence:.0f}%** of target group ({affected_count:,} individuals)")
                with st.expander("View Policy Details"):
                    st.info(f"**Recommended Action:** {detail['action']}")
                    st.success(f"**Expected Impact:** {detail['impact']}")
            st.divider()

    top_3_titles = [POLICY_DETAILS[lever]['title'] for lever, _ in sorted_levers[:3]]
    st.markdown("#### Prioritized Action Plan")
    st.markdown(
        f"For your selected target group of **{high_risk_count:,} individuals**, "
        f"implement the following policies in order of priority:\n\n"
        f"1. **{top_3_titles[0]}** (highest impact)\n"
        f"2. **{top_3_titles[1]}**\n"
        f"3. **{top_3_titles[2]}**"
    )
