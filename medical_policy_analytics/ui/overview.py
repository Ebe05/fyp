"""Overview tab: cross-disease EDA and risk factor analysis."""

import pandas as pd
import plotly.express as px
import streamlit as st

from medical_policy_analytics.config import DISEASE_CONFIG, RISK_FACTORS


def render_overview_tab(df):
    """Render the Overview tab with cross-disease EDA"""
    st.subheader("📊 Cross-Disease Population Overview")
    st.markdown("A comprehensive analysis of all diseases and their common risk factors.")

    # Ensure all required columns are numeric
    numeric_cols = ['Diabetes_012', 'HeartDiseaseorAttack', 'HighBP', 'Stroke',
                    'HighChol', 'BMI', 'Smoker', 'PhysActivity', 'HvyAlcoholConsump', 'Age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create binary indicators for diseases (for correlation)
    df['Has_Diabetes'] = (df['Diabetes_012'] > 0).astype(int)
    df['Has_HeartDisease'] = df['HeartDiseaseorAttack'].astype(int)
    df['Has_Hypertension'] = df['HighBP'].astype(int)
    df['Has_Stroke'] = df['Stroke'].astype(int)
    df['Is_Obese'] = (df['BMI'] >= 30).astype(int)

    # --- SECTION 1: CORRELATION MATRICES ---
    st.markdown("### 🔗 Correlation Analysis")
    st.markdown("Understanding relationships between diseases, health conditions, and lifestyle choices.")

    # Define column groups
    disease_cols = ['Has_Diabetes', 'Has_HeartDisease', 'Has_Hypertension', 'Has_Stroke']
    health_cols = ['HighChol', 'Is_Obese']
    lifestyle_cols = ['Smoker', 'PhysActivity', 'HvyAlcoholConsump']

    # Display names
    disease_names = {'Has_Diabetes': 'Diabetes', 'Has_HeartDisease': 'Heart Disease',
                     'Has_Hypertension': 'Hypertension', 'Has_Stroke': 'Stroke'}
    health_names = {'HighChol': 'High Cholesterol', 'Is_Obese': 'Obesity'}
    lifestyle_names = {'Smoker': 'Smoker', 'PhysActivity': 'Physical Activity',
                       'HvyAlcoholConsump': 'Heavy Alcohol'}

    # Create 3 columns for side-by-side heatmaps
    hm_col1, hm_col2, hm_col3 = st.columns(3)

    with hm_col1:
        st.markdown("#### Diseases & Health Factors")
        corr1 = df[disease_cols + health_cols].corr(method='spearman').loc[disease_cols, health_cols]
        corr1 = corr1.rename(index=disease_names, columns=health_names)
        fig1 = px.imshow(corr1, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         title="Diseases × Health")
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with hm_col2:
        st.markdown("#### Diseases & Lifestyle")
        corr2 = df[disease_cols + lifestyle_cols].corr(method='spearman').loc[disease_cols, lifestyle_cols]
        corr2 = corr2.rename(index=disease_names, columns=lifestyle_names)
        fig2 = px.imshow(corr2, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         title="Diseases × Lifestyle")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with hm_col3:
        st.markdown("#### Health & Lifestyle")
        corr3 = df[health_cols + lifestyle_cols].corr(method='spearman').loc[health_cols, lifestyle_cols]
        corr3 = corr3.rename(index=health_names, columns=lifestyle_names)
        fig3 = px.imshow(corr3, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         title="Health × Lifestyle")
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    st.caption("📊 Spearman rank correlation: Red = positive correlation (factors increase together), Blue = negative correlation (one increases as other decreases)")
    st.markdown("---")

    # --- SECTION 2: DISEASE PREVALENCE COMPARISON ---
    st.markdown("### 📈 Disease Prevalence Comparison")
    st.markdown("Which disease affects the largest portion of the population?")

    prevalence_data = []
    for disease_name, config in DISEASE_CONFIG.items():
        col = config["column"]
        count = (df[col] > 0).sum()
        pct = (df[col] > 0).mean() * 100
        prevalence_data.append({
            "Disease": disease_name,
            "Cases": count,
            "Prevalence (%)": pct
        })

    prevalence_df = pd.DataFrame(prevalence_data)
    prevalence_df = prevalence_df.sort_values("Prevalence (%)", ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_prev = px.bar(
            prevalence_df,
            x="Prevalence (%)",
            y="Disease",
            orientation='h',
            title="Disease Prevalence in Population (%)",
            color="Prevalence (%)",
            color_continuous_scale="Reds",
            text="Prevalence (%)"
        )
        fig_prev.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prev.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_prev, use_container_width=True)

    with col2:
        fig_cases = px.bar(
            prevalence_df,
            x="Cases",
            y="Disease",
            orientation='h',
            title="Total Disease Cases in Population",
            color="Cases",
            color_continuous_scale="Blues",
            text="Cases"
        )
        fig_cases.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_cases.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_cases, use_container_width=True)

    most_prevalent = prevalence_df.iloc[-1]
    st.info(f"**Key Finding:** {most_prevalent['Disease']} is the most prevalent condition, "
            f"affecting {most_prevalent['Prevalence (%)']:.1f}% of the population ({most_prevalent['Cases']:,.0f} cases).")
    st.markdown("---")

    # --- SECTION 3: HOW LIFESTYLE CHOICES IMPACT HEALTH CONDITIONS ---
    st.markdown("### 🏃 How Lifestyle Choices Impact Health Conditions")
    st.markdown("Direct cause-effect relationships: Understanding how daily choices impact your health metrics.")

    health_conditions = {
        "High Blood Pressure": "HighBP",
        "High Cholesterol": "HighChol",
        "Obesity (BMI≥30)": "Is_Obese"
    }

    lifestyle_health_data = []
    for lifestyle_name, lifestyle_config in RISK_FACTORS["Lifestyle Factors"].items():
        lifestyle_col = lifestyle_config["column"]
        labels = lifestyle_config["labels"]
        for health_name, health_col in health_conditions.items():
            no_lifestyle = df[df[lifestyle_col] == 0]
            if health_col == "Is_Obese":
                no_lifestyle_prev = no_lifestyle[health_col].mean() * 100
            else:
                no_lifestyle_prev = (no_lifestyle[health_col] == 1).mean() * 100
            has_lifestyle = df[df[lifestyle_col] == 1]
            if health_col == "Is_Obese":
                has_lifestyle_prev = has_lifestyle[health_col].mean() * 100
            else:
                has_lifestyle_prev = (has_lifestyle[health_col] == 1).mean() * 100
            lifestyle_health_data.append({
                "Lifestyle Factor": lifestyle_name,
                "Health Condition": health_name,
                "Status": labels[0.0],
                "Prevalence (%)": no_lifestyle_prev
            })
            lifestyle_health_data.append({
                "Lifestyle Factor": lifestyle_name,
                "Health Condition": health_name,
                "Status": labels[1.0],
                "Prevalence (%)": has_lifestyle_prev
            })

    lifestyle_health_df = pd.DataFrame(lifestyle_health_data)
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    for idx, (lifestyle_name, lifestyle_config) in enumerate(RISK_FACTORS["Lifestyle Factors"].items()):
        with columns[idx]:
            factor_data = lifestyle_health_df[lifestyle_health_df["Lifestyle Factor"] == lifestyle_name]
            fig = px.bar(
                factor_data,
                x="Health Condition",
                y="Prevalence (%)",
                color="Status",
                barmode="group",
                title=f"Health Impact of {lifestyle_name}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            pivot = factor_data.pivot(index="Health Condition", columns="Status", values="Prevalence (%)")
            differences = pivot.iloc[:, 1] - pivot.iloc[:, 0]
            max_diff_condition = differences.abs().idxmax()
            max_diff_value = differences[max_diff_condition]
            if max_diff_value > 0:
                st.caption(f"📊 Biggest impact: {max_diff_value:.1f}% higher {max_diff_condition}")
            else:
                st.caption(f"📊 Biggest impact: {abs(max_diff_value):.1f}% lower {max_diff_condition}")

    st.info("**Takeaway:** These charts show direct lifestyle-health relationships. "
            "The larger the gap between groups, the stronger the impact of that lifestyle choice.")
    st.markdown("---")

    # --- SECTION 4: UNIFIED RISK FACTOR ANALYSIS ---
    st.markdown("### 🎯 Identifying the 'Common Enemy' (Global Risk Factors)")
    st.markdown("Which risk factors contribute most to **all** diseases? This helps identify universal intervention targets.")

    risk_tab1, risk_tab2 = st.tabs(["Lifestyle Factors", "Health Factors"])

    with risk_tab1:
        st.markdown("#### Impact of Lifestyle Choices on All Diseases")
        lifestyle_data = []
        for factor_name, factor_config in RISK_FACTORS["Lifestyle Factors"].items():
            col = factor_config["column"]
            labels = factor_config["labels"]
            for disease_name, disease_config in DISEASE_CONFIG.items():
                disease_col = disease_config["column"]
                no_factor_prev = (df[df[col] == 0][disease_col] > 0).mean() * 100
                has_factor_prev = (df[df[col] == 1][disease_col] > 0).mean() * 100
                lifestyle_data.append({"Risk Factor": factor_name, "Status": labels[0.0], "Disease": disease_name, "Prevalence (%)": no_factor_prev})
                lifestyle_data.append({"Risk Factor": factor_name, "Status": labels[1.0], "Disease": disease_name, "Prevalence (%)": has_factor_prev})
        lifestyle_df = pd.DataFrame(lifestyle_data)
        for factor_name in RISK_FACTORS["Lifestyle Factors"].keys():
            factor_data = lifestyle_df[lifestyle_df["Risk Factor"] == factor_name]
            fig_lifestyle = px.bar(
                factor_data,
                x="Disease",
                y="Prevalence (%)",
                color="Status",
                barmode="group",
                title=f"Disease Prevalence by {factor_name} Status",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_lifestyle.update_layout(height=350)
            st.plotly_chart(fig_lifestyle, use_container_width=True)

    with risk_tab2:
        st.markdown("#### Impact of Health Conditions on All Diseases")
        health_data = []
        for factor_name, factor_config in RISK_FACTORS["Health Factors"].items():
            col = factor_config["column"]
            if factor_config.get("is_continuous"):
                threshold = factor_config["threshold"]
                for disease_name, disease_config in DISEASE_CONFIG.items():
                    disease_col = disease_config["column"]
                    below_prev = (df[df[col] < threshold][disease_col] > 0).mean() * 100
                    above_prev = (df[df[col] >= threshold][disease_col] > 0).mean() * 100
                    health_data.append({"Risk Factor": factor_name, "Status": f"BMI < {threshold}", "Disease": disease_name, "Prevalence (%)": below_prev})
                    health_data.append({"Risk Factor": factor_name, "Status": f"BMI ≥ {threshold}", "Disease": disease_name, "Prevalence (%)": above_prev})
            else:
                labels = factor_config["labels"]
                for disease_name, disease_config in DISEASE_CONFIG.items():
                    disease_col = disease_config["column"]
                    no_factor_prev = (df[df[col] == 0][disease_col] > 0).mean() * 100
                    has_factor_prev = (df[df[col] == 1][disease_col] > 0).mean() * 100
                    health_data.append({"Risk Factor": factor_name, "Status": labels[0.0], "Disease": disease_name, "Prevalence (%)": no_factor_prev})
                    health_data.append({"Risk Factor": factor_name, "Status": labels[1.0], "Disease": disease_name, "Prevalence (%)": has_factor_prev})
        health_df = pd.DataFrame(health_data)
        for factor_name in RISK_FACTORS["Health Factors"].keys():
            factor_data = health_df[health_df["Risk Factor"] == factor_name]
            fig_health = px.bar(
                factor_data,
                x="Disease",
                y="Prevalence (%)",
                color="Status",
                barmode="group",
                title=f"Disease Prevalence by {factor_name}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_health.update_layout(height=350)
            st.plotly_chart(fig_health, use_container_width=True)

    st.success("**Policy Insight:** Risk factors that elevate prevalence across ALL diseases represent the highest-impact targets "
               "for public health interventions. Focus resources on factors showing the largest gaps between groups.")
