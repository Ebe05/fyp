"""Budget Priority tab: department pressure analysis and what-if simulation."""

import pandas as pd
import plotly.express as px
import streamlit as st

from medical_policy_analytics.analytics.hospital_ops import calculate_department_pressure


def render_budget_priority_tab(df):
    """Render the Budget Priority tab with department pressure analysis."""
    st.markdown("### 💰 Budget Priority Analysis")
    st.markdown("Identify departments facing the most strain to prioritize resource allocation.")

    dept_pressure = calculate_department_pressure(df)

    if len(dept_pressure) == 0:
        st.warning("No department data available for analysis.")
        return

    st.markdown("#### 🚨 Top 3 High-Pressure Departments")
    top_3 = dept_pressure.head(3)
    kpi_cols = st.columns(3)
    priority_badges = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top_3.iterrows()):
        with kpi_cols[i]:
            st.metric(
                label=f"{priority_badges[i]} {row['Department'][:20]}...",
                value=f"{row['Pressure_Index']:.2f}",
                delta=f"Vol: {row['Volume']:,}",
                delta_color="off"
            )
            st.caption(f"Bed-days: {row['Intensity']:,.0f} | Avg Labs: {row['Complexity']:.1f}")

    st.markdown("---")
    st.markdown("#### 📊 Pressure Index by Department")
    top_n = dept_pressure.head(15)

    fig_pressure = px.bar(
        top_n,
        x='Pressure_Index',
        y='Department',
        orientation='h',
        title="Department Pressure Index (Top 15)",
        labels={'Pressure_Index': 'Pressure Index (0-1)', 'Department': 'Medical Specialty'},
        color='Pressure_Index',
        color_continuous_scale='Reds'
    )
    fig_pressure.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
        height=500
    )
    st.plotly_chart(fig_pressure, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Pressure Index Component Breakdown")
    st.markdown("See how Volume, Intensity, and Complexity contribute to each department's score.")

    breakdown_data = []
    for _, row in top_n.iterrows():
        breakdown_data.append({'Department': row['Department'], 'Component': 'Volume', 'Score': row['Volume_norm'] / 3})
        breakdown_data.append({'Department': row['Department'], 'Component': 'Intensity', 'Score': row['Intensity_norm'] / 3})
        breakdown_data.append({'Department': row['Department'], 'Component': 'Complexity', 'Score': row['Complexity_norm'] / 3})
    breakdown_df = pd.DataFrame(breakdown_data)

    fig_breakdown = px.bar(
        breakdown_df,
        x='Score',
        y='Department',
        color='Component',
        orientation='h',
        title="Contribution of Each Component to Pressure Index",
        labels={'Score': 'Contribution to Index', 'Department': 'Medical Specialty'},
        color_discrete_map={'Volume': '#636EFA', 'Intensity': '#EF553B', 'Complexity': '#00CC96'}
    )
    fig_breakdown.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        barmode='stack',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚠️ Critical Priority Departments")

    threshold_75 = dept_pressure['Pressure_Index'].quantile(0.75)
    critical_depts = dept_pressure[dept_pressure['Pressure_Index'] >= threshold_75]

    st.warning(f"**{len(critical_depts)} departments** are above the 75th percentile (Pressure Index ≥ {threshold_75:.2f})")

    critical_display = critical_depts[['Department', 'Volume', 'Intensity', 'Complexity', 'Pressure_Index']].copy()
    critical_display.columns = ['Department', 'Admissions', 'Total Bed-Days', 'Avg Lab Procedures', 'Pressure Index']
    critical_display['Pressure Index'] = critical_display['Pressure Index'].apply(lambda x: f"{x:.3f}")
    critical_display['Admissions'] = critical_display['Admissions'].apply(lambda x: f"{x:,}")
    critical_display['Total Bed-Days'] = critical_display['Total Bed-Days'].apply(lambda x: f"{x:,.0f}")
    critical_display['Avg Lab Procedures'] = critical_display['Avg Lab Procedures'].apply(lambda x: f"{x:.1f}")

    st.dataframe(critical_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📋 Complete Department Rankings")

    with st.expander("View All Departments"):
        full_display = dept_pressure[['Department', 'Volume', 'Intensity', 'Complexity', 'Pressure_Index']].copy()
        full_display.columns = ['Department', 'Admissions', 'Total Bed-Days', 'Avg Lab Procedures', 'Pressure Index']
        full_display['Rank'] = range(1, len(full_display) + 1)
        full_display = full_display[['Rank', 'Department', 'Admissions', 'Total Bed-Days', 'Avg Lab Procedures', 'Pressure Index']]
        full_display['Pressure Index'] = full_display['Pressure Index'].apply(lambda x: f"{x:.3f}")
        full_display['Admissions'] = full_display['Admissions'].apply(lambda x: f"{x:,}")
        full_display['Total Bed-Days'] = full_display['Total Bed-Days'].apply(lambda x: f"{x:,.0f}")
        full_display['Avg Lab Procedures'] = full_display['Avg Lab Procedures'].apply(lambda x: f"{x:.1f}")
        st.dataframe(full_display, use_container_width=True, hide_index=True)

    st.markdown("#### 🎯 Budget Allocation Recommendation")
    top_dept = dept_pressure.iloc[0]
    st.success(
        f"**Priority Focus:** {top_dept['Department']}\n\n"
        f"This department has the highest Pressure Index ({top_dept['Pressure_Index']:.3f}) with:\n"
        f"- **{top_dept['Volume']:,}** admissions (Volume)\n"
        f"- **{top_dept['Intensity']:,.0f}** total bed-days (Intensity)\n"
        f"- **{top_dept['Complexity']:.1f}** avg lab procedures (Complexity)\n\n"
        f"Consider allocating additional resources including staffing, equipment, and bed capacity."
    )

    st.markdown("---")
    st.markdown("### 🧪 What-If Resource Simulation")
    st.markdown("Simulate how targeted resource investments could reduce department pressure.")

    top_depts = dept_pressure.head(10)['Department'].tolist()
    selected_dept = st.selectbox(
        "Select Department to Simulate",
        top_depts,
        key="whatif_dept_select"
    )

    dept_row = dept_pressure[dept_pressure['Department'] == selected_dept].iloc[0]
    original_rank = dept_pressure.reset_index(drop=True).index[dept_pressure['Department'] == selected_dept].tolist()[0] + 1

    col_current1, col_current2, col_current3 = st.columns(3)
    with col_current1:
        st.metric("Current Pressure Index", f"{dept_row['Pressure_Index']:.3f}")
    with col_current2:
        st.metric("Current Rank", f"#{original_rank} of {len(dept_pressure)}")
    with col_current3:
        st.metric("Total Admissions", f"{dept_row['Volume']:,}")

    st.markdown("#### Adjust Investment Levels")
    st.caption("Each slider represents the % reduction in that pressure component from resource investment.")

    slider_col1, slider_col2, slider_col3 = st.columns(3)
    with slider_col1:
        staffing = st.slider(
            "👥 Staffing Investment", 0, 50, 0, 5,
            help="% reduction in volume pressure from additional staff capacity",
            key="whatif_staffing"
        )
    with slider_col2:
        beds = st.slider(
            "🛏️ Bed Capacity Investment", 0, 50, 0, 5,
            help="% reduction in intensity pressure from added bed capacity",
            key="whatif_beds"
        )
    with slider_col3:
        equipment = st.slider(
            "🔬 Equipment Investment", 0, 50, 0, 5,
            help="% reduction in complexity pressure from better technology",
            key="whatif_equipment"
        )

    any_investment = staffing > 0 or beds > 0 or equipment > 0

    if any_investment:
        new_volume_norm = dept_row['Volume_norm'] * (1 - staffing/100)
        new_intensity_norm = dept_row['Intensity_norm'] * (1 - beds/100)
        new_complexity_norm = dept_row['Complexity_norm'] * (1 - equipment/100)
        new_pressure = (new_volume_norm + new_intensity_norm + new_complexity_norm) / 3

        pressure_reduction = dept_row['Pressure_Index'] - new_pressure
        reduction_pct = (pressure_reduction / dept_row['Pressure_Index']) * 100

        simulated_pressures = dept_pressure.copy()
        simulated_pressures.loc[simulated_pressures['Department'] == selected_dept, 'Pressure_Index'] = new_pressure
        simulated_pressures = simulated_pressures.sort_values('Pressure_Index', ascending=False).reset_index(drop=True)
        new_rank = simulated_pressures.reset_index(drop=True).index[simulated_pressures['Department'] == selected_dept].tolist()[0] + 1
        rank_change = new_rank - original_rank

        st.markdown("---")
        st.markdown("#### 📊 Simulation Results")

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric(
                "New Pressure Index",
                f"{new_pressure:.3f}",
                delta=f"-{pressure_reduction:.3f}",
                delta_color="normal"
            )
        with result_col2:
            if rank_change > 0:
                st.metric(
                    "New Rank",
                    f"#{new_rank}",
                    delta=f"↓ {rank_change} less critical",
                    delta_color="normal"
                )
            elif rank_change < 0:
                st.metric(
                    "New Rank",
                    f"#{new_rank}",
                    delta=f"↑ {abs(rank_change)} MORE critical",
                    delta_color="inverse"
                )
            else:
                st.metric("New Rank", f"#{new_rank}", delta="No change", delta_color="off")
        with result_col3:
            st.metric("Pressure Reduction", f"{reduction_pct:.1f}%")

        st.markdown("#### Component Breakdown: Before vs After")
        comparison_data = [
            {"Component": "Volume", "State": "Before", "Value": dept_row['Volume_norm'] / 3},
            {"Component": "Volume", "State": "After", "Value": new_volume_norm / 3},
            {"Component": "Intensity", "State": "Before", "Value": dept_row['Intensity_norm'] / 3},
            {"Component": "Intensity", "State": "After", "Value": new_intensity_norm / 3},
            {"Component": "Complexity", "State": "Before", "Value": dept_row['Complexity_norm'] / 3},
            {"Component": "Complexity", "State": "After", "Value": new_complexity_norm / 3},
        ]
        comparison_df = pd.DataFrame(comparison_data)

        fig_comparison = px.bar(
            comparison_df,
            x="Component",
            y="Value",
            color="State",
            barmode="group",
            title=f"Pressure Components: {selected_dept}",
            labels={"Value": "Contribution to Index", "Component": "Pressure Component"},
            color_discrete_map={"Before": "#EF553B", "After": "#00CC96"}
        )
        fig_comparison.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        active_investments = []
        if staffing > 0:
            active_investments.append(f"**Staffing** ({staffing}% volume reduction)")
        if beds > 0:
            active_investments.append(f"**Bed Capacity** ({beds}% intensity reduction)")
        if equipment > 0:
            active_investments.append(f"**Equipment** ({equipment}% complexity reduction)")

        rank_msg = ""
        if rank_change > 0:
            rank_msg = f", moving from rank **#{original_rank}** to **#{new_rank}** (became **{rank_change} positions less critical**)"
        elif rank_change < 0:
            rank_msg = f", but still remains at rank **#{new_rank}** (other departments may need more urgent attention)"
        else:
            rank_msg = f", maintaining rank **#{new_rank}**"

        st.success(
            f"**Investment Summary for {selected_dept}:**\n\n"
            f"With investments in {', '.join(active_investments)}, the department's Pressure Index would decrease "
            f"from **{dept_row['Pressure_Index']:.3f}** to **{new_pressure:.3f}** "
            f"(a **{reduction_pct:.1f}%** reduction){rank_msg}."
        )
    else:
        st.info("👆 Adjust the investment sliders above to simulate resource allocation impact.")
