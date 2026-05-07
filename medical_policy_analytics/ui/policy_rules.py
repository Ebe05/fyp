"""Policy Rules tab: Apriori association rule discovery."""

import pandas as pd
import streamlit as st

from medical_policy_analytics.analytics.apriori_rules import (
    discover_rules,
    format_rule_for_policy,
    prepare_apriori_data,
)
from medical_policy_analytics.config import POLICY_DETAILS


def render_policy_rules_tab(df):
    """Render the Policy Rules tab with Apriori association rule discovery"""

    st.markdown("### Policy-Focused Association Rule Discovery")
    st.markdown("Discover actionable patterns: Which interventions can prevent diseases in specific populations?")

    with st.expander("Understanding the Output"):
        st.markdown("""
        **How Rules Are Structured:**
        - **Target Audience**: Demographics to focus the policy on (Elderly, Low Income, Male, Low Education)
        - **Intervention**: Actionable levers that policy can address (Smoking, Exercise, Diet, Obesity, Cholesterol)
        - **Prevents**: The disease outcome that can be reduced

        **Metrics:**
        - **Support**: How common is this pattern? (e.g., 5% = 5% of population has this combination)
        - **Confidence**: How reliable is the prediction? (e.g., 70% = 70% with these factors have the disease)
        - **Lift**: Risk multiplier vs general population (e.g., 2.0x = twice as likely)

        **Column Categories:**
        - **Actionable Levers**: Smoker, No Exercise, Heavy Alcohol, Poor Diet, Obese, High Cholesterol
        - **Target Audience**: Elderly 60+, Male, Low Income, Low Education
        - **Diseases**: Diabetes, Heart Disease, Hypertension, Stroke
        """)

    apriori_col1, apriori_col2, apriori_col3 = st.columns(3)
    with apriori_col1:
        min_support = st.slider("Min Support", 0.01, 0.20, 0.05, 0.01,
                                help="How common must the pattern be?", key="apriori_support")
    with apriori_col2:
        min_confidence = st.slider("Min Confidence", 0.3, 0.9, 0.5, 0.05,
                                   help="How reliable is the rule?", key="apriori_confidence")
    with apriori_col3:
        min_lift = st.slider("Min Lift", 1.0, 3.0, 1.2, 0.1,
                             help="How much does risk increase vs average?", key="apriori_lift")

    if st.button("Run Policy Rule Discovery", type="primary", key="run_apriori"):
        with st.spinner("Mining association rules..."):
            apriori_df = prepare_apriori_data(df)
            rules = discover_rules(apriori_df, min_support, min_confidence, min_lift)

            if len(rules) > 0:
                st.success(f"Found {len(rules)} actionable policy rules!")

                st.markdown("#### Top Policy Recommendations")
                for idx, row in rules.head(10).iterrows():
                    policy = format_rule_for_policy(row['antecedents'], row['consequents'])
                    with st.container():
                        col_left, col_right = st.columns([3, 1])
                        with col_left:
                            st.markdown(f"**Target Audience:** {policy['audience']}")
                            st.markdown(f"**Intervention:** {policy['intervention']}")
                            st.markdown(f"**Prevents:** {policy['outcome']}")
                        with col_right:
                            st.metric("Lift", f"{row['lift']:.2f}x")
                            st.caption(f"Conf: {row['confidence']:.0%}")
                        st.caption(f"Support: {row['support']:.1%} of population")
                        st.divider()

                st.markdown("#### All Discovered Policy Rules")
                policy_data = []
                for idx, row in rules.iterrows():
                    policy = format_rule_for_policy(row['antecedents'], row['consequents'])
                    policy_data.append({
                        "Target Audience": policy['audience'],
                        "Intervention": policy['intervention'],
                        "Prevents": policy['outcome'],
                        "Support": f"{row['support']:.1%}",
                        "Confidence": f"{row['confidence']:.0%}",
                        "Lift": f"{row['lift']:.2f}x"
                    })
                policy_df = pd.DataFrame(policy_data)
                st.dataframe(policy_df, use_container_width=True, hide_index=True)
                st.info("**How to Use These Rules:**\n"
                        "1. **Target Audience** tells you WHO to focus your campaign on\n"
                        "2. **Intervention** tells you WHAT behavior/condition to address\n"
                        "3. **Lift** tells you the IMPACT - higher lift = stronger effect\n"
                        "4. **Support** tells you the SCALE - higher support = more people affected")
            else:
                st.warning("No actionable rules found with current thresholds. Try lowering the minimum values.")
