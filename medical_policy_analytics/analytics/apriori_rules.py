"""Apriori association rule mining for policy-focused rules."""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from medical_policy_analytics.config import APRIORI_CONFIG


def prepare_apriori_data(df):
    """Prepare binary transaction data for Apriori algorithm with proper binning"""
    apriori_df = pd.DataFrame()

    # === OUTCOME COLUMNS (Diseases) - Consequents ===
    apriori_df['Diabetes'] = (df['Diabetes_012'] > 0).astype(int)
    apriori_df['Heart_Disease'] = df['HeartDiseaseorAttack'].astype(int)
    apriori_df['Hypertension'] = df['HighBP'].astype(int)
    apriori_df['Stroke'] = df['Stroke'].astype(int)

    # === ACTIONABLE LEVERS (Policy Interventions) ===
    # Behavioral
    apriori_df['Smoker'] = df['Smoker'].astype(int)
    apriori_df['No_Exercise'] = (df['PhysActivity'] == 0).astype(int)
    apriori_df['Heavy_Alcohol'] = df['HvyAlcoholConsump'].astype(int)
    apriori_df['Poor_Diet'] = ((df['Fruits'] == 0) & (df['Veggies'] == 0)).astype(int)

    # Modifiable Health Conditions
    apriori_df['Obese'] = (df['BMI'] >= 30).astype(int)
    apriori_df['High_Cholesterol'] = df['HighChol'].astype(int)

    # === NON-ACTIONABLE (Target Audience Demographics) ===
    apriori_df['Elderly_60+'] = (df['Age'] >= 9).astype(int)  # Age 9 = 60-64 in BRFSS encoding
    apriori_df['Male'] = df['Sex'].astype(int)  # 1 = Male, 0 = Female
    apriori_df['Low_Income'] = (df['Income'] <= 4).astype(int)  # Income brackets 1-4 (lower half)
    apriori_df['Low_Education'] = (df['Education'] <= 3).astype(int)  # Education 1-3 (no college)

    return apriori_df


def discover_rules(apriori_df, min_support=0.05, min_confidence=0.5, min_lift=1.2):
    """Run Apriori and filter rules ending in diseases with at least one actionable lever"""

    disease_cols = APRIORI_CONFIG["diseases"]
    actionable_cols = APRIORI_CONFIG["actionable"]

    # Run Apriori to find frequent itemsets
    frequent_itemsets = apriori(apriori_df, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        return pd.DataFrame()

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if len(rules) == 0:
        return pd.DataFrame()

    # Filter 1: Consequent must be disease(s) only
    def is_disease_only_consequent(consequents):
        consequents_set = set(consequents)
        return len(consequents_set) > 0 and consequents_set.issubset(disease_cols)

    disease_rules = rules[rules['consequents'].apply(is_disease_only_consequent)].copy()

    if len(disease_rules) == 0:
        return pd.DataFrame()

    # Filter 2: Antecedents must contain at least one actionable lever
    def has_actionable_lever(antecedents):
        antecedents_set = set(antecedents)
        return len(antecedents_set.intersection(actionable_cols)) > 0

    disease_rules = disease_rules[disease_rules['antecedents'].apply(has_actionable_lever)]

    if len(disease_rules) == 0:
        return pd.DataFrame()

    # Filter 3: Apply lift threshold
    disease_rules = disease_rules[disease_rules['lift'] >= min_lift]

    # Sort by lift (strongest associations first)
    if len(disease_rules) > 0:
        disease_rules = disease_rules.sort_values('lift', ascending=False)

    return disease_rules


def format_rule_for_policy(antecedents, consequents):
    """Split antecedents into Target Audience and Intervention Levers"""
    antecedents_set = set(antecedents)
    actionable_cols = APRIORI_CONFIG["actionable"]
    audience_cols = APRIORI_CONFIG["audience"]

    # Split into categories
    interventions = antecedents_set.intersection(actionable_cols)
    audience = antecedents_set.intersection(audience_cols)

    # Format for display
    intervention_str = ", ".join(sorted(interventions)) if interventions else "General Population"
    audience_str = ", ".join(sorted(audience)) if audience else "All Demographics"
    outcome_str = ", ".join(sorted(set(consequents)))

    return {
        "audience": audience_str,
        "intervention": intervention_str,
        "intervention_list": list(interventions),  # Raw list for expander details
        "outcome": outcome_str
    }
