import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Policy Analytics", layout="wide")

# --- DISEASE CONFIGURATION ---
DISEASE_CONFIG = {
    "Diabetes": {
        "column": "Diabetes_012",
        "is_binary": False,  # 0/1/2 encoding
        "risk_threshold": 15,  # % for policy alerts
        "labels": {0.0: "No Diabetes", 1.0: "Prediabetes", 2.0: "Diabetes"}
    },
    "Heart Disease": {
        "column": "HeartDiseaseorAttack",
        "is_binary": True,  # 0/1 encoding
        "risk_threshold": 12,
        "labels": {0.0: "No", 1.0: "Yes"}
    },
    "Hypertension": {
        "column": "HighBP",
        "is_binary": True,
        "risk_threshold": 20,
        "labels": {0.0: "No", 1.0: "Yes"}
    },
    "Stroke": {
        "column": "Stroke",
        "is_binary": True,
        "risk_threshold": 8,
        "labels": {0.0: "No", 1.0: "Yes"}
    }
}

# --- RISK FACTORS CONFIGURATION ---
RISK_FACTORS = {
    "Lifestyle Factors": {
        "Smoker": {"column": "Smoker", "labels": {0.0: "Non-Smoker", 1.0: "Smoker"}},
        "Physical Activity": {"column": "PhysActivity", "labels": {0.0: "No Activity", 1.0: "Active"}},
        "Heavy Alcohol": {"column": "HvyAlcoholConsump", "labels": {0.0: "No", 1.0: "Yes"}}
    },
    "Health Factors": {
        "High Blood Pressure": {"column": "HighBP", "labels": {0.0: "Normal", 1.0: "High BP"}},
        "High Cholesterol": {"column": "HighChol", "labels": {0.0: "Normal", 1.0: "High Chol"}},
        "Obesity (BMI≥30)": {"column": "BMI", "is_continuous": True, "threshold": 30}
    }
}

# --- LOAD DATA ---
@st.cache_data  # This keeps the app fast by loading data once
def load_data():
    """Load the cleaned datasets"""
    cdc = pd.read_csv("fyp_data/data/diabetes_012_health_indicators_BRFSS2015_cleaned.csv")
    hosp = pd.read_csv("fyp_data/data/diabetic_data_cleaned.csv")
    
    # Map Diabetes_012 to readable labels for CDC data
    diabetes_map = {0.0: "No Diabetes", 1.0: "Prediabetes", 2.0: "Diabetes"}
    cdc['Diabetes_Status'] = cdc['Diabetes_012'].map(diabetes_map)
    
    # Load IDS mapping for hospital data
    def parse_ids_mapping(file_path):
        """Parse the IDS_mapping.csv file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        mappings = {}
        current_id_type = None
        current_mapping = {}
        
        for line in lines:
            line = line.strip()
            if not line or line == ',':
                if current_id_type and current_mapping:
                    mappings[current_id_type] = current_mapping
                    current_mapping = {}
                    current_id_type = None
                continue
            
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2 and parts[0].strip().endswith('_id'):
                    if current_id_type and current_mapping:
                        mappings[current_id_type] = current_mapping
                    current_id_type = parts[0].strip()
                    current_mapping = {}
                    continue
            
            if current_id_type and ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    try:
                        id_val = int(parts[0].strip())
                        description = parts[1].strip().strip('"')
                        current_mapping[id_val] = description
                    except ValueError:
                        continue
        
        if current_id_type and current_mapping:
            mappings[current_id_type] = current_mapping
        
        return mappings
    
    # Map admission types for hospital data
    try:
        id_mappings = parse_ids_mapping("fyp_data/data/IDS_mapping.csv")
        if 'admission_type_id' in id_mappings:
            hosp['admission_type'] = hosp['admission_type_id'].map(id_mappings['admission_type_id'])
            hosp['admission_type'] = hosp['admission_type'].fillna('Unknown')
    except:
        # Fallback if mapping fails
        hosp['admission_type'] = hosp['admission_type_id'].astype(str)
    
    return cdc, hosp

cdc_df, hosp_df = load_data()

# --- HELPER FUNCTIONS ---
def calculate_high_risk(df, disease_name, target_col):
    """Calculate high-risk population based on disease-specific criteria"""
    if disease_name == "Diabetes":
        # Prediabetic/Diabetic + (HighBP OR Obese OR Smoker OR Inactive OR HighChol OR Poor Diet OR Heavy Alcohol)
        poor_diet = (df['Fruits'] == 0) & (df['Veggies'] == 0)
        return ((df['Diabetes_012'] > 0) & 
                ((df['HighBP'] >= 1) | (df['BMI'] >= 30) | (df['Smoker'] == 1) | 
                 (df['PhysActivity'] == 0) | (df['HighChol'] >= 1) | poor_diet | (df['HvyAlcoholConsump'] == 1)))
    elif disease_name == "Heart Disease":
        # Heart Disease + (HighBP OR Obese OR Smoker OR HighChol OR Inactive OR Poor Diet OR Heavy Alcohol)
        poor_diet = (df['Fruits'] == 0) & (df['Veggies'] == 0)
        return ((df[target_col] == 1) & 
                ((df['HighBP'] >= 1) | (df['BMI'] >= 30) | (df['Smoker'] == 1) | 
                 (df['HighChol'] >= 1) | (df['PhysActivity'] == 0) | poor_diet | (df['HvyAlcoholConsump'] == 1)))
    elif disease_name == "Hypertension":
        # HighBP + (HighChol OR Obese OR Smoker OR Inactive OR Heavy Alcohol)
        return ((df[target_col] == 1) & 
                ((df['HighChol'] >= 1) | (df['BMI'] >= 30) | (df['Smoker'] == 1) | 
                 (df['PhysActivity'] == 0) | (df['HvyAlcoholConsump'] == 1)))
    elif disease_name == "Stroke":
        # Stroke + (HighBP OR Smoker OR HighChol OR Elderly OR Inactive)
        return ((df[target_col] == 1) & 
                ((df['HighBP'] >= 1) | (df['Age'] >= 9) | (df['Smoker'] == 1) | 
                 (df['HighChol'] >= 1) | (df['PhysActivity'] == 0)))
    else:
        return pd.Series([False] * len(df))

def simulate_combined_intervention(df, disease_name, target_col, high_risk_baseline, interventions):
    """
    Simulate combined policy intervention impact.
    interventions: dict of {intervention_type: reduction_pct}
    Returns: dict with individual impacts and combined total
    """
    import numpy as np
    
    # Create simulated dataframe with all interventions applied
    simulated_df = df.copy()
    individual_impacts = {}
    
    # --- Apply each intervention ---
    
    # 1. BMI Reduction
    if interventions.get("bmi", 0) > 0:
        simulated_df['BMI'] = simulated_df['BMI'] * (1 - interventions["bmi"]/100)
    
    # 2. Smoking Cessation - X% of smokers quit
    if interventions.get("smoking", 0) > 0:
        smoker_mask = simulated_df['Smoker'] == 1
        quit_count = int(smoker_mask.sum() * interventions["smoking"] / 100)
        if quit_count > 0:
            np.random.seed(42)  # For reproducibility
            quit_indices = simulated_df[smoker_mask].sample(n=quit_count, random_state=42).index
            simulated_df.loc[quit_indices, 'Smoker'] = 0
    
    # 3. Physical Activity Increase - X% of inactive people start exercising
    if interventions.get("exercise", 0) > 0:
        inactive_mask = simulated_df['PhysActivity'] == 0
        active_count = int(inactive_mask.sum() * interventions["exercise"] / 100)
        if active_count > 0:
            np.random.seed(43)
            active_indices = simulated_df[inactive_mask].sample(n=active_count, random_state=43).index
            simulated_df.loc[active_indices, 'PhysActivity'] = 1
    
    # 4. Cholesterol Control - X% of high cholesterol people achieve normal levels
    if interventions.get("cholesterol", 0) > 0:
        high_chol_mask = simulated_df['HighChol'] == 1
        controlled_count = int(high_chol_mask.sum() * interventions["cholesterol"] / 100)
        if controlled_count > 0:
            np.random.seed(44)
            controlled_indices = simulated_df[high_chol_mask].sample(n=controlled_count, random_state=44).index
            simulated_df.loc[controlled_indices, 'HighChol'] = 0
    
    # 5. Diet Improvement - X% of poor diet people improve
    if interventions.get("diet", 0) > 0:
        poor_diet_mask = (simulated_df['Fruits'] == 0) & (simulated_df['Veggies'] == 0)
        improved_count = int(poor_diet_mask.sum() * interventions["diet"] / 100)
        if improved_count > 0:
            np.random.seed(45)
            improved_indices = simulated_df[poor_diet_mask].sample(n=improved_count, random_state=45).index
            simulated_df.loc[improved_indices, 'Fruits'] = 1
            simulated_df.loc[improved_indices, 'Veggies'] = 1
    
    # 6. Alcohol Reduction - X% of heavy drinkers reduce consumption
    if interventions.get("alcohol", 0) > 0:
        heavy_alcohol_mask = simulated_df['HvyAlcoholConsump'] == 1
        reduced_count = int(heavy_alcohol_mask.sum() * interventions["alcohol"] / 100)
        if reduced_count > 0:
            np.random.seed(46)
            reduced_indices = simulated_df[heavy_alcohol_mask].sample(n=reduced_count, random_state=46).index
            simulated_df.loc[reduced_indices, 'HvyAlcoholConsump'] = 0
    
    # 7. BP Control - X% of high BP people achieve control
    if interventions.get("bp", 0) > 0:
        high_bp_mask = simulated_df['HighBP'] == 1
        controlled_count = int(high_bp_mask.sum() * interventions["bp"] / 100)
        if controlled_count > 0:
            np.random.seed(47)
            controlled_indices = simulated_df[high_bp_mask].sample(n=controlled_count, random_state=47).index
            simulated_df.loc[controlled_indices, 'HighBP'] = 0
    
    # Calculate combined high-risk after all interventions
    combined_high_risk = calculate_high_risk(simulated_df, disease_name, target_col).sum()
    combined_reduction = high_risk_baseline - combined_high_risk
    
    # Calculate individual impacts (one intervention at a time)
    intervention_names = {
        "bmi": "BMI Reduction",
        "smoking": "Smoking Cessation",
        "exercise": "Exercise Increase",
        "cholesterol": "Cholesterol Control",
        "diet": "Diet Improvement",
        "alcohol": "Alcohol Reduction",
        "bp": "BP Control"
    }
    
    for int_type, pct in interventions.items():
        if pct > 0:
            # Simulate only this intervention
            single_df = df.copy()
            
            if int_type == "bmi":
                single_df['BMI'] = single_df['BMI'] * (1 - pct/100)
            elif int_type == "smoking":
                smoker_mask = single_df['Smoker'] == 1
                quit_count = int(smoker_mask.sum() * pct / 100)
                if quit_count > 0:
                    quit_indices = single_df[smoker_mask].sample(n=quit_count, random_state=42).index
                    single_df.loc[quit_indices, 'Smoker'] = 0
            elif int_type == "exercise":
                inactive_mask = single_df['PhysActivity'] == 0
                active_count = int(inactive_mask.sum() * pct / 100)
                if active_count > 0:
                    active_indices = single_df[inactive_mask].sample(n=active_count, random_state=43).index
                    single_df.loc[active_indices, 'PhysActivity'] = 1
            elif int_type == "cholesterol":
                high_chol_mask = single_df['HighChol'] == 1
                controlled_count = int(high_chol_mask.sum() * pct / 100)
                if controlled_count > 0:
                    controlled_indices = single_df[high_chol_mask].sample(n=controlled_count, random_state=44).index
                    single_df.loc[controlled_indices, 'HighChol'] = 0
            elif int_type == "diet":
                poor_diet_mask = (single_df['Fruits'] == 0) & (single_df['Veggies'] == 0)
                improved_count = int(poor_diet_mask.sum() * pct / 100)
                if improved_count > 0:
                    improved_indices = single_df[poor_diet_mask].sample(n=improved_count, random_state=45).index
                    single_df.loc[improved_indices, 'Fruits'] = 1
                    single_df.loc[improved_indices, 'Veggies'] = 1
            elif int_type == "alcohol":
                heavy_alcohol_mask = single_df['HvyAlcoholConsump'] == 1
                reduced_count = int(heavy_alcohol_mask.sum() * pct / 100)
                if reduced_count > 0:
                    reduced_indices = single_df[heavy_alcohol_mask].sample(n=reduced_count, random_state=46).index
                    single_df.loc[reduced_indices, 'HvyAlcoholConsump'] = 0
            elif int_type == "bp":
                high_bp_mask = single_df['HighBP'] == 1
                controlled_count = int(high_bp_mask.sum() * pct / 100)
                if controlled_count > 0:
                    controlled_indices = single_df[high_bp_mask].sample(n=controlled_count, random_state=47).index
                    single_df.loc[controlled_indices, 'HighBP'] = 0
            
            single_high_risk = calculate_high_risk(single_df, disease_name, target_col).sum()
            individual_impacts[intervention_names[int_type]] = high_risk_baseline - single_high_risk
    
    return {
        "combined_reduction": combined_reduction,
        "individual_impacts": individual_impacts,
        "new_high_risk": combined_high_risk
    }

# --- APRIORI CONFIGURATION ---
# Columns categorized by type for policy-focused rule mining
APRIORI_CONFIG = {
    # Diseases (Consequents/Outcomes)
    "diseases": {'Diabetes', 'Heart_Disease', 'Hypertension', 'Stroke'},
    
    # Actionable Levers (can be changed by policy interventions)
    "actionable": {
        'Smoker',           # Smoking cessation programs
        'No_Exercise',      # Physical activity campaigns
        'Heavy_Alcohol',    # Alcohol awareness programs
        'Poor_Diet',        # Nutrition education
        'Obese',            # Weight management programs
        'High_Cholesterol'  # Cholesterol screening & treatment
    },
    
    # Non-Actionable (Target Audience - demographics for policy targeting)
    "audience": {
        'Elderly_60+',      # Age-based targeting
        'Male',             # Gender-based targeting
        'Low_Income',       # Income-based targeting
        'Low_Education'     # Education-based targeting
    }
}

# --- POLICY DETAILS FOR INTERVENTION LEVERS ---
POLICY_DETAILS = {
    "Smoker": {
        "title": "Tobacco Cessation & Prevention Strategy",
        "action": "Increase tobacco excise tax by 15%, fund free Nicotine Replacement Therapy (NRT) via community pharmacies, and mandate graphic health warnings on packaging.",
        "impact": "Estimated 10-15% reduction in smoking rates, lowering cardiovascular and respiratory disease burden."
    },
    "No_Exercise": {
        "title": "Physical Activity Promotion Program",
        "action": "Subsidize gym memberships for low-income brackets, implement 'Active Transport' urban planning, and mandate workplace wellness breaks.",
        "impact": "15-20% improvement in metabolic health markers; reduces obesity and diabetes progression."
    },
    "Heavy_Alcohol": {
        "title": "Alcohol Harm Reduction Initiative",
        "action": "Restrict alcohol advertising, implement minimum unit pricing, and expand free counseling services in community health centers.",
        "impact": "Reduces liver disease risk and secondary hypertension by an estimated 12-18%."
    },
    "Poor_Diet": {
        "title": "Nutritional Standards & Access Reform",
        "action": "Mandate front-of-pack nutrition labeling, subsidize fresh produce in food deserts, and tax sugar-sweetened beverages.",
        "impact": "Improves population-wide metabolic profiles; 10-15% reduction in obesity-related conditions."
    },
    "Obese": {
        "title": "National Weight Management Initiative",
        "action": "Fund community-led fitness programs, prioritize 'Walkability' in urban planning, and provide bariatric care subsidies for severe cases.",
        "impact": "Directly lowers cardiovascular strain and reduces long-term diabetic complications by 20-25%."
    },
    "High_Cholesterol": {
        "title": "Cholesterol Management & Dietary Reform",
        "action": "Implement 'Green-Labeling' on low-saturated-fat foods, subsidize statin access for at-risk populations, and mandate cholesterol screening in annual checkups.",
        "impact": "Reduces biological precursors to Hypertension and Stroke by an estimated 15-20%."
    }
}

# --- APRIORI HELPER FUNCTIONS ---
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

# --- RISK SCORE CALCULATION ---
def calculate_risk_score(df):
    """
    Calculate a weighted risk score for each individual based on health and lifestyle factors.
    Weights are derived from average Spearman correlations with disease outcomes.
    Returns df with new 'risk_score' column (0-100 scale).
    """
    # Create a working copy
    df = df.copy()
    
    # Define disease columns for correlation
    disease_cols = ['Diabetes_012', 'HeartDiseaseorAttack', 'HighBP', 'Stroke']
    
    # Define risk factor columns and their transformations
    risk_factors = {
        # Existing conditions (binary)
        'HighBP': df['HighBP'],
        'HighChol': df['HighChol'],
        'HeartDiseaseorAttack': df['HeartDiseaseorAttack'],
        'Stroke': df['Stroke'],
        'DiffWalk': df['DiffWalk'],
        # Diabetes (binary: any diabetes/prediabetes)
        'Diabetes': (df['Diabetes_012'] > 0).astype(int),
        # Clinical markers (derived)
        'Obese': (df['BMI'] >= 30).astype(int),
        # Behavioral factors
        'Smoker': df['Smoker'],
        'No_Exercise': (df['PhysActivity'] == 0).astype(int),
        'Heavy_Alcohol': df['HvyAlcoholConsump'],
        'Poor_Diet': ((df['Fruits'] == 0) & (df['Veggies'] == 0)).astype(int)
    }
    
    # Calculate correlation-based weights
    # For each risk factor, compute average absolute Spearman correlation with all diseases
    weights = {}
    risk_df = pd.DataFrame(risk_factors)
    
    for factor_name, factor_values in risk_factors.items():
        correlations = []
        for disease in disease_cols:
            # Skip self-correlation for disease factors
            if factor_name in ['HighBP', 'HeartDiseaseorAttack', 'Stroke', 'Diabetes']:
                if (factor_name == 'HighBP' and disease == 'HighBP') or \
                   (factor_name == 'HeartDiseaseorAttack' and disease == 'HeartDiseaseorAttack') or \
                   (factor_name == 'Stroke' and disease == 'Stroke') or \
                   (factor_name == 'Diabetes' and disease == 'Diabetes_012'):
                    continue
            corr = factor_values.corr(df[disease], method='spearman')
            if not pd.isna(corr):
                correlations.append(abs(corr))
        
        # Average correlation as weight (higher correlation = higher weight)
        weights[factor_name] = sum(correlations) / len(correlations) if correlations else 0.05
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate weighted risk score for each individual
    risk_score = pd.Series(0.0, index=df.index)
    for factor_name, factor_values in risk_factors.items():
        risk_score += factor_values * weights[factor_name]
    
    # Normalize to 0-100 scale
    max_possible = sum(weights.values())  # All factors = 1
    df['risk_score'] = (risk_score / max_possible) * 100
    
    # Store weights for display
    df.attrs['risk_weights'] = weights
    
    return df

# --- OVERVIEW TAB FUNCTION ---
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
        # Calculate correlation between diseases and health factors
        corr1 = df[disease_cols + health_cols].corr(method='spearman').loc[disease_cols, health_cols]
        corr1 = corr1.rename(index=disease_names, columns=health_names)
        
        fig1 = px.imshow(corr1, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         title="Diseases × Health")
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with hm_col2:
        st.markdown("#### Diseases & Lifestyle")
        # Calculate correlation between diseases and lifestyle
        corr2 = df[disease_cols + lifestyle_cols].corr(method='spearman').loc[disease_cols, lifestyle_cols]
        corr2 = corr2.rename(index=disease_names, columns=lifestyle_names)
        
        fig2 = px.imshow(corr2, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         title="Diseases × Lifestyle")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    with hm_col3:
        st.markdown("#### Health & Lifestyle")
        # Calculate correlation between health and lifestyle
        corr3 = df[health_cols + lifestyle_cols].corr(method='spearman').loc[health_cols, lifestyle_cols]
        corr3 = corr3.rename(index=health_names, columns=lifestyle_names)
        
        fig3 = px.imshow(corr3, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         title="Health × Lifestyle")
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Insight caption
    st.caption("📊 Spearman rank correlation: Red = positive correlation (factors increase together), Blue = negative correlation (one increases as other decreases)")
    
    st.markdown("---")

    # --- SECTION 2: DISEASE PREVALENCE COMPARISON ---
    st.markdown("### 📈 Disease Prevalence Comparison")
    st.markdown("Which disease affects the largest portion of the population?")
    
    # Calculate prevalence for each disease
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
    
    # Insight callout
    most_prevalent = prevalence_df.iloc[-1]
    st.info(f"**Key Finding:** {most_prevalent['Disease']} is the most prevalent condition, "
            f"affecting {most_prevalent['Prevalence (%)']:.1f}% of the population ({most_prevalent['Cases']:,.0f} cases).")
    
    st.markdown("---")

    # --- SECTION 3: HOW LIFESTYLE CHOICES IMPACT HEALTH CONDITIONS ---
    st.markdown("### 🏃 How Lifestyle Choices Impact Health Conditions")
    st.markdown("Direct cause-effect relationships: Understanding how daily choices impact your health metrics.")
    
    # Prepare health condition data
    health_conditions = {
        "High Blood Pressure": "HighBP",
        "High Cholesterol": "HighChol",
        "Obesity (BMI≥30)": "Is_Obese"  # Already created earlier
    }
    
    # For each lifestyle factor, calculate health condition prevalence
    lifestyle_health_data = []
    for lifestyle_name, lifestyle_config in RISK_FACTORS["Lifestyle Factors"].items():
        lifestyle_col = lifestyle_config["column"]
        labels = lifestyle_config["labels"]
        
        for health_name, health_col in health_conditions.items():
            # Calculate prevalence for No lifestyle factor
            no_lifestyle = df[df[lifestyle_col] == 0]
            if health_col == "Is_Obese":
                no_lifestyle_prev = no_lifestyle[health_col].mean() * 100
            else:
                no_lifestyle_prev = (no_lifestyle[health_col] == 1).mean() * 100
            
            # Calculate prevalence for Has lifestyle factor
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
    
    # Create a chart for each lifestyle factor
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
            
            # Calculate biggest impact
            pivot = factor_data.pivot(index="Health Condition", columns="Status", values="Prevalence (%)")
            differences = pivot.iloc[:, 1] - pivot.iloc[:, 0]  # Assuming column 1 is "Has factor"
            max_diff_condition = differences.abs().idxmax()
            max_diff_value = differences[max_diff_condition]
            
            # Show insight
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
    
    # Create tabs for lifestyle vs health factors
    risk_tab1, risk_tab2 = st.tabs(["Lifestyle Factors", "Health Factors"])
    
    with risk_tab1:
        st.markdown("#### Impact of Lifestyle Choices on All Diseases")
        
        # Calculate prevalence by lifestyle factors
        lifestyle_data = []
        for factor_name, factor_config in RISK_FACTORS["Lifestyle Factors"].items():
            col = factor_config["column"]
            labels = factor_config["labels"]
            
            for disease_name, disease_config in DISEASE_CONFIG.items():
                disease_col = disease_config["column"]
                
                # Group 0 (No factor)
                no_factor_prev = (df[df[col] == 0][disease_col] > 0).mean() * 100
                # Group 1 (Has factor)
                has_factor_prev = (df[df[col] == 1][disease_col] > 0).mean() * 100
                
                lifestyle_data.append({
                    "Risk Factor": factor_name,
                    "Status": labels[0.0],
                    "Disease": disease_name,
                    "Prevalence (%)": no_factor_prev
                })
                lifestyle_data.append({
                    "Risk Factor": factor_name,
                    "Status": labels[1.0],
                    "Disease": disease_name,
                    "Prevalence (%)": has_factor_prev
                })
        
        lifestyle_df = pd.DataFrame(lifestyle_data)
        
        # Create grouped bar chart for each lifestyle factor
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
        
        # Calculate prevalence by health factors
        health_data = []
        for factor_name, factor_config in RISK_FACTORS["Health Factors"].items():
            col = factor_config["column"]
            
            if factor_config.get("is_continuous"):
                # Handle BMI (continuous -> binary)
                threshold = factor_config["threshold"]
                for disease_name, disease_config in DISEASE_CONFIG.items():
                    disease_col = disease_config["column"]
                    
                    # Below threshold
                    below_prev = (df[df[col] < threshold][disease_col] > 0).mean() * 100
                    # Above threshold
                    above_prev = (df[df[col] >= threshold][disease_col] > 0).mean() * 100
                    
                    health_data.append({
                        "Risk Factor": factor_name,
                        "Status": f"BMI < {threshold}",
                        "Disease": disease_name,
                        "Prevalence (%)": below_prev
                    })
                    health_data.append({
                        "Risk Factor": factor_name,
                        "Status": f"BMI ≥ {threshold}",
                        "Disease": disease_name,
                        "Prevalence (%)": above_prev
                    })
            else:
                # Binary factors
                labels = factor_config["labels"]
                for disease_name, disease_config in DISEASE_CONFIG.items():
                    disease_col = disease_config["column"]
                    
                    no_factor_prev = (df[df[col] == 0][disease_col] > 0).mean() * 100
                    has_factor_prev = (df[df[col] == 1][disease_col] > 0).mean() * 100
                    
                    health_data.append({
                        "Risk Factor": factor_name,
                        "Status": labels[0.0],
                        "Disease": disease_name,
                        "Prevalence (%)": no_factor_prev
                    })
                    health_data.append({
                        "Risk Factor": factor_name,
                        "Status": labels[1.0],
                        "Disease": disease_name,
                        "Prevalence (%)": has_factor_prev
                    })
        
        health_df = pd.DataFrame(health_data)
        
        # Create grouped bar chart for each health factor
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
    
    # Final insight
    st.success("**Policy Insight:** Risk factors that elevate prevalence across ALL diseases represent the highest-impact targets "
               "for public health interventions. Focus resources on factors showing the largest gaps between groups.")

# --- POLICY RULES TAB FUNCTION ---
def render_policy_rules_tab(df):
    """Render the Policy Rules tab with Apriori association rule discovery"""
    
    st.markdown("### 🔍 Policy-Focused Association Rule Discovery")
    st.markdown("Discover actionable patterns: Which interventions can prevent diseases in specific populations?")
    
    # Metrics explanation expander
    with st.expander("📖 Understanding the Output"):
        st.markdown("""
        **How Rules Are Structured:**
        - **🎯 Target Audience**: Demographics to focus the policy on (Elderly, Low Income, Male, Low Education)
        - **💡 Intervention**: Actionable levers that policy can address (Smoking, Exercise, Diet, Obesity, Cholesterol)
        - **🏥 Prevents**: The disease outcome that can be reduced
        
        **Metrics:**
        - **Support**: How common is this pattern? (e.g., 5% = 5% of population has this combination)
        - **Confidence**: How reliable is the prediction? (e.g., 70% = 70% with these factors have the disease)
        - **Lift**: Risk multiplier vs general population (e.g., 2.0x = twice as likely)
        
        **Column Categories:**
        - **Actionable Levers**: Smoker, No Exercise, Heavy Alcohol, Poor Diet, Obese, High Cholesterol
        - **Target Audience**: Elderly 60+, Male, Low Income, Low Education
        - **Diseases**: Diabetes, Heart Disease, Hypertension, Stroke
        """)
    
    # Parameter controls
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
    
    # Run discovery button
    if st.button("🚀 Run Policy Rule Discovery", type="primary", key="run_apriori"):
        with st.spinner("Mining association rules..."):
            # Prepare data with binning
            apriori_df = prepare_apriori_data(df)
            
            # Discover rules
            rules = discover_rules(apriori_df, min_support, min_confidence, min_lift)
            
            if len(rules) > 0:
                st.success(f"✅ Found {len(rules)} actionable policy rules!")
                
                # Display top rules in policy format
                st.markdown("#### 🏆 Top Policy Recommendations")
                
                for idx, row in rules.head(10).iterrows():
                    # Format rule for policy display
                    policy = format_rule_for_policy(row['antecedents'], row['consequents'])
                    
                    # Create policy card
                    with st.container():
                        col_left, col_right = st.columns([3, 1])
                        
                        with col_left:
                            st.markdown(f"**🎯 Target Audience:** {policy['audience']}")
                            st.markdown(f"**💡 Intervention:** {policy['intervention']}")
                            st.markdown(f"**🏥 Prevents:** {policy['outcome']}")
                        
                        with col_right:
                            st.metric("Lift", f"{row['lift']:.2f}x")
                            st.caption(f"Conf: {row['confidence']:.0%}")
                        
                        st.caption(f"Support: {row['support']:.1%} of population")
                        
                        # Expandable detailed policy recommendations
                        with st.expander("📝 View Detailed Policy Recommendations"):
                            if policy['intervention_list']:
                                for lever in policy['intervention_list']:
                                    if lever in POLICY_DETAILS:
                                        detail = POLICY_DETAILS[lever]
                                        st.markdown(f"**{detail['title']}**")
                                        st.info(f"**Action:** {detail['action']}")
                                        st.caption(f"📈 **Expected Impact:** {detail['impact']}")
                                        st.divider()
                                    else:
                                        st.write(f"**{lever}** - Policy recommendation under development.")
                            else:
                                st.write("General population intervention - no specific lever identified.")
                            st.caption(f"Rule confidence: {row['confidence']:.0%}")
                        
                        st.markdown("---")
                
                # Full table with policy columns
                st.markdown("#### 📋 All Discovered Policy Rules")
                
                # Create policy-formatted dataframe
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
                
                # Policy insight
                st.info("**How to Use These Rules:**\n"
                        "1. **Target Audience** tells you WHO to focus your campaign on\n"
                        "2. **Intervention** tells you WHAT behavior/condition to address\n"
                        "3. **Lift** tells you the IMPACT - higher lift = stronger effect\n"
                        "4. **Support** tells you the SCALE - higher support = more people affected")
            else:
                st.warning("No actionable rules found with current thresholds. Try lowering the minimum values.")

# --- DISEASE ANALYSIS TAB FUNCTION ---
def render_disease_analysis_tab(df):
    """Render the Disease Analysis tab with disease-specific features"""
    
    # Ensure required columns are numeric
    df['HighBP'] = pd.to_numeric(df['HighBP'], errors='coerce')
    df['Diabetes_012'] = pd.to_numeric(df['Diabetes_012'], errors='coerce')
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['HighChol'] = pd.to_numeric(df['HighChol'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # Disease selector at top of view
    selected_disease = st.selectbox(
        "Select Disease to Analyze", 
        options=list(DISEASE_CONFIG.keys()),
        help="Choose which health condition to analyze"
    )
    
    disease_info = DISEASE_CONFIG[selected_disease]
    target_col = disease_info["column"]
    
    st.subheader(f"Analyzing: {selected_disease}")
    
    # Move Income filter into the view (not sidebar)
    st.markdown("### Policy Intervention Filters")
    income_filter = st.multiselect(
        "Select Target Income Brackets", 
        options=sorted(df['Income'].unique()),
        default=sorted(df['Income'].unique()),
        key="income_filter"
    )
    
    # --- RISK STRATIFICATION LOGIC ---
    # Filter by income
    mask = df['Income'].isin(income_filter)
    filtered_df = df[mask].copy()
    
    # Create disease status column with labels
    filtered_df['Disease_Status'] = filtered_df[target_col].map(disease_info["labels"])
    
    # Calculate high-risk condition
    high_risk_condition = calculate_high_risk(filtered_df, selected_disease, target_col)
    high_risk_count = high_risk_condition.sum()
    total_count = len(filtered_df)
    risk_percentage = (high_risk_count / total_count) * 100 if total_count > 0 else 0
    prevalence = (filtered_df[target_col] > 0).mean() * 100

    # --- METRIC TILES (KPIs) ---
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

    # --- VISUALIZATIONS ---
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

    # --- POLICY RECOMMENDATION ENGINE ---
    st.info("### 🤖 Automated Policy Recommendation")
    threshold = disease_info["risk_threshold"]
    
    if risk_percentage > threshold:
        st.error(f"**CRITICAL:** {selected_disease} high-risk population is at {risk_percentage:.1f}%. "
                 f"Recommend immediate intervention programs targeting these income brackets.")
    elif risk_percentage > threshold * 0.6:
        st.warning(f"**MODERATE:** Elevated {selected_disease} risk detected. Increase screening programs.")
    else:
        st.success(f"**STABLE:** {selected_disease} metrics within acceptable ranges.")

    # --- WHAT-IF SIMULATION ---
    st.markdown("### 🧪 'What-If' Policy Simulation")
    st.markdown("Simulate the combined impact of multiple policy interventions on the high-risk population.")
    
    # Define disease-specific relevance for interventions
    disease_relevance = {
        "Diabetes": {"bmi": "high", "exercise": "high", "diet": "high", "smoking": "medium", "cholesterol": "medium", "alcohol": "low", "bp": "medium"},
        "Heart Disease": {"bmi": "high", "exercise": "high", "diet": "high", "smoking": "high", "cholesterol": "high", "alcohol": "medium", "bp": "high"},
        "Hypertension": {"bmi": "high", "exercise": "medium", "diet": "medium", "smoking": "high", "cholesterol": "high", "alcohol": "high", "bp": "high"},
        "Stroke": {"bmi": "medium", "exercise": "medium", "diet": "medium", "smoking": "high", "cholesterol": "high", "alcohol": "medium", "bp": "high"}
    }
    
    relevance = disease_relevance.get(selected_disease, {})
    
    # Create intervention sliders in two columns
    st.markdown("#### Adjust Intervention Levels")
    
    slider_col1, slider_col2 = st.columns(2)
    
    with slider_col1:
        bmi_reduction = st.slider(
            f"🏋️ BMI Reduction ({'⭐' if relevance.get('bmi') == 'high' else ''})", 
            0, 15, 0, key="sim_bmi",
            help="% reduction in population BMI levels"
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
    
    # BP Control slider (especially relevant for Stroke)
    bp_control = st.slider(
        f"❤️ Blood Pressure Control ({'⭐' if relevance.get('bp') == 'high' else ''})", 
        0, 50, 0, key="sim_bp",
        help="% of high BP cases brought under control"
    )
    
    st.caption("⭐ = High relevance for selected disease")
    
    # Collect interventions
    interventions = {
        "bmi": bmi_reduction,
        "smoking": smoking_cessation,
        "exercise": exercise_increase,
        "cholesterol": cholesterol_control,
        "diet": diet_improvement,
        "alcohol": alcohol_reduction,
        "bp": bp_control
    }
    
    # Check if any intervention is active
    any_intervention = any(v > 0 for v in interventions.values())
    
    if any_intervention:
        # Run combined simulation
        results = simulate_combined_intervention(
            filtered_df, selected_disease, target_col, high_risk_count, interventions
        )
        
        st.markdown("---")
        st.markdown("#### 📊 Simulation Results")
        
        # Summary metrics
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                "High-Risk Before", 
                f"{high_risk_count:,}"
            )
        with result_col2:
            st.metric(
                "High-Risk After", 
                f"{results['new_high_risk']:,}",
                delta=f"-{results['combined_reduction']:,}",
                delta_color="normal"
            )
        with result_col3:
            reduction_pct = (results['combined_reduction'] / high_risk_count * 100) if high_risk_count > 0 else 0
            st.metric(
                "Reduction %", 
                f"{reduction_pct:.1f}%"
            )
        
        # Impact breakdown chart
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
            
            # Find most effective intervention
            if results['individual_impacts']:
                best_intervention = max(results['individual_impacts'].items(), key=lambda x: x[1])
                st.success(
                    f"**Most Effective Single Intervention:** {best_intervention[0]} "
                    f"(removes {best_intervention[1]:,} from high-risk)\n\n"
                    f"**Combined Effect:** All interventions together remove {results['combined_reduction']:,} people "
                    f"({reduction_pct:.1f}% of high-risk population)"
                )
    else:
        st.info("👆 Adjust the sliders above to simulate policy interventions and see their impact.")

# --- FIND YOUR TARGET TAB FUNCTION ---
def render_target_tab(df):
    """Render the Find Your Target tab for risk-based population targeting"""
    
    st.markdown("### 🎯 Find Your Target Population")
    st.markdown("Identify high-risk individuals for targeted policy interventions based on a composite risk score.")
    
    # Calculate risk scores
    df = calculate_risk_score(df)
    
    # --- SECTION A: Risk Score Distribution ---
    st.markdown("#### 📊 Population Risk Score Distribution")
    
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
    
    st.markdown("---")
    
    # --- SECTION B: Intervention Threshold Slider ---
    st.markdown("#### 🎚️ Set Intervention Threshold")
    st.markdown("Select the risk percentile to define your target group. Higher percentiles = smaller, higher-risk groups.")
    
    threshold_percentile = st.slider(
        "Risk Threshold (Percentile)", 
        min_value=50, max_value=99, value=90, step=1,
        help="Select the minimum percentile for intervention. 90 = top 10% highest risk."
    )
    
    # Calculate threshold value and filter
    threshold_value = df['risk_score'].quantile(threshold_percentile / 100)
    high_risk_df = df[df['risk_score'] >= threshold_value]
    
    high_risk_count = len(high_risk_df)
    high_risk_pct = (high_risk_count / len(df)) * 100
    
    # Display gauge and stats
    col_gauge, col_info = st.columns([2, 2])
    
    with col_gauge:
        # Gauge chart for high-risk count
        fig_gauge = px.pie(
            values=[high_risk_count, len(df) - high_risk_count],
            names=['High-Risk (Target)', 'Below Threshold'],
            title=f"Target Group: Top {100 - threshold_percentile}%",
            color_discrete_sequence=['#EF553B', '#E8E8E8']
        )
        fig_gauge.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_info:
        st.markdown("##### 📋 Target Group Summary")
        st.metric("Individuals in Target Group", f"{high_risk_count:,}")
        st.metric("Percentage of Population", f"{high_risk_pct:.1f}%")
        st.metric("Minimum Risk Score", f"{threshold_value:.1f}")
        st.metric("Avg Risk Score (Target)", f"{high_risk_df['risk_score'].mean():.1f}")
        
        st.success(f"**{high_risk_count:,} people ({high_risk_pct:.1f}%)** qualify for intervention based on your threshold.")
    
    st.markdown("---")
    
    # --- SECTION C: Demographic Profile of High-Risk Group ---
    st.markdown("#### 👥 Demographic Profile of Target Group")
    st.markdown("Understand WHO your high-risk individuals are to design targeted outreach.")
    
    # Prepare demographic data
    # Income mapping (BRFSS encoding)
    income_labels = {
        1: '<$10k', 2: '$10-15k', 3: '$15-20k', 4: '$20-25k',
        5: '$25-35k', 6: '$35-50k', 7: '$50-75k', 8: '$75k+'
    }
    high_risk_df['Income_Label'] = high_risk_df['Income'].map(income_labels)
    
    # Age mapping (BRFSS encoding: 1=18-24, 2=25-29, ..., 13=80+)
    age_labels = {
        1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
        6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
        11: '70-74', 12: '75-79', 13: '80+'
    }
    high_risk_df['Age_Label'] = high_risk_df['Age'].map(age_labels)
    
    # Education mapping (BRFSS encoding)
    edu_labels = {
        1: 'Never attended', 2: 'Elementary', 3: 'Some high school',
        4: 'High school grad', 5: 'Some college', 6: 'College grad'
    }
    high_risk_df['Education_Label'] = high_risk_df['Education'].map(edu_labels)
    
    # Gender mapping
    high_risk_df['Gender'] = high_risk_df['Sex'].map({0: 'Female', 1: 'Male'})
    
    # Create demographic charts
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        # Income distribution
        income_counts = high_risk_df['Income_Label'].value_counts().sort_index()
        fig_income = px.bar(
            x=list(income_labels.values()),
            y=[income_counts.get(label, 0) for label in income_labels.values()],
            title="Income Distribution of Target Group",
            labels={'x': 'Income Bracket', 'y': 'Count'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_income, use_container_width=True)
        
        # Age distribution
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
        # Education distribution
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
        
        # Gender split
        gender_counts = high_risk_df['Gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Gender Split of Target Group",
            color_discrete_sequence=['#FF6692', '#19D3F3']
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    st.markdown("---")
    
    # --- SECTION D: Key Insights ---
    st.markdown("#### 💡 Key Insights About Your Target Group")
    
    # Calculate insights
    low_income_pct = (high_risk_df['Income'] <= 4).mean() * 100
    elderly_pct = (high_risk_df['Age'] >= 9).mean() * 100  # 60+
    low_edu_pct = (high_risk_df['Education'] <= 3).mean() * 100
    male_pct = (high_risk_df['Sex'] == 1).mean() * 100
    
    # Calculate prevalent risk factors
    smoker_pct = high_risk_df['Smoker'].mean() * 100
    no_exercise_pct = (high_risk_df['PhysActivity'] == 0).mean() * 100
    obese_pct = (high_risk_df['BMI'] >= 30).mean() * 100
    high_bp_pct = high_risk_df['HighBP'].mean() * 100
    high_chol_pct = high_risk_df['HighChol'].mean() * 100
    
    # Display insights in columns
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("##### 📍 Demographics")
        st.info(f"**{low_income_pct:.0f}%** are in low-income brackets (<$25k)")
        st.info(f"**{elderly_pct:.0f}%** are elderly (60+ years)")
        st.info(f"**{low_edu_pct:.0f}%** have no high school diploma")
        st.info(f"**{male_pct:.0f}%** are male")
    
    with insight_col2:
        st.markdown("##### 🎯 Primary Intervention Needs")
        # Sort by prevalence
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
    
    # Summary recommendation
    st.success(
        f"**Recommended Targeting Strategy:** Focus outreach on "
        f"{'low-income' if low_income_pct > 50 else 'middle-income'} "
        f"{'elderly' if elderly_pct > 40 else 'adult'} populations. "
        f"Primary intervention: {interventions[0][2]}."
    )
    
    st.markdown("---")
    
    # --- SECTION E: National Baseline Comparison ---
    st.markdown("#### 📊 Target Group vs. National Baseline")
    st.markdown("Compare disease prevalence in your target group against the general population to quantify the urgency.")
    
    # Calculate national baseline (full dataset) vs target group prevalence
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
    
    # Create comparison dataframe for plotting
    comparison_data = []
    for disease, values in diseases_comparison.items():
        comparison_data.append({"Disease": disease, "Group": "National Average", "Prevalence (%)": values["national"]})
        comparison_data.append({"Disease": disease, "Group": "Target Group", "Prevalence (%)": values["target"]})
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Side-by-side bar chart
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
    
    # Display multipliers
    st.markdown("##### ⚠️ Risk Multipliers (Target vs. National)")
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
    
    st.markdown("---")
    
    # --- SECTION F: Strategic Policy Recommendations ---
    st.markdown("#### 📋 Strategic Policy Recommendations")
    st.markdown("Based on your target group's profile, here are the prioritized policy interventions:")
    
    # Calculate prevalence of each actionable lever in high-risk group
    lever_prevalence = {
        "Smoker": high_risk_df['Smoker'].mean() * 100,
        "No_Exercise": (high_risk_df['PhysActivity'] == 0).mean() * 100,
        "Heavy_Alcohol": high_risk_df['HvyAlcoholConsump'].mean() * 100,
        "Poor_Diet": ((high_risk_df['Fruits'] == 0) & (high_risk_df['Veggies'] == 0)).mean() * 100,
        "Obese": (high_risk_df['BMI'] >= 30).mean() * 100,
        "High_Cholesterol": high_risk_df['HighChol'].mean() * 100
    }
    
    # Sort by prevalence to identify top priorities
    sorted_levers = sorted(lever_prevalence.items(), key=lambda x: -x[1])
    
    # Display top 3 policy recommendations
    priority_badges = ["🥇", "🥈", "🥉"]
    
    for rank, (lever, prevalence) in enumerate(sorted_levers[:3]):
        detail = POLICY_DETAILS[lever]
        affected_count = int(high_risk_count * prevalence / 100)
        
        # Create policy card
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
            
            st.markdown("---")
    
    # Final action plan summary
    top_3_titles = [POLICY_DETAILS[lever]['title'] for lever, _ in sorted_levers[:3]]
    st.markdown("#### 🚀 Prioritized Action Plan")
    st.markdown(
        f"For your selected target group of **{high_risk_count:,} individuals**, "
        f"implement the following policies in order of priority:\n\n"
        f"1. **{top_3_titles[0]}** (highest impact)\n"
        f"2. **{top_3_titles[1]}**\n"
        f"3. **{top_3_titles[2]}**"
    )

# --- VIEW 1 FUNCTION: POPULATION HEALTH ---
def view_1_population_health(df):
    st.header("📍 View 1: Comparative Population Analytics")
    st.markdown("---")
    
    # Create tabbed interface
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

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Policy Navigation")
view = st.sidebar.radio("Go to:", ["Population Health (CDC)", "Hospital Operations (UCI)"])

# --- VIEW 1: POPULATION HEALTH ---
if view == "Population Health (CDC)":
    view_1_population_health(cdc_df)

# --- VIEW 2: HOSPITAL OPERATIONS ---
elif view == "Hospital Operations (UCI)":
    st.header("Hospital Efficiency & Resource Allocation")
    st.markdown("Focus: Analyzing clinical data to optimize hospital support and spending.")
    
    # Dataset info
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
        # Length of Stay by Admission Type
        fig = px.histogram(hosp_df, x="time_in_hospital", color="admission_type",
                           title="Distribution of Stay Duration by Admission Type",
                           labels={"time_in_hospital": "Days in Hospital", 
                                  "admission_type": "Admission Type",
                                  "count": "Number of Encounters"},
                           barmode="group",
                           nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Readmission by Admission Type
        readmit_by_type = pd.crosstab(hosp_df['admission_type'], 
                                      hosp_df['readmitted'],
                                      normalize='index') * 100
        fig2 = px.bar(readmit_by_type, barmode='group',
                      title="Readmission Rate by Admission Type (%)",
                      labels={"value": "Percentage", "readmitted": "Readmission Status"})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Additional visualizations
    st.subheader("Resource Utilization Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Number of Medications vs Readmission
        fig3 = px.box(hosp_df, x="readmitted", y="num_medications", color="readmitted",
                      title="Number of Medications by Readmission Status",
                      labels={"num_medications": "Number of Medications", 
                             "readmitted": "Readmission Status"})
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Lab Procedures vs Readmission
        fig4 = px.box(hosp_df, x="readmitted", y="num_lab_procedures", color="readmitted",
                      title="Number of Lab Procedures by Readmission Status",
                      labels={"num_lab_procedures": "Number of Lab Procedures",
                             "readmitted": "Readmission Status"})
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Time in hospital distribution
    st.subheader("Stay Duration Analysis")
    fig5 = px.histogram(hosp_df, x="time_in_hospital", 
                        title="Overall Distribution of Hospital Stay Duration",
                        labels={"time_in_hospital": "Days in Hospital", "count": "Number of Encounters"},
                        nbins=30)
    st.plotly_chart(fig5, use_container_width=True)
