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
        # Prediabetic/Diabetic + HighBP + BMI>=30
        return (df['Diabetes_012'] > 0) & (df['HighBP'] >= 1) & (df['BMI'] >= 30)
    elif disease_name == "Heart Disease":
        # Heart Disease + HighBP + BMI>=30
        return (df[target_col] == 1) & (df['HighBP'] >= 1) & (df['BMI'] >= 30)
    elif disease_name == "Hypertension":
        # HighBP + High Cholesterol + BMI>=30
        return (df[target_col] == 1) & (df['HighChol'] >= 1) & (df['BMI'] >= 30)
    elif disease_name == "Stroke":
        # Stroke + HighBP + Age>=9 (older adults)
        return (df[target_col] == 1) & (df['HighBP'] >= 1) & (df['Age'] >= 9)
    else:
        return pd.Series([False] * len(df))

def simulate_intervention(df, disease_name, target_col, high_risk_baseline, intervention_type, reduction_pct):
    """Simulate policy intervention impact"""
    if intervention_type == "bmi" and reduction_pct > 0:
        # BMI reduction simulation
        if disease_name == "Diabetes":
            simulated_high_risk = ((df['BMI'] * (1 - reduction_pct/100) >= 30) & 
                                   (df['HighBP'] >= 1) & 
                                   (df['Diabetes_012'] > 0)).sum()
        elif disease_name == "Heart Disease":
            simulated_high_risk = ((df['BMI'] * (1 - reduction_pct/100) >= 30) & 
                                   (df['HighBP'] >= 1) & 
                                   (df[target_col] == 1)).sum()
        elif disease_name == "Hypertension":
            simulated_high_risk = ((df['BMI'] * (1 - reduction_pct/100) >= 30) & 
                                   (df['HighChol'] >= 1) & 
                                   (df[target_col] == 1)).sum()
        else:
            simulated_high_risk = high_risk_baseline
        return high_risk_baseline - simulated_high_risk
    
    elif intervention_type == "bp" and reduction_pct > 0:
        # Blood pressure control simulation for Stroke
        # Simulate reducing the proportion of people with high BP
        simulated_high_risk = int(high_risk_baseline * (1 - reduction_pct/100))
        return high_risk_baseline - simulated_high_risk
    
    return 0

# --- APRIORI HELPER FUNCTIONS ---
def prepare_apriori_data(df):
    """Prepare binary transaction data for Apriori algorithm with proper binning"""
    apriori_df = pd.DataFrame()
    
    # Outcome columns (diseases) - these will be consequents
    apriori_df['Diabetes'] = (df['Diabetes_012'] > 0).astype(int)
    apriori_df['Heart_Disease'] = df['HeartDiseaseorAttack'].astype(int)
    apriori_df['Hypertension'] = df['HighBP'].astype(int)
    apriori_df['Stroke'] = df['Stroke'].astype(int)
    
    # Clinical columns (physical markers) - antecedents
    apriori_df['Obese'] = (df['BMI'] >= 30).astype(int)
    apriori_df['High_Cholesterol'] = df['HighChol'].astype(int)
    apriori_df['Difficulty_Walking'] = df['DiffWalk'].astype(int)
    apriori_df['Elderly_60+'] = (df['Age'] >= 9).astype(int)  # Age 9 = 60-64 in BRFSS encoding
    
    # Behavioral columns (policy levers) - antecedents
    apriori_df['Smoker'] = df['Smoker'].astype(int)
    apriori_df['No_Exercise'] = (df['PhysActivity'] == 0).astype(int)
    apriori_df['Heavy_Alcohol'] = df['HvyAlcoholConsump'].astype(int)
    apriori_df['Poor_Diet'] = ((df['Fruits'] == 0) & (df['Veggies'] == 0)).astype(int)
    
    return apriori_df

def discover_rules(apriori_df, min_support=0.05, min_confidence=0.5, min_lift=1.2):
    """Run Apriori and filter rules ending in diseases"""
    
    # Disease columns (consequents we care about)
    disease_cols = {'Diabetes', 'Heart_Disease', 'Hypertension', 'Stroke'}
    
    # Run Apriori to find frequent itemsets
    frequent_itemsets = apriori(apriori_df, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return pd.DataFrame()
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if len(rules) == 0:
        return pd.DataFrame()
    
    # Filter: Only keep rules where consequent contains at least one disease
    # and consequent is ONLY diseases (no clinical/behavioral factors as outcomes)
    def is_disease_only_consequent(consequents):
        consequents_set = set(consequents)
        return len(consequents_set) > 0 and consequents_set.issubset(disease_cols)
    
    disease_rules = rules[rules['consequents'].apply(is_disease_only_consequent)]
    
    # Filter by lift
    if len(disease_rules) > 0:
        disease_rules = disease_rules[disease_rules['lift'] >= min_lift]
    
    # Sort by lift (strongest associations first)
    if len(disease_rules) > 0:
        disease_rules = disease_rules.sort_values('lift', ascending=False)
    
    return disease_rules

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

    # --- SECTION 4: ASSOCIATION RULE DISCOVERY (APRIORI) ---
    st.markdown("### 🔍 Association Rule Discovery (Apriori Algorithm)")
    st.markdown("Discover hidden patterns: Which combinations of clinical and behavioral factors lead to diseases?")
    
    # Metrics explanation expander
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        - **Support**: How common is this pattern in the population? (e.g., 5% means 5% of people have this combination)
        - **Confidence**: If someone has the antecedents, what's the probability they have the disease? (e.g., 70% confidence)
        - **Lift**: How much more likely is the disease compared to the general population? (e.g., 2.0x means twice as likely)
        
        **Example Rule:** `Smoker, Obese → Diabetes` with Lift 1.8x means smokers who are obese are 1.8 times more likely to have diabetes than the average person.
        
        **Column Categories:**
        - **Diseases (Outcomes):** Diabetes, Heart Disease, Hypertension, Stroke
        - **Clinical (Physical Markers):** Obese, High Cholesterol, Difficulty Walking, Elderly 60+
        - **Behavioral (Policy Levers):** Smoker, No Exercise, Heavy Alcohol, Poor Diet
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
    if st.button("🚀 Run Rule Discovery", type="primary", key="run_apriori"):
        with st.spinner("Mining association rules..."):
            # Prepare data with binning
            apriori_df = prepare_apriori_data(df)
            
            # Discover rules
            rules = discover_rules(apriori_df, min_support, min_confidence, min_lift)
            
            if len(rules) > 0:
                st.success(f"✅ Found {len(rules)} rules leading to diseases!")
                
                # Display top rules
                st.markdown("#### 🏆 Top Disease-Causing Patterns")
                for idx, row in rules.head(10).iterrows():
                    antecedents = ", ".join(list(row['antecedents']))
                    consequents = ", ".join(list(row['consequents']))
                    
                    # Create a formatted rule display
                    st.markdown(f"**{antecedents}** → **{consequents}**")
                    st.caption(f"Support: {row['support']:.1%} | Confidence: {row['confidence']:.1%} | Lift: {row['lift']:.2f}x")
                    st.markdown("")  # Spacing
                
                # Full table
                st.markdown("#### 📋 All Discovered Rules")
                display_df = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                display_df['antecedents'] = display_df['antecedents'].apply(lambda x: ", ".join(list(x)))
                display_df['consequents'] = display_df['consequents'].apply(lambda x: ", ".join(list(x)))
                display_df['support'] = display_df['support'].apply(lambda x: f"{x:.1%}")
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                display_df['lift'] = display_df['lift'].apply(lambda x: f"{x:.2f}x")
                display_df.columns = ['If (Antecedents)', 'Then (Disease)', 'Support', 'Confidence', 'Lift']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Policy insight
                st.info("**Policy Insight:** Rules with behavioral factors (Smoker, No Exercise, Poor Diet) "
                        "represent actionable intervention targets. Higher lift values indicate stronger opportunities for impact.")
            else:
                st.warning("No rules found with current thresholds. Try lowering the minimum values.")
    
    st.markdown("---")

    # --- SECTION 5: UNIFIED RISK FACTOR ANALYSIS ---
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
    
    if selected_disease in ["Diabetes", "Heart Disease", "Hypertension"]:
        # BMI reduction simulation
        bmi_reduction = st.slider("Simulate BMI reduction policy (%):", 0, 10, 0, key="bmi_slider")
    if bmi_reduction > 0:
            reduction_count = simulate_intervention(
                filtered_df, selected_disease, target_col, 
                high_risk_count, "bmi", bmi_reduction
            )
            st.write(f"✨ Reducing BMI by {bmi_reduction}% would remove **{reduction_count:,}** people from high-risk category.")
    
    elif selected_disease == "Stroke":
        # Blood pressure control simulation
        bp_control = st.slider("Simulate BP control program effectiveness (%):", 0, 50, 0, key="bp_slider")
        if bp_control > 0:
            reduction_count = simulate_intervention(
                filtered_df, selected_disease, target_col,
                high_risk_count, "bp", bp_control
            )
            st.write(f"✨ With {bp_control}% BP control, **{reduction_count:,}** fewer high-risk individuals.")

# --- VIEW 1 FUNCTION: POPULATION HEALTH ---
def view_1_population_health(df):
    st.header("📍 View 1: Comparative Population Analytics")
    st.markdown("---")
    
    # Create tabbed interface
    tab_overview, tab_disease = st.tabs(["📊 Overview", "🔬 Disease Analysis"])
    
    with tab_overview:
        render_overview_tab(df.copy())
    
    with tab_disease:
        render_disease_analysis_tab(df.copy())

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
