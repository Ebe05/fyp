"""Configuration constants for disease, risk factors, Apriori, and policy details."""

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
