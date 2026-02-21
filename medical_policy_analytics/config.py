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

# --- DISEASE-SPECIFIC RISK SCORING (EVIDENCE-ALIGNED, NON-ML) ---
#
# These weights are intended as *literature-informed priors* for a simple, transparent
# disease-specific risk profile score built from BRFSS-style proxy variables.
#
# Implementation notes:
# - A per-disease intercept is calibrated at runtime so mean predicted probability
#   matches the observed prevalence in the active (filtered) dataset slice.
# - For Stroke, modifiable-factor weights are anchored to INTERSTROKE 2016 odds ratios
#   via w = ln(OR) where available; lipid is a proxy because BRFSS provides HighChol.
#
# Important: This is a risk *profile* score (cross-sectional) and should not be
# interpreted as a validated prospective 10-year absolute risk calculator.
DISEASE_SCORE_FEATURES = [
    # Modifiable / intermediate risk proxies
    "high_bp",
    "high_chol",
    "obese",
    "smoker",
    "inactive",
    "poor_diet",
    "heavy_alcohol",
    "dysglycemia",
    # Non-modifiable / context (used for targeting and baseline adjustment)
    "elderly_60plus",
    "male",
    "low_income",
    "low_education",
]

DISEASE_SCORE_WEIGHTS = {
    # Weights are on the log-odds scale (logistic-score style).
    "Stroke": {
        # INTERSTROKE (Lancet 2016): hypertension, smoking, alcohol, inactivity, diet, obesity, diabetes
        # Hypertension OR~2.98 => ln=1.09 (largest contributor)
        "high_bp": 1.09,
        # Physical activity protective OR~0.60 => inactivity OR~1/0.60=1.67 => ln=0.51
        "inactive": 0.51,
        # Diet quality protective OR~0.60 => poor diet OR~1.67 => ln=0.51 (proxy: Fruits==0 & Veggies==0)
        "poor_diet": 0.51,
        # Smoking OR~1.67 => ln=0.51
        "smoker": 0.51,
        # Heavy episodic/high intake alcohol OR~2.09 => ln=0.74 (proxy: HvyAlcoholConsump)
        "heavy_alcohol": 0.74,
        # Abdominal obesity OR~1.44 => ln=0.36 (proxy: BMI>=30)
        "obese": 0.36,
        # Diabetes OR~1.16 => ln=0.15 (proxy: Diabetes_012>0)
        "dysglycemia": 0.15,
        # Lipids in INTERSTROKE use ApoB/ApoA1; HighChol is a proxy with conservative weight
        "high_chol": 0.25,
        # Context features (smaller weights; calibrated intercept absorbs baseline prevalence)
        "elderly_60plus": 0.35,
        "male": 0.10,
        "low_income": 0.10,
        "low_education": 0.10,
    },
    "Heart Disease": {
        # Guideline-aligned priors (ACC/AHA primary prevention): BP, smoking, cholesterol, diabetes, obesity/inactivity
        "high_bp": 0.75,
        "high_chol": 0.60,
        "smoker": 0.70,
        "dysglycemia": 0.55,
        "obese": 0.35,
        "inactive": 0.30,
        "poor_diet": 0.25,
        "heavy_alcohol": 0.15,
        "elderly_60plus": 0.40,
        "male": 0.20,
        "low_income": 0.15,
        "low_education": 0.15,
    },
    "Hypertension": {
        # Lifestyle and metabolic context for elevated BP risk profile
        "obese": 0.55,
        "heavy_alcohol": 0.35,
        "inactive": 0.25,
        "poor_diet": 0.20,
        "smoker": 0.15,
        "high_chol": 0.20,
        "dysglycemia": 0.20,
        "elderly_60plus": 0.55,
        "male": 0.15,
        "low_income": 0.10,
        "low_education": 0.10,
        # Note: high_bp is intentionally not used as a predictor for Hypertension outcome.
        "high_bp": 0.0,
    },
    "Diabetes": {
        # Screening/prevention priors (USPSTF/ADA): adiposity + age + cardiometabolic/lifestyle factors
        "obese": 0.75,
        "elderly_60plus": 0.35,
        "inactive": 0.30,
        "high_bp": 0.30,
        "high_chol": 0.20,
        "poor_diet": 0.20,
        "heavy_alcohol": 0.10,
        "smoker": 0.10,
        "male": 0.10,
        "low_income": 0.15,
        "low_education": 0.15,
        # Note: dysglycemia is the outcome definition for Diabetes and is not used as a predictor.
        "dysglycemia": 0.0,
    },
}

# Fixed “high risk” cutoffs expressed as score percentages (0–100).
# Defaults mirror DISEASE_CONFIG risk_threshold values for consistency with existing policy alerts.
DISEASE_SCORE_CUTOFFS = {
    "Diabetes": DISEASE_CONFIG["Diabetes"]["risk_threshold"],
    "Heart Disease": DISEASE_CONFIG["Heart Disease"]["risk_threshold"],
    "Hypertension": DISEASE_CONFIG["Hypertension"]["risk_threshold"],
    "Stroke": DISEASE_CONFIG["Stroke"]["risk_threshold"],
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
