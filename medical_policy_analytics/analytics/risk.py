"""Risk calculation and policy intervention simulation."""

import numpy as np
import pandas as pd


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
    # Create simulated dataframe with all interventions applied
    simulated_df = df.copy()
    individual_impacts = {}

    # --- Apply each intervention ---

    # 1. BMI Reduction - only a % of high-BMI (≥25) individuals get reduced by X%
    bmi_reduction = interventions.get("bmi", 0)
    bmi_coverage = interventions.get("bmi_coverage", 0)
    if bmi_reduction > 0 and bmi_coverage > 0:
        eligible_mask = simulated_df['BMI'] >= 25
        treat_count = int(eligible_mask.sum() * bmi_coverage / 100)
        if treat_count > 0:
            simulated_df['BMI'] = simulated_df['BMI'].astype(float)
            treat_indices = simulated_df[eligible_mask].sample(n=treat_count, random_state=41).index
            simulated_df.loc[treat_indices, 'BMI'] = (
                simulated_df.loc[treat_indices, 'BMI'] * (1 - bmi_reduction / 100)
            )

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
        if int_type not in intervention_names:
            continue
        if int_type == "bmi":
            bmi_pct = interventions.get("bmi", 0)
            bmi_cov = interventions.get("bmi_coverage", 0)
            if bmi_pct <= 0 or bmi_cov <= 0:
                continue
            single_df = df.copy()
            eligible_mask = single_df['BMI'] >= 25
            treat_count = int(eligible_mask.sum() * bmi_cov / 100)
            if treat_count > 0:
                single_df['BMI'] = single_df['BMI'].astype(float)
                treat_indices = single_df[eligible_mask].sample(n=treat_count, random_state=41).index
                single_df.loc[treat_indices, 'BMI'] = (
                    single_df.loc[treat_indices, 'BMI'] * (1 - bmi_pct / 100)
                )
            single_high_risk = calculate_high_risk(single_df, disease_name, target_col).sum()
            individual_impacts[intervention_names["bmi"]] = high_risk_baseline - single_high_risk
            continue
        if pct <= 0:
            continue
        # Simulate only this intervention
        single_df = df.copy()

        if int_type == "smoking":
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
