"""Risk calculation and policy intervention simulation."""

import numpy as np
import pandas as pd

from medical_policy_analytics.config import (
    DISEASE_SCORE_CUTOFFS,
    DISEASE_SCORE_FEATURES,
    DISEASE_SCORE_WEIGHTS,
)

def _sigmoid(x: pd.Series) -> pd.Series:
    # Stable sigmoid for pandas/numpy arrays.
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def _build_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create standardized binary features used by disease-specific score configs."""
    f = pd.DataFrame(index=df.index)

    # Modifiable / intermediate risk proxies
    f["high_bp"] = (pd.to_numeric(df.get("HighBP"), errors="coerce").fillna(0) >= 1).astype(int)
    f["high_chol"] = (pd.to_numeric(df.get("HighChol"), errors="coerce").fillna(0) >= 1).astype(int)
    f["obese"] = (pd.to_numeric(df.get("BMI"), errors="coerce").fillna(0) >= 30).astype(int)
    f["smoker"] = (pd.to_numeric(df.get("Smoker"), errors="coerce").fillna(0) >= 1).astype(int)
    f["inactive"] = (pd.to_numeric(df.get("PhysActivity"), errors="coerce").fillna(0) == 0).astype(int)
    f["poor_diet"] = (
        (pd.to_numeric(df.get("Fruits"), errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(df.get("Veggies"), errors="coerce").fillna(0) == 0)
    ).astype(int)
    f["heavy_alcohol"] = (pd.to_numeric(df.get("HvyAlcoholConsump"), errors="coerce").fillna(0) >= 1).astype(int)
    f["dysglycemia"] = (pd.to_numeric(df.get("Diabetes_012"), errors="coerce").fillna(0) > 0).astype(int)

    # Non-modifiable / context features (for targeting and baseline adjustment)
    # Age 9 corresponds to 60–64 in BRFSS encoding; used as a 60+ proxy across the app.
    f["elderly_60plus"] = (pd.to_numeric(df.get("Age"), errors="coerce").fillna(0) >= 9).astype(int)
    f["male"] = (pd.to_numeric(df.get("Sex"), errors="coerce").fillna(0) == 1).astype(int)
    f["low_income"] = (pd.to_numeric(df.get("Income"), errors="coerce").fillna(99) <= 4).astype(int)
    f["low_education"] = (pd.to_numeric(df.get("Education"), errors="coerce").fillna(99) <= 3).astype(int)

    # Ensure all expected columns exist
    for col in DISEASE_SCORE_FEATURES:
        if col not in f.columns:
            f[col] = 0

    return f[DISEASE_SCORE_FEATURES]


def _get_outcome_mask(df: pd.DataFrame, disease_name: str, target_col: str) -> pd.Series:
    """Binary outcome definition used for calibration and primary/secondary mode filters."""
    if disease_name == "Diabetes":
        y = pd.to_numeric(df.get("Diabetes_012"), errors="coerce").fillna(0) > 0
        return y.astype(int)

    y = pd.to_numeric(df.get(target_col), errors="coerce").fillna(0) >= 1
    return y.astype(int)


def _calibrate_intercept(z_no_intercept: pd.Series, target_prevalence: float) -> float:
    """
    Calibrate intercept b0 so mean(sigmoid(b0 + z)) matches target_prevalence.
    This is a 1D, deterministic calibration (not ML training).
    """
    if target_prevalence <= 0:
        return -20.0
    if target_prevalence >= 1:
        return 20.0

    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pred_mean = float(_sigmoid(z_no_intercept + mid).mean())
        if pred_mean < target_prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calculate_disease_risk_score(
    df: pd.DataFrame,
    disease_name: str,
    target_col: str,
    *,
    calibrate_intercept: bool = True,
) -> pd.Series:
    """
    Disease-specific risk profile score (0–100).

    - Uses literature-informed priors (weights) + optional intercept calibration
      to match observed prevalence in the current dataframe slice.
    - Returns a score interpretable as a calibrated probability proxy *within this dataset slice*.
    """
    weights = DISEASE_SCORE_WEIGHTS.get(disease_name)
    if not weights:
        return pd.Series(0.0, index=df.index, name="risk_score")

    X = _build_score_features(df)
    w = pd.Series({k: float(weights.get(k, 0.0)) for k in X.columns})

    z = X.mul(w, axis=1).sum(axis=1).astype(float)
    if calibrate_intercept:
        y = _get_outcome_mask(df, disease_name, target_col)
        b0 = _calibrate_intercept(z, float(y.mean()))
    else:
        b0 = 0.0

    score = (_sigmoid(z + b0) * 100.0).astype(float)
    score.name = f"{disease_name}_risk_score"
    return score


def calculate_high_risk(
    df: pd.DataFrame,
    disease_name: str,
    target_col: str,
    *,
    mode: str = "secondary",
    cutoff=None,
) -> pd.Series:
    """
    Calculate a high-risk boolean mask using disease-specific scoring.

    Modes:
    - secondary (default): high risk among those WITH the disease (y==1) and score>=cutoff
    - primary: high risk among those WITHOUT the disease (y==0) and score>=cutoff
    """
    if cutoff is None:
        cutoff = float(DISEASE_SCORE_CUTOFFS.get(disease_name, 10))

    y = _get_outcome_mask(df, disease_name, target_col)
    score = calculate_disease_risk_score(df, disease_name, target_col, calibrate_intercept=True)

    if mode == "primary":
        mask = (y == 0) & (score >= cutoff)
    else:
        # Default to secondary prevention targeting to preserve previous semantics.
        mask = (y == 1) & (score >= cutoff)

    return mask.reindex(df.index).fillna(False).astype(bool)


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
