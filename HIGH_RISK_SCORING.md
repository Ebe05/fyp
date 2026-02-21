# Disease-specific high-risk scoring (policy targeting)

This project defines a **disease-specific risk profile score** and a **fixed-cutoff high-risk flag** for each disease in the CDC/BRFSS-style dataset.

The purpose is **policy targeting and “what-if” intervention simulation** using variables available in BRFSS-derived data (binary proxies such as `HighBP`, `HighChol`, `Smoker`, etc.).

## Summary of what was implemented

1. **Actual score**
  - For each disease, compute a 0–100 score representing a *calibrated probability proxy* (within the current filtered dataset slice) that an individual is a case of that disease given their risk profile.
2. **Fixed cutoff for “high risk”**
  - For each disease, flag high-risk if `score >= cutoff` (cutoff is fixed per disease).
3. **Disease-specific scoring**
  - Each disease uses its own feature weights (priors), reflecting disease-specific evidence and guideline emphasis.

Implementation lives in:

- `medical_policy_analytics/analytics/risk.py`
- `medical_policy_analytics/config.py`

## Terminology

- **OR (odds ratio)**: ratio of odds in exposed vs unexposed groups. When an OR is available from literature, we convert it to a log-odds weight using w = \ln(OR).
- **Primary vs secondary mode**
  - `mode="primary"`: high risk among those **without** the disease (score above cutoff).
  - `mode="secondary"` (default): high risk among those **with** the disease (score above cutoff).

## Feature engineering (BRFSS proxies)

The scoring system uses standardized binary features created from BRFSS-style columns:

- **Modifiable / intermediate risk proxies**
  - `high_bp`: `HighBP >= 1`
  - `high_chol`: `HighChol >= 1`
  - `obese`: `BMI >= 30`
  - `smoker`: `Smoker == 1`
  - `inactive`: `PhysActivity == 0`
  - `poor_diet`: `Fruits == 0 and Veggies == 0` (proxy for low fruit/veg intake)
  - `heavy_alcohol`: `HvyAlcoholConsump == 1`
  - `dysglycemia`: `Diabetes_012 > 0` (prediabetes/diabetes proxy; used as a predictor for non-diabetes diseases)
- **Non-modifiable / context**
  - `elderly_60plus`: `Age >= 9` (BRFSS age group encoding; 9 corresponds to 60–64)
  - `male`: `Sex == 1` (project convention: 1=Male, 0=Female)
  - `low_income`: `Income <= 4` (project convention)
  - `low_education`: `Education <= 3` (project convention)

## Score equation

For a given disease, create a linear predictor:


z = \sum_i w_i x_i


Then calibrate an intercept b_0 (see calibration below) and compute:


p = sigma(z + b_0) = frac{1}{1 + e^{-(z + b_0)}}


Finally, map to a 0–100 score:


score = 100 cdot p


This score is used for a **fixed cutoff** classification:

- Primary mode:
  - `high_risk = (y == 0) and (score >= cutoff)`
- Secondary mode:
  - `high_risk = (y == 1) and (score >= cutoff)`

Where y is the disease outcome mask used for the current disease:

- Diabetes outcome: `Diabetes_012 > 0`
- Other diseases: `target_col >= 1`

## Intercept-only calibration (non-ML)

Weights w_i are **not fitted** on the dataset; they are treated as **literature-informed priors**.

To make the score numerically meaningful within the current filtered dataset slice, we calibrate only the intercept b_0 so the mean predicted probability matches the observed prevalence:


frac{1}{N}\sum_{j=1}^{N} sigma(z_j + b_0) = bar{y}


This is solved via 1D binary search (monotonic in b_0). It is deterministic and does not change the relative ranking induced by the weights.

## Disease-specific weights and evidence anchoring

Weights are defined in `medical_policy_analytics/config.py` as `DISEASE_SCORE_WEIGHTS`.

- **Stroke**
  - Modifiable-factor weights are anchored to the INTERSTROKE (Lancet 2016) reported odds ratios using w = \ln(OR) where possible.
  - Lipids are proxied using `HighChol` because BRFSS does not contain ApoB/ApoA1 ratio.

Other diseases use guideline-aligned priors reflecting major known risk factors (BP, cholesterol, smoking, obesity, activity, diet, dysglycemia) with intercept calibration absorbing baseline differences across filtered slices.

## Fixed cutoffs

Cutoffs are defined in `medical_policy_analytics/config.py` as `DISEASE_SCORE_CUTOFFS`.

Defaults mirror existing `DISEASE_CONFIG[*]["risk_threshold"]` values to keep policy alert logic consistent with prior UI behavior.

## Limitations (important)

- This is a **risk profile score** built from **proxy variables** in a **cross-sectional** dataset.
- It is **not** a validated prospective absolute risk calculator (e.g., “10-year risk”) and should not be interpreted as such.
- Some literature predictors (e.g., continuous SBP, lab lipids, atrial fibrillation) are not available in BRFSS and are approximated or omitted.

## Key references used for justification (examples)

- INTERSTROKE (global modifiable stroke risk factors; odds ratios): `https://pubmed.ncbi.nlm.nih.gov/27431356/`
- ACC/AHA 2019 primary prevention of cardiovascular disease (risk factor emphasis and prevention framework): `https://pmc.ncbi.nlm.nih.gov/articles/PMC7734661/`
- ADA Standards of Care 2026 (prediabetes/diabetes risk assessment and prevention): `https://pmc.ncbi.nlm.nih.gov/articles/PMC12690170/`
- USPSTF 2021 diabetes screening (age + overweight/obesity as core screening gate): `https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/screening-for-prediabetes-and-type-2-diabetes`

