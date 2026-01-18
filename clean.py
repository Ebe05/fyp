"""
Data Cleaning Script for Diabetes Datasets
Cleans diabetes_012_health_indicators_BRFSS2015.csv and diabetic_data.csv
"""

import pandas as pd
import numpy as np
import os

def clean_brfss_dataset(input_path, output_path):
    """
    Clean the BRFSS 2015 diabetes dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
    """
    print("=" * 60)
    print("Cleaning BRFSS 2015 Diabetes Dataset")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(input_path)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Check for missing values
    print("\nMissing values before cleaning:")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0])
    
    # Check for infinite values
    print("\nInfinite values:")
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
            print(f"  {col}: {np.isinf(df[col]).sum()} infinite values")
    
    # Replace infinite values with NaN
    if inf_cols:
        df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
    
    # Check for negative values in columns that shouldn't have them
    # BMI, MentHlth, PhysHlth should be non-negative
    non_negative_cols = ['BMI', 'MentHlth', 'PhysHlth']
    for col in non_negative_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"\n  {col}: {negative_count} negative values found, replacing with NaN")
                df.loc[df[col] < 0, col] = np.nan
    
    # Check for unrealistic BMI values (typically 10-60)
    if 'BMI' in df.columns:
        unrealistic_bmi = ((df['BMI'] < 10) | (df['BMI'] > 60)).sum()
        if unrealistic_bmi > 0:
            print(f"\n  BMI: {unrealistic_bmi} unrealistic values (<10 or >60), replacing with NaN")
            df.loc[(df['BMI'] < 10) | (df['BMI'] > 60), 'BMI'] = np.nan
    
    # Check for unrealistic health days (MentHlth, PhysHlth should be 0-30)
    for col in ['MentHlth', 'PhysHlth']:
        if col in df.columns:
            unrealistic = (df[col] > 30).sum()
            if unrealistic > 0:
                print(f"\n  {col}: {unrealistic} values > 30, capping at 30")
                df.loc[df[col] > 30, col] = 30
    
    # Remove rows with missing target variable
    if 'Diabetes_012' in df.columns:
        missing_target = df['Diabetes_012'].isnull().sum()
        if missing_target > 0:
            print(f"\nRemoving {missing_target} rows with missing target variable")
            df = df.dropna(subset=['Diabetes_012'])
    
    # Ensure proper data types (all should be numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Final missing value check
    print("\nMissing values after cleaning:")
    missing_after = df.isnull().sum()
    print(missing_after[missing_after > 0])
    
    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Rows removed: {missing_before.sum() - missing_after.sum()}")
    
    return df


def clean_diabetic_data(input_path, output_path, mapping_path=None):
    """
    Clean the diabetic data dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        mapping_path: Optional path to IDS_mapping.csv for reference
    """
    print("\n" + "=" * 60)
    print("Cleaning Diabetic Data Dataset")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(input_path)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Replace '?' with NaN for proper missing value handling
    print("\nReplacing '?' with NaN...")
    df = df.replace('?', np.nan)
    
    # Count missing values before cleaning
    print("\nMissing values before cleaning:")
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    print(missing_cols)
    
    # Handle specific columns with high missing values
    # Weight column - has many missing values, can drop or impute
    if 'weight' in df.columns:
        weight_missing = df['weight'].isnull().sum()
        print(f"\nWeight column: {weight_missing} missing values ({weight_missing/len(df)*100:.2f}%)")
        # Option: Drop weight column if too many missing, or keep as is
        # For now, we'll keep it but mark as missing
    
    # Payer code - has many missing values
    if 'payer_code' in df.columns:
        payer_missing = df['payer_code'].isnull().sum()
        print(f"Payer code: {payer_missing} missing values ({payer_missing/len(df)*100:.2f}%)")
    
    # Medical specialty - has many missing values
    if 'medical_specialty' in df.columns:
        specialty_missing = df['medical_specialty'].isnull().sum()
        print(f"Medical specialty: {specialty_missing} missing values ({specialty_missing/len(df)*100:.2f}%)")
    
    # Diagnosis columns (diag_1, diag_2, diag_3) - handle missing
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        if col in df.columns:
            diag_missing = df[col].isnull().sum()
            print(f"{col}: {diag_missing} missing values ({diag_missing/len(df)*100:.2f}%)")
    
    # Remove rows with missing target variable
    if 'readmitted' in df.columns:
        missing_target = df['readmitted'].isnull().sum()
        if missing_target > 0:
            print(f"\nRemoving {missing_target} rows with missing target variable")
            df = df.dropna(subset=['readmitted'])
    
    # Clean numerical columns
    numeric_cols = ['encounter_id', 'patient_nbr', 'admission_type_id', 
                    'discharge_disposition_id', 'admission_source_id',
                    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                    'num_medications', 'number_outpatient', 'number_emergency',
                    'number_inpatient', 'number_diagnoses']
    
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Check for negative values where they shouldn't exist
            if col in ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                      'num_medications', 'number_outpatient', 'number_emergency',
                      'number_inpatient', 'number_diagnoses']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"  {col}: {negative_count} negative values, replacing with 0")
                    df.loc[df[col] < 0, col] = 0
    
    # Clean categorical columns - standardize values
    # Race column
    if 'race' in df.columns:
        # Replace NaN with 'Unknown'
        df['race'] = df['race'].fillna('Unknown')
        print(f"\nRace categories: {df['race'].unique()}")
    
    # Gender column
    if 'gender' in df.columns:
        # Remove rows with missing gender (should be rare)
        gender_missing = df['gender'].isnull().sum()
        if gender_missing > 0:
            print(f"Gender: {gender_missing} missing values, removing rows")
            df = df.dropna(subset=['gender'])
    
    # Age column - already in brackets format, keep as is
    if 'age' in df.columns:
        age_missing = df['age'].isnull().sum()
        if age_missing > 0:
            print(f"Age: {age_missing} missing values")
            df = df.dropna(subset=['age'])
    
    # Medication columns - standardize (No, Steady, Up, Down)
    medication_cols = [col for col in df.columns if col in 
                      ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                       'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                       'miglitol', 'troglitazone', 'tolazamide', 'examide',
                       'citoglipton', 'insulin', 'glyburide-metformin',
                       'glipizide-metformin', 'glimepiride-pioglitazone',
                       'metformin-rosiglitazone', 'metformin-pioglitazone']]
    
    for col in medication_cols:
        if col in df.columns:
            # Replace NaN with 'No'
            df[col] = df[col].fillna('No')
            # Standardize values
            df[col] = df[col].replace(['no', 'NO'], 'No')
            df[col] = df[col].replace(['steady', 'Steady', 'STEADY'], 'Steady')
            df[col] = df[col].replace(['up', 'Up', 'UP'], 'Up')
            df[col] = df[col].replace(['down', 'Down', 'DOWN'], 'Down')
    
    # max_glu_serum and A1Cresult
    for col in ['max_glu_serum', 'A1Cresult']:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            df[col] = df[col].replace(['none', 'NONE'], 'None')
            df[col] = df[col].replace(['>7', '>8'], '>7')
            df[col] = df[col].replace(['>8'], '>8')
            df[col] = df[col].replace(['Norm'], 'Normal')
    
    # change column
    if 'change' in df.columns:
        df['change'] = df['change'].fillna('No')
        df['change'] = df['change'].replace(['no', 'NO'], 'No')
        df['change'] = df['change'].replace(['ch', 'Ch', 'CH'], 'Ch')
    
    # diabetesMed column
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = df['diabetesMed'].fillna('No')
        df['diabetesMed'] = df['diabetesMed'].replace(['no', 'NO'], 'No')
        df['diabetesMed'] = df['diabetesMed'].replace(['yes', 'Yes', 'YES'], 'Yes')
    
    # readmitted column - standardize
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].replace(['no', 'NO'], 'NO')
        df['readmitted'] = df['readmitted'].replace(['<30'], '<30')
        df['readmitted'] = df['readmitted'].replace(['>30'], '>30')
    
    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\nRemoving {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    # Final missing value check
    print("\nMissing values after cleaning:")
    missing_after = df.isnull().sum()
    missing_cols_after = missing_after[missing_after > 0]
    if len(missing_cols_after) > 0:
        print(missing_cols_after)
    else:
        print("No missing values remaining!")
    
    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Rows removed: {len(df) - (df.shape[0] + duplicates)}")
    
    return df


def main():
    """Main function to clean both datasets."""
    
    # Define paths
    base_dir = 'fyp_data/data'
    
    # BRFSS dataset
    brfss_input = os.path.join(base_dir, 'diabetes_012_health_indicators_BRFSS2015.csv')
    brfss_output = os.path.join(base_dir, 'diabetes_012_health_indicators_BRFSS2015_cleaned.csv')
    
    # Diabetic data dataset
    diabetic_input = os.path.join(base_dir, 'diabetic_data.csv')
    diabetic_output = os.path.join(base_dir, 'diabetic_data_cleaned.csv')
    mapping_file = os.path.join(base_dir, 'IDS_mapping.csv')
    
    # Clean BRFSS dataset
    if os.path.exists(brfss_input):
        brfss_df = clean_brfss_dataset(brfss_input, brfss_output)
    else:
        print(f"Error: {brfss_input} not found!")
    
    # Clean diabetic data dataset
    if os.path.exists(diabetic_input):
        diabetic_df = clean_diabetic_data(diabetic_input, diabetic_output, mapping_file)
    else:
        print(f"Error: {diabetic_input} not found!")
    
    print("\n" + "=" * 60)
    print("Data Cleaning Complete!")
    print("=" * 60)
    print(f"\nCleaned files saved:")
    print(f"  - {brfss_output}")
    print(f"  - {diabetic_output}")


if __name__ == "__main__":
    main()
