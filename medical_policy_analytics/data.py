"""Data loading and parsing for CDC and hospital datasets."""

import pandas as pd
import streamlit as st


def parse_ids_mapping(file_path: str) -> dict:
    """Parse the IDS_mapping.csv file."""
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


@st.cache_data
def load_data():
    """Load the cleaned datasets."""
    cdc = pd.read_csv("fyp_data/data/diabetes_012_health_indicators_BRFSS2015_cleaned.csv")
    hosp = pd.read_csv("fyp_data/data/diabetic_data_cleaned.csv")

    # Map Diabetes_012 to readable labels for CDC data
    diabetes_map = {0.0: "No Diabetes", 1.0: "Prediabetes", 2.0: "Diabetes"}
    cdc['Diabetes_Status'] = cdc['Diabetes_012'].map(diabetes_map)

    # Map admission types for hospital data
    try:
        id_mappings = parse_ids_mapping("fyp_data/data/IDS_mapping.csv")
        if 'admission_type_id' in id_mappings:
            hosp['admission_type'] = hosp['admission_type_id'].map(id_mappings['admission_type_id'])
            hosp['admission_type'] = hosp['admission_type'].fillna('Unknown')
    except Exception:
        # Fallback if mapping fails
        hosp['admission_type'] = hosp['admission_type_id'].astype(str)

    return cdc, hosp
