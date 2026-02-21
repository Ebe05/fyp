"""Hospital operations analytics: department pressure index."""


def calculate_department_pressure(df):
    """Calculate pressure index for each medical specialty."""
    # Filter out rows with missing medical_specialty
    df_filtered = df[df['medical_specialty'].notna() & (df['medical_specialty'] != '')]

    # Group by medical_specialty
    dept_stats = df_filtered.groupby('medical_specialty').agg({
        'encounter_id': 'count',           # Volume
        'time_in_hospital': 'sum',         # Intensity
        'num_lab_procedures': 'mean'       # Complexity
    }).reset_index()

    dept_stats.columns = ['Department', 'Volume', 'Intensity', 'Complexity']

    # Min-max normalization
    for col in ['Volume', 'Intensity', 'Complexity']:
        min_val = dept_stats[col].min()
        max_val = dept_stats[col].max()
        if max_val > min_val:
            dept_stats[f'{col}_norm'] = (dept_stats[col] - min_val) / (max_val - min_val)
        else:
            dept_stats[f'{col}_norm'] = 0.5  # Handle edge case where all values are the same

    # Calculate Pressure Index (equal weights)
    dept_stats['Pressure_Index'] = (
        dept_stats['Volume_norm'] +
        dept_stats['Intensity_norm'] +
        dept_stats['Complexity_norm']
    ) / 3

    return dept_stats.sort_values('Pressure_Index', ascending=False)
