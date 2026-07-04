import pandas as pd
import numpy as np

def get_inverter_cols(df, inv_id):
    """
    Returns a dictionary of columns for a given inverter ID.
    Handles the typo in inverter 15.
    """
    inv_str = f"inv_{inv_id:02d}"
    cols = df.columns
    
    dc_current = [c for c in cols if c.startswith(f"{inv_str}_dc_current")][0]
    ac_current = [c for c in cols if c.startswith(f"{inv_str}_ac_current")][0]
    ac_voltage = [c for c in cols if c.startswith(f"{inv_str}_ac_voltage")][0]
    
    # Inverter 15 typo handling
    ac_power_candidates = [c for c in cols if c.startswith(f"{inv_str}_ac_power") or c.startswith(f"{inv_str}_ac_power_iinv")]
    ac_power = ac_power_candidates[0]
    
    # Inverter 05 lacks dc_voltage
    dc_voltage_candidates = [c for c in cols if c.startswith(f"{inv_str}_dc_voltage")]
    dc_voltage = dc_voltage_candidates[0] if len(dc_voltage_candidates) > 0 else None
    
    return {
        'dc_current': dc_current,
        'dc_voltage': dc_voltage,
        'ac_current': ac_current,
        'ac_voltage': ac_voltage,
        'ac_power': ac_power
    }

def clean_and_impute(df):
    """
    Cleans column names, imputes missing values (including Inverter 05 DC voltage),
    and handles missing records.
    """
    print("Starting data cleaning and imputation...")
    df_clean = df.copy()
    
    # 1. Rename Inverter 15 column with typo
    old_typo_col = "inv_15_ac_power_iinv_149653"
    new_correct_col = "inv_15_ac_power_inv_149653"
    if old_typo_col in df_clean.columns:
        df_clean.rename(columns={old_typo_col: new_correct_col}, inplace=True)
        print(f"Renamed column: {old_typo_col} -> {new_correct_col}")
        
    # 2. Impute missing values for general columns (using forward fill then backward fill)
    # We do this for small random dropouts (366 or 1728 rows)
    null_counts_before = df_clean.isnull().sum().sum()
    if null_counts_before > 0:
        df_clean.ffill(inplace=True)
        df_clean.bfill(inplace=True)
        print(f"Imputed {null_counts_before} minor missing cells via ffill/bfill.")
        
    # 3. Impute Inverter 05 DC Voltage
    # Inverter 05 has no dc_voltage column in the dataset.
    # We will find all other DC voltage columns, calculate their average at each timestamp,
    # and create the column 'inv_05_dc_voltage_inv_149600' with these values.
    dc_voltage_cols = [c for c in df_clean.columns if '_dc_voltage_' in c]
    mean_dc_voltage = df_clean[dc_voltage_cols].mean(axis=1)
    
    new_inv05_voltage_col = "inv_05_dc_voltage_inv_149600"
    df_clean[new_inv05_voltage_col] = mean_dc_voltage.astype('float32')
    print(f"Imputed missing DC voltage for Inverter 05 as the mean of other {len(dc_voltage_cols)} inverters.")
    
    # Re-verify no nuls are left
    null_counts_after = df_clean.isnull().sum().sum()
    print(f"Data cleaning completed. Total missing values: {null_counts_after}")
    
    return df_clean

def feature_engineering(df):
    """
    Performs feature engineering for the 24 inverters:
    - Calculates DC Power (kW) = (DC Current * DC Voltage) / 1000
    - Calculates conversion efficiency (%) = (AC Power / DC Power) * 100
    - Clips efficiency to [0, 100] to handle physical limits and outliers.
    """
    print("Performing feature engineering...")
    df_feat = df.copy()
    
    # We will process each of the 24 inverters
    for i in range(1, 25):
        cols = get_inverter_cols(df_feat, i)
        
        # Extract variables
        dc_i = df_feat[cols['dc_current']]
        dc_v = df_feat[cols['dc_voltage']]
        ac_p = df_feat[cols['ac_power']]
        
        # Calculate DC Power (kW)
        dc_power_kw = (dc_i * dc_v) / 1000.0
        df_feat[f"inv_{i:02d}_dc_power_kW"] = dc_power_kw.astype('float32')
        
        # Calculate Inverter Conversion Efficiency (%)
        # Efficiency = (AC Power / DC Power) * 100
        # Only compute when DC power is non-negligible to avoid division by zero
        efficiency = np.where(
            dc_power_kw > 0.01,
            (ac_p / dc_power_kw) * 100.0,
            0.0
        )
        # Clip efficiency to [0, 100]%
        df_feat[f"inv_{i:02d}_efficiency"] = np.clip(efficiency, 0.0, 100.0).astype('float32')
        
    print("Feature engineering completed.")
    return df_feat

def filter_daylight(df):
    """
    Filters the dataset to keep only daylight records (solar radiation active).
    A row is daylight if the mean DC current across all inverters is > 0.5 A.
    """
    dc_current_cols = [c for c in df.columns if '_dc_current_' in c]
    mean_dc_current = df[dc_current_cols].mean(axis=1)
    
    # Define daylight as mean DC current > 0.5 Amps
    is_daylight = mean_dc_current > 0.5
    df_daylight = df[is_daylight]
    
    print(f"Filtered to daylight hours: {df_daylight.shape[0]} rows (out of {df.shape[0]} original).")
    return df_daylight
