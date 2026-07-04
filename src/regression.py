import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import os

from src.preprocessing import get_inverter_cols

def prepare_regression_data(df, inv_id=22):
    """
    Prepares features and target for predicting AC power of a specific inverter.
    Only keeps records where the inverter is actively generating (daylight).
    """
    cols = get_inverter_cols(df, inv_id)
    
    # Active generation: DC current > 0.5 A
    df_active = df[df[cols['dc_current']] > 0.5].copy()
    
    X = df_active[[
        cols['dc_current'],
        cols['dc_voltage'],
        cols['ac_current'],
        cols['ac_voltage']
    ]]
    y = df_active[cols['ac_power']]
    
    # Clean up column names for predictors to make it readable
    X = X.rename(columns={
        cols['dc_current']: 'dc_current',
        cols['dc_voltage']: 'dc_voltage',
        cols['ac_current']: 'ac_current',
        cols['ac_voltage']: 'ac_voltage'
    })
    
    return X, y

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error, avoiding division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0.1 # only calculate on values where true AC power is significant
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_compare_regressors(X, y):
    """
    Splits data, trains Linear Regression, Random Forest, and HistGradientBoosting,
    and returns evaluation metrics for comparison.
    """
    print("Splitting regression data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting (Hist)': HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=6),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=50, max_depth=8, n_jobs=-1)
    }
    
    metrics_list = []
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = root_mean_squared_error(y_test, y_test_pred)
        mape = calculate_mape(y_test, y_test_pred)
        
        r2_tr = r2_score(y_train, y_train_pred)
        mae_tr = mean_absolute_error(y_train, y_train_pred)
        rmse_tr = root_mean_squared_error(y_train, y_train_pred)
        mape_tr = calculate_mape(y_train, y_train_pred)
        
        metrics_list.append({
            'Model': name,
            'Train R2': r2_tr,
            'Train MAE (kW)': mae_tr,
            'Train RMSE (kW)': rmse_tr,
            'Train MAPE (%)': mape_tr,
            'Test R2': r2,
            'Test MAE (kW)': mae,
            'Test RMSE (kW)': rmse,
            'Test MAPE (%)': mape
        })
        
        trained_models[name] = model
        predictions[name] = y_test_pred
        
    df_metrics = pd.DataFrame(metrics_list)
    
    # Save metrics table
    base_dir = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
    tables_dir = os.path.join(base_dir, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(tables_dir, "supervised_regression_metrics.csv"), index=False)
    print(f"Saved regression metrics to {os.path.join(tables_dir, 'supervised_regression_metrics.csv')}")
    
    return df_metrics, X_test, y_test, predictions
