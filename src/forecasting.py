import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import os

from src.preprocessing import get_inverter_cols

def prepare_forecasting_data(df):
    """
    Aggregates AC Power across all 24 inverters, resamples to hourly (1H) resolution,
    interpolates gaps, and performs time-series feature engineering.
    """
    print("Aggregating plant AC power...")
    
    # 1. Extract AC power columns for all 24 inverters and sum them
    ac_cols = []
    for i in range(1, 25):
        cols = get_inverter_cols(df, i)
        ac_cols.append(cols['ac_power'])
        
    df_power = pd.DataFrame(index=df.index)
    df_power['total_ac_power'] = df[ac_cols].sum(axis=1)
    
    # 2. Resample to hourly mean (represents average power generated in kW)
    print("Resampling to hourly frequency...")
    df_hourly = df_power.resample('1h').mean()
    
    # Fill any small gaps in hourly index
    df_hourly = df_hourly.interpolate(method='time')
    
    # 3. Time Series Feature Engineering (Lags and Rolling Statistics)
    print("Generating time-series lag and rolling features...")
    df_feat = df_hourly.copy()
    
    # Lags (in hours)
    for lag in [1, 2, 24, 48]:
        df_feat[f'lag_{lag}'] = df_feat['total_ac_power'].shift(lag)
        
    # Rolling statistics
    df_feat['rolling_mean_3'] = df_feat['total_ac_power'].shift(1).rolling(window=3).mean()
    df_feat['rolling_mean_24'] = df_feat['total_ac_power'].shift(1).rolling(window=24).mean()
    df_feat['rolling_std_3'] = df_feat['total_ac_power'].shift(1).rolling(window=3).std()
    
    # Calendar features
    df_feat['hour'] = df_feat.index.hour
    df_feat['month'] = df_feat.index.month
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_year'] = df_feat.index.dayofyear
    
    # Drop rows with NaN (due to lags and rolling windows)
    df_feat.dropna(inplace=True)
    
    return df_feat

def run_forecasting_analysis(df_feat, test_days=30):
    """
    Performs train-test split chronologically to avoid data leakage.
    Trains Baseline Persistence, Linear Regression with Lags, and Gradient Boosting forecasters.
    Evaluates them on the test set.
    """
    print(f"Splitting forecasting data (Test size: last {test_days} days)...")
    
    # Split chronologically
    split_date = df_feat.index.max() - pd.Timedelta(days=test_days)
    
    train = df_feat.loc[df_feat.index < split_date]
    test = df_feat.loc[df_feat.index >= split_date]
    
    features = [c for c in df_feat.columns if c != 'total_ac_power']
    target = 'total_ac_power'
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # Model 1: Baseline Persistence (predict today at hour H using yesterday at hour H, i.e., lag_24)
    print("Evaluating Baseline Persistence Model...")
    y_test_pred_base = X_test['lag_24']
    
    # Model 2: Linear Regression with Lags
    print("Training Linear Regression with Lags...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_test_pred_lr = lr_model.predict(X_test)
    
    # Model 3: HistGradientBoostingRegressor
    print("Training Gradient Boosting Forecaster...")
    gb_model = HistGradientBoostingRegressor(random_state=42, max_iter=150, learning_rate=0.05, max_depth=6)
    gb_model.fit(X_train, y_train)
    y_test_pred_gb = gb_model.predict(X_test)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        # Avoid division by zero in MAPE for night hours
        mask = y_true > 1.0 # calculate MAPE on daytime power
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return rmse, mae, mape
        
    base_rmse, base_mae, base_mape = calculate_metrics(y_test, y_test_pred_base)
    lr_rmse, lr_mae, lr_mape = calculate_metrics(y_test, y_test_pred_lr)
    gb_rmse, gb_mae, gb_mape = calculate_metrics(y_test, y_test_pred_gb)
    
    metrics = [
        {
            'Model': 'Baseline Persistence (Lag-24)',
            'RMSE (kW)': base_rmse,
            'MAE (kW)': base_mae,
            'MAPE (%)': base_mape
        },
        {
            'Model': 'Linear Regression with Lags',
            'RMSE (kW)': lr_rmse,
            'MAE (kW)': lr_mae,
            'MAPE (%)': lr_mape
        },
        {
            'Model': 'Gradient Boosting (Hist)',
            'RMSE (kW)': gb_rmse,
            'MAE (kW)': gb_mae,
            'MAPE (%)': gb_mape
        }
    ]
    
    df_metrics = pd.DataFrame(metrics)
    
    # Save forecasting metrics table
    base_dir = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
    tables_dir = os.path.join(base_dir, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(tables_dir, "forecasting_metrics.csv"), index=False)
    print(f"Saved forecasting metrics to {os.path.join(tables_dir, 'forecasting_metrics.csv')}")
    
    return df_metrics, train, test, y_test_pred_base, y_test_pred_lr, y_test_pred_gb
