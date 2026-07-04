import pandas as pd
import numpy as np
import os
import time

def load_data(file_path):
    """
    Loads the electrical data from a CSV file.
    Optimizes memory usage by converting float64 columns to float32.
    Parses 'measured_on' as datetime and sets it as index.
    """
    start_time = time.time()
    print(f"Loading data from {os.path.basename(file_path)}...")
    
    # We will read in chunks or load it directly, but specify dtype
    # Let's read headers first to check column count and types
    df_head = pd.read_csv(file_path, nrows=2)
    
    # Define float32 for all numeric columns
    dtypes = {}
    for col in df_head.columns:
        if col != 'measured_on':
            dtypes[col] = 'float32'
            
    # Read the full dataset with optimized data types
    df = pd.read_csv(file_path, dtype=dtypes)
    
    # Convert date and set as index
    df['measured_on'] = pd.to_datetime(df['measured_on'])
    df.set_index('measured_on', inplace=True)
    
    elapsed = time.time() - start_time
    mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns in {elapsed:.2f} seconds.")
    print(f"DataFrame memory usage: {mem_usage:.2f} MB")
    
    return df
