import pandas as pd
import numpy as np

import os
from sklearn.base import BaseEstimator, TransformerMixin

def unwrap_smart_7(df_in) -> pd.DataFrame:
    """Fix the jumps in the smart_7 feature

    Args:
        df_in (_type_): Drive stats data

    Returns:
        pd.DataFrame: Data with updated feature
    """
    # Copy input dataframe
    df = df_in.copy()
    df["smart_7_mod"] = df.smart_7_raw
    # Extract individual drives
    drives = df.serial_number.unique()
    for drive in drives: # Loop over drives
        # Create dataframe with time series for drive, reindex and store the old index
        temp_data = df[df.serial_number == drive].sort_values("date", ascending=True).reset_index()
        # Calculate the derivate and use spikes to determine jumps
        jumps = temp_data.smart_7_raw.diff() < -5e8
        # Extract index of jumps
        jump_idx = jumps[jumps].index
        # Backup the smart_7_raw series
        smart_7_temp = temp_data.smart_7_raw.copy()
        for idx in jump_idx: # Loop over the jumps
            # Add the value before the jump to all the following  values
            temp_data.loc[idx:, "smart_7_raw"] += smart_7_temp[idx-1]
        # Restore the original index
        temp_data = temp_data.set_index("index")
        # Update the dataframe with the unwrapped data for this drive
        df.loc[temp_data.index,"smart_7_mod"] = temp_data.smart_7_raw
    return df

def calculate_ema(df_in, days=30) -> pd.DataFrame:
    """Calculate the EMA of the features over time.

    Args:
        df_in (_type_): Dataframe with some features

    Returns:
        pd.DataFrame: Dataframe with EMA columns
    """
    # Dataframe with only the relevant data
    df = df_in.copy()
    # Sort the values
    df.sort_values(["serial_number", "date"], inplace=True)
    # Drop date to suppress warnings
    df = df.drop("date", axis=1)
    # Create grouped object
    grouped_data = df.groupby("serial_number")
    # Calculate EMA
    int_series = grouped_data.ewm(span=days, min_periods=0).mean()
    # Extract index from multiindex
    index_series = int_series.index.get_level_values(1)
    # Fix indices
    int_series = pd.DataFrame(int_series.values, index=index_series, columns=int_series.columns)
    # Merge dataframes
    df_out = pd.merge(  left=df_in, right=int_series, how="left", 
                        left_index=True, right_index=True, suffixes=("", "_ema"),
                        )
    return df_out

def calculate_smart_999(df_in, trigger=0.05) -> pd.DataFrame:
    """Calculate the smart_999 feature. If the raw differs from the EMA by more 
    than trigger_percent, the corresponding feature initiates a trigger. Smart_999
    sums over all those triggers.

    Args:
        df_in (_type_): Drive stats data
        trigger (float, optional): Percentage for triggering. Defaults to 0.05.

    Returns:
        pd.DataFrame: Dataframe with features
    """
    df = df_in.copy()
    # Select columns to use for the calculation
    cols_to_use =   ['smart_4_raw', 'smart_5_raw',
                    'smart_12_raw', 'smart_183_raw', 'smart_184_raw',
                    'smart_187_raw', 'smart_188_raw', 'smart_189_raw',
                    'smart_193_raw', 'smart_192_raw', 'smart_197_raw',
                    'smart_198_raw', 'smart_199_raw',
                    ]
    # Loop over columns
    for col in cols_to_use:
        # Check if raw differs from ema by more than 5%
        df[col+"_trigger"] = 1/2 * np.abs((df[col] + df[col+"_ema"]) / df[col+"_ema"]) > (1+trigger)
    #print("Shape after calculation of EMA triggers:", df.shape)
    # Sum over all triggers
    sum_cols = []
    for col in df.columns:
        if 'trigger' in col:
            sum_cols.append(col)
    df["smart_999"] = df[sum_cols].sum(axis=1)
    #print("Shape after calculation of sum of EMA triggers:", df.shape)
    return df

def drop_feats(df_in) -> pd.DataFrame:
    """Drop columns with missing values. A threshold allows to tune which columns are dropped.

    Args:
        df (_type_): Drive stats data

    Returns:
        pd.DataFrame: Drive stats file with dropped columns
    """
    cols_of_importance = ['smart_4_raw', 'smart_5_raw', 'smart_7_mod', 'smart_9_raw',
                            'smart_12_raw', 'smart_183_raw', 'smart_184_raw', 'smart_187_raw',
                            'smart_188_raw', 'smart_189_raw', 'smart_192_raw', 'smart_193_raw',
                            'smart_197_raw', 'smart_198_raw', 'smart_199_raw', 'smart_240_raw',
                            'smart_241_raw', 'smart_242_raw', 'smart_999', 'serial_number']
    df = df_in.loc[:,cols_of_importance]
    return df

class log_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1):
        self.offset = offset

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return np.log(X+self.offset)

def create_features(df_in, days=30, trigger=0.05) -> pd.DataFrame:
    """Create the fancy features.

    Args:
        df_in (_type_): Dataframe as output by preprocessing script
        interval (int, optional): Time interval for EMA. Defaults to 30.
        trigger_percentage (float, optional): Normalized distance between raw and EMA. Defaults to 0.05.

    Returns:
        pd.DataFrame: Dataset with new features
    """
    df = df_in.copy()
    #print("Feature engineering")
    #print("Unwrapping smart_7_raw")
    df = unwrap_smart_7(df)
    #print("Calculating of EMAs")
    df = calculate_ema(df, days=days)
    #print("Calculating smart_999 feature")
    df = calculate_smart_999(df, trigger=trigger)
    #print("Dropping unused columns")
    df = drop_feats(df)
    #print("Feature engineering finished")
    #print("Size if the dataframe:", df.shape)
    #print("-----------------------------------------------------")
    return df

class hdd_preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, days=30, trigger=0.05):
        self.days = days
        self.trigger = trigger

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = create_features(X, days=self.days, trigger=self.trigger)
        X = X.drop("serial_number", axis=1)
        return X

if __name__ == "__main__":
    from src.data.hdd_preprocessing import load_preprocess_data
    df, y = load_preprocess_data(days=30, filename="ST4000DM000_history_total", path=os.getcwd())
    df = create_features(df, days=30, trigger=0.05)