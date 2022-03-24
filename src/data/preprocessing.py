import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

def load_drive_stats(filename, path) -> pd.DataFrame:
    """Load drive stats file

    Args:
        filename (_type_): Name of the csv file
        path (_type_): Path of the repo

    Returns:
        pd.DataFrame: Dataframe containing the drive stats
    """
    file = f"{path}/data/raw/{filename}.csv"
    df = pd.read_csv(file, parse_dates=["date"])
    return df

def countdown(df) -> pd.DataFrame:
    """Create column with failure date and calculate countdown

    Args:
        df (_type_): Drive stats file

    Returns:
        pd.DataFrame: Drive stats with countdown column
    """
    # Series of all the hdds the day they failed to obtain failure date
    failure = df[df.failure == 1]
    # Only use first failure per hdd
    #failure.sort_values('date', inplace=True)
    failure = failure.drop_duplicates(keep='first', subset="serial_number")
    # Assign failure dates
    df['date_failure'] = df['serial_number'].map(failure.set_index('serial_number')['date'])
    # Days to fail as int
    df["countdown"] = (df.date_failure - df.date).dt.days
    df = df[df.countdown >= 0]
    return df

def train_test_splitter(X, y, test_size=0.3, random_state=42) -> pd.DataFrame:
    """Train test split of the drive data

    Args:
        X (_type_): Feature variable
        y (_type_): Target variable
        test_size (float, optional): Size of the test subset. Defaults to 0.3.
        random_state (int, optional): Random state for comparability over different runs. Defaults to 42.

    Returns:
        pd.DataFrame: _description_
    """
    # All the unique serial numbers
    drives = pd.Series(X.serial_number.unique(), name="HDD")
    # Random sampling of drives
    drives_test = drives.sample(int(test_size * len(drives)), random_state=random_state)
    # Remaining drives end up in the train set
    drives_train = drives.drop(drives_test.index, axis=0)
    # Create split X and y
    X_train = X[X.serial_number.isin(drives_train)]
    X_test = X[X.serial_number.isin(drives_test)]
    y_train = y[X.serial_number.isin(drives_train)]
    y_test = y[X.serial_number.isin(drives_test)]
    return X_train, X_test, y_train, y_test

def drop_missing_cols(df, threshold=0.8) -> pd.DataFrame:
    """Drop columns with missing values. A threshold allows to tune which columns are dropped.

    Args:
        df (_type_): Drive stats data
        threshold (float, optional): Percentage of missings in a column so that it is dropped. Defaults to 0.8.

    Returns:
        pd.DataFrame: Drive stats file with dropped columns
    """
    cols_to_drop = df.columns[df.notna().sum() < (threshold * len(df))] # Columns that contain lot of NaNs
    #print("Number of columns to drop:", len(cols_to_drop))
    df = df.drop(cols_to_drop, axis=1) # Drop the cols
    #print("Shape of the dataframe", df.shape)
    return df

def drop_constant_cols(df) -> pd.DataFrame:
    """Drop columns with constant values since they play no role for modeling

    Args:
        df (_type_): Drive stats data

    Returns:
        pd.DataFrame: Drive stats file with dropped columns
    """
    # check columns which only contain 0 values and drop them from the data frame
    cols_to_drop = df.describe().T.query('std == 0').reset_index()['index'].to_list()
    #print(cols_to_drop)
    df = df.drop(cols_to_drop, axis=1)
    return df

def drop_normalized_cols(df) -> pd.DataFrame:
    """Drop columns with normalized values

    Args:
        df (_type_): Drive stats data

    Returns:
        pd.DataFrame: Drive stats file with dropped columns
    """
    # Check which columns contain normalized data
    cols_to_drop = df.columns[df.columns.str.contains("normalized")]
    df.drop(cols_to_drop, axis=1, inplace=True) # Drop the cols
    return df

def drop_missing_rows(df) -> pd.DataFrame:
    """Drop rows with missing values (measurement errors, see EDA)

    Args:
        df (_type_): Drive stats

    Returns:
        pd.DataFrame: Drive stats data with removed rows
    """
    df = df.dropna(how="any")
    return df

def drop_doublicate_rows(df) -> pd.DataFrame:
    """Drop doublicated rows (measurement errors, see EDA)

    Args:
        df (_type_): Drive stats data

    Returns:
        pd.DataFrame: Drive stats data with removed rows
    """
    df.drop_duplicates(keep='first', subset=["serial_number", "date"])
    return df

def calculate_target(df, days=30) -> pd.Series:
    """Calculate target values (will the drive fail in the next days?)

    Args:
        df (_type_): Drive stat data containing the countdown
        days (int, optional): Time window we use for the classification. Defaults to 30.

    Returns:
        pd.Series: Target variable
    """
    return df.countdown <= days

def load_preprocess_data(filename="ST4000DM000_history_total", path=os.getcwd()) -> pd.DataFrame:
    """Load and preprocess drive stats data

    Args:
        filename (str, optional): Name of the csv file. Defaults to "ST4000DM000_history".
        path (_type_, optional): Path of the repo. Defaults to os.getcwd().

    Returns:
        pd.DataFrame: Dataframe with the drive stats data
    """
    df = load_drive_stats(filename, path)
    df = countdown(df)
    df = drop_missing_cols(df)
    df = drop_missing_rows(df)
    df = drop_constant_cols(df)
    df = drop_doublicate_rows(df)
    return df

def save_preprocessed_data(filename="ST4000DM000_history_total", path=os.getcwd()):
    """Load and preprocess the drive stats data and store the result in a csv file

    Args:
        filename (str, optional): Name of the csv file. Defaults to "ST4000DM000_history".
        path (_type_, optional): Path of the repo. Defaults to os.getcwd().

    Returns:
        pd.DataFrame: Dataframe with the drive stats data
    """
    df = load_preprocess_data(filename=filename, path=path)
    file = f"{path}/data/processed/{filename}_preprocessed.csv"
    folder = f"{path}/data/processed/"
    if not os.path.exists(folder):
        os.mkdir(f"{os.getcwd()}/data/processed/")
    df.to_csv(file, index=False)
    return df

if __name__ == "__main__":
    _ = save_preprocessed_data()