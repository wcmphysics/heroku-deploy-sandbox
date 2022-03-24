import pandas as pd
import numpy as np
import os

def load_drive_stats(filename:str, path:str) -> pd.DataFrame:
    """Load drive stats file

    Args:
        filename (str): Name of the csv file
        path (str): Path of the repo

    Returns:
        pd.DataFrame: Dataframe containing the drive stats
    """
    file = f"{path}/data/raw/{filename}.csv"
    df = pd.read_csv(file, parse_dates=["date"])
    return df

def calculate_target(df_in, days=30):
    """Merge failure date, calculate the countdown and the target.

    Args:
        df (pd.DataFrame): Drive stats file
        days (int): Time interval for the target calculation

    Returns:
        pd.Series: Target variable
    """
    df = df_in.copy() # Copy to protect input dataframe
    # Series of all the hdds the day they failed to obtain failure date
    failure = df[df.failure == 1]
    # Only use first failure per hdd
    failure = failure.sort_values('date')
    failure = failure.drop_duplicates(keep='first', subset="serial_number")
    # Assign failure dates
    date_failure = df['serial_number'].map(failure.set_index('serial_number')['date'])
    # Days to fail as int
    countdown = (date_failure - df.date).dt.days
    # Remove observations with negative countdown (repaired drives) and with more than 800 days left
    df = df[(countdown >= 0) & (countdown < 800)]
    target = (countdown <= days)[(countdown >= 0) & (countdown < 800)]
    return df, target

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

def drop_cols(df_in) -> pd.DataFrame:
    """Drop columns with missing values. A threshold allows to tune which columns are dropped.

    Args:
        df (_type_): Drive stats data

    Returns:
        pd.DataFrame: Drive stats file with dropped columns
    """
    cols_of_importance = ['smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw',
                            'smart_12_raw', 'smart_183_raw', 'smart_184_raw', 'smart_187_raw',
                            'smart_188_raw', 'smart_189_raw', 'smart_190_raw', 'smart_192_raw',
                            'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_198_raw',
                            'smart_199_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw',
                            'serial_number', 'date']
    df = df_in.loc[:,cols_of_importance]
    return df

def drop_missing_rows(df_in) -> pd.DataFrame:
    """Drop rows with missing values (measurement errors, see EDA)

    Args:
        df (_type_): Drive stats

    Returns:
        pd.DataFrame: Drive stats data with removed rows
    """
    df = df_in.copy().dropna(how="any")
    return df

def drop_duplicate_rows(df_in) -> pd.DataFrame:
    """Drop doublicated rows (measurement errors, see EDA)

    Args:
        df (_type_): Drive stats data

    Returns:
        pd.DataFrame: Drive stats data with removed rows
    """
    df = df_in.copy().drop_duplicates(keep='first', subset=["serial_number", "date"])
    return df

def remove_smart_7_outliers(df_in, threshold=5e10) -> pd.DataFrame:
    """_summary_

    Args:
        df_in (_type_): _description_
        threshold (_type_, optional): _description_. Defaults to 5e10.

    Returns:
        pd.DataFrame: _description_
    """
    df = df_in.copy()
    sn_to_drop = df[df.smart_7_raw > threshold].serial_number.unique()
    for sn in sn_to_drop:
        df = df.drop(df[df.serial_number == sn].index)
    return df

def load_preprocess_data(filename="ST4000DM000_history_total", path=os.getcwd(), days=30) -> pd.DataFrame:
    """Load and preprocess drive stats data

    Args:
        filename (str, optional): Name of the csv file. Defaults to "ST4000DM000_history".
        path (_type_, optional): Path of the repo. Defaults to os.getcwd().

    Returns:
        pd.DataFrame: Dataframe with the drive stats data
    """
    #print("Preprocessing")
    #print("Loading file", filename)
    X = load_drive_stats(filename, path)
    #print("Calculate the target variable")
    X, y = calculate_target(X)
    #print("Removing smart_7_raw outliers")
    X = remove_smart_7_outliers(X)
    #print("Dropping unused columns")
    X = drop_cols(X)
    #print("Dropping missings")
    X = drop_missing_rows(X)
    #print("Dropping dublicated observations")
    X = drop_duplicate_rows(X)
    y = y[X.index]
    #print("Preprocessing finished")
    #print("-----------------------------------------------------")
    return X, y

def load_preprocess_testdata(filename="ST4000DM000_history_total", path=os.getcwd(), days=30) -> pd.DataFrame:
    """Load and preprocess drive stats data

    Args:
        filename (str, optional): Name of the csv file. Defaults to "ST4000DM000_history".
        path (_type_, optional): Path of the repo. Defaults to os.getcwd().

    Returns:
        pd.DataFrame: Dataframe with the drive stats data
    """
    #print("Preprocessing")
    #print("Loading file", filename)
    df = load_drive_stats(filename, path)
    #print("Dropping unused columns")
    df = drop_cols(df)
    #print("Dropping missings")
    df = drop_missing_rows(df)
    #print("Dropping dublicated observations")
    df = drop_duplicate_rows(df)
    #print("Preprocessing finished")
    #print("-----------------------------------------------------")
    return df

def preprocess_testdata(df, days=30) -> pd.DataFrame:
    """Preprocess drive stats data

    Args:
        filename (str, optional): Name of the csv file. Defaults to "ST4000DM000_history".
        path (_type_, optional): Path of the repo. Defaults to os.getcwd().

    Returns:
        pd.DataFrame: Dataframe with the drive stats data
    """
    #print("Preprocessing")
    #print("Dropping unused columns")
    df = drop_cols(df)
    #print("Dropping missings")
    df = drop_missing_rows(df)
    #print("Dropping dublicated observations")
    df = drop_duplicate_rows(df)
    #print("Preprocessing finished")
    #print("-----------------------------------------------------")
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
    df, y = load_preprocess_data(days=30, filename="ST4000DM000_history_total", path=os.getcwd())
