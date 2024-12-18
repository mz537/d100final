import numpy as np
import pandas as pd


def load_mcdata(file_path):
    """
    Load the dataset from a file path.
    Parameters:
    - file_path (str): Path to the data file.
    Returns:
    - df (DataFrame): Loaded dataset as a pandas DataFrame.
    """

    df = pd.read_csv(file_path, sep="\t", encoding="utf-8")

    return df


def handle_missing_values(df, strategy="mean", drop_columns=None):
    """
    Handle missing values in the dataset.

    Parameters:
    - df (DataFrame): The dataset.
    - strategy (str): Strategy to fill missing values ('mean', 'median', 'most_frequent').
    - drop_columns (list): Columns to drop.

    Returns:
    - df (DataFrame): Dataset with missing values handled.
    """
    print("Handling missing values...")
    if drop_columns:
        df = df.drop(columns=drop_columns)
        print(f"Dropped columns: {drop_columns}")

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if strategy == "mean":
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == "median":
                df[column] = df[column].fillna(df[column].median())
            elif strategy == "most_frequent":
                df[column] = df[column].fillna(df[column].mode()[0])
            print(f"Filled missing values in {column} using {strategy}.")
    return df


def load_mcdata_p(file_path):
    """
    Load the dataset from a Parquet file.

    Parameters:
    - file_path (str): Path to the Parquet file.

    Returns:
    - df (DataFrame): Loaded dataset as a pandas DataFrame.
    """
    print(f"Loading dataset from {file_path}...")
    try:
        # Load the Parquet file
        df = pd.read_parquet(file_path, engine="pyarrow")
        print(
            f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns."
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

    return df
