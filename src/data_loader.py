"""
data_loader.py — Load and preprocess the Credit Card Fraud Detection dataset.

Handles loading the raw CSV, basic preprocessing, class separation,
and stratified train/test splitting. Automatically downloads the dataset
from OpenML if not found locally.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def _download_dataset(save_path):
    """
    Download the Credit Card Fraud dataset from OpenML and save locally.

    Parameters
    ----------
    save_path : str
        Path to save the downloaded CSV file.

    Returns
    -------
    pd.DataFrame
        The downloaded dataset.
    """
    print("Dataset not found locally. Downloading from OpenML...")
    try:
        from sklearn.datasets import fetch_openml

        # Credit Card Fraud Detection dataset on OpenML (ID 1597)
        data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
        df = data.frame

        # Rename the target column to 'Class' if needed
        if "Class" not in df.columns:
            # OpenML may name the target differently
            target_col = data.target_names[0] if hasattr(data, "target_names") and data.target_names else df.columns[-1]
            df = df.rename(columns={target_col: "Class"})

        # Ensure ALL columns are numeric (OpenML returns object/category dtypes)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill any NaN created by coercion and set Class as int
        df = df.fillna(0)
        df["Class"] = df["Class"].astype(int)

        # Save locally for future runs
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset downloaded and saved to: {save_path}")
        return df

    except Exception as e:
        print(f"Auto-download failed: {e}")
        raise FileNotFoundError(
            "Could not download dataset automatically. "
            "Please download 'creditcard.csv' from Kaggle and place it in the 'data/' directory.\n"
            "Download link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )


def load_data(filepath=None):
    """
    Load the credit card fraud dataset from CSV.
    If the file is not found, automatically downloads it from OpenML.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'data/creditcard.csv' relative to project root.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    if filepath is None:
        # Default path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(project_root, "data", "creditcard.csv")

    if not os.path.exists(filepath):
        # Try to auto-download
        return _download_dataset(filepath)

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Validate expected columns
    expected_cols = ["Time", "Amount", "Class"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in dataset. Columns: {list(df.columns)}")

    return df


def preprocess_data(df):
    """
    Preprocess the dataset: handle missing values, type casting, and basic cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    # Create a shallow copy to avoid SettingWithCopy warnings
    df = df.copy(deep=False)

    # Fast missing value check
    if df.isnull().any().any():
        print("Found missing values. Dropping rows with NaN...")
        df = df.dropna()
    else:
        print("No missing values found.")

    # Fast duplicate check
    if df.duplicated().any():
        print("Found duplicate rows. Removing duplicates...")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    # Reset index to avoid out-of-bounds/alignment errors in seaborn/pandas
    df = df.reset_index(drop=True)

    # Ensure all columns are numeric (safety net for OpenML-sourced data)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    # Ensure 'Class' column is integer
    df["Class"] = df["Class"].astype(int)

    print(f"Preprocessed dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_by_class(df):
    """
    Separate the dataset into fraud and non-fraud DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset with a 'Class' column.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (non_fraud_df, fraud_df) where Class=0 and Class=1 respectively.
    """
    fraud_df = df[df["Class"] == 1].copy()
    non_fraud_df = df[df["Class"] == 0].copy()

    print(f"Non-fraud (Class=0): {len(non_fraud_df)} samples ({len(non_fraud_df)/len(df)*100:.2f}%)")
    print(f"Fraud (Class=1):     {len(fraud_df)} samples ({len(fraud_df)/len(df)*100:.2f}%)")
    print(f"Imbalance ratio:     1:{len(non_fraud_df)//max(len(fraud_df),1)}")

    return non_fraud_df, fraud_df


def get_train_test_split(df, test_size=0.2, random_state=42):
    """
    Perform a stratified train/test split on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset with a 'Class' column.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Train fraud ratio: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.2f}%)")
    print(f"Test fraud ratio:  {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


def get_dataset_info(df):
    """
    Return a summary dictionary with key dataset statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    dict
        Dictionary containing dataset statistics.
    """
    info = {
        "total_samples": len(df),
        "total_features": len(df.columns) - 1,  # Exclude 'Class'
        "fraud_count": int(df["Class"].sum()),
        "non_fraud_count": int((df["Class"] == 0).sum()),
        "fraud_percentage": float(df["Class"].mean() * 100),
        "non_fraud_percentage": float((1 - df["Class"].mean()) * 100),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return info
