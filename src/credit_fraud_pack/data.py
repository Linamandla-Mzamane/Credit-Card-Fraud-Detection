from pathlib import Path

import kagglehub
import pandas as pd

from credit_fraud_pack.config import RAW_DATA_DIR

RAW_CSV_NAME = "creditcard.csv"

# Function for downloading dataset from kaggle
def download_dataset() -> Path:
    """Download the Kaggle credit-card-fraud dataset into data/raw/.

    :return: Path to the downloaded creditcard.csv file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(
        handle="mlg-ulb/creditcardfraud",
        output_dir=str(RAW_DATA_DIR)
    )
    return RAW_DATA_DIR / RAW_CSV_NAME

# Function for loading raw data into a pandas dataframe
def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw creditcard.csv into a DataFrame.

    :param path: CSV location. Defaults to data/raw/creditcard.csv
    :return: The dataset as a pandas DataFrame
    """
    if path is None:
        path = RAW_DATA_DIR / RAW_CSV_NAME
    return pd.read_csv(path)
