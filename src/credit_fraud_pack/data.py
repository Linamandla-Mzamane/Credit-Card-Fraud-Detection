import kagglehub
import pandas as pd
from pathlib import Path
from credit_fraud_pack.config import RAW_DATA_DIR

RAW_CSV_NAME = "creditcard.csv"

# Function for downloading dataset from kaggle
def download_dataset() -> str:
    """
    :return: A dataset from kaggle in csv format containing credit card info
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return kagglehub.dataset_download(handle="mlg-ulb/creditcardfraud", output_dir=str(RAW_DATA_DIR))

# Function for loading raw data into a pandas dataframe
def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """
    :param path: A path, where the raw data is located.
    :return: A dataframe with the creditcard.csv dataset loaded into it
    """
    if path is None:
        path = RAW_DATA_DIR / RAW_CSV_NAME
    return pd.read_csv(path)
