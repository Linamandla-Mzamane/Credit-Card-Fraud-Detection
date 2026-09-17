import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from credit_fraud_pack.config import RAW_DATA_DIR

RAW_CSV_NAME = "creditcard.csv"

# Function for downloading dataset from kaggle
def download_dataset() -> Path:
    """Make sure creditcard.csv is available at data/raw/creditcard.csv.

    Downloads the dataset from Kaggle only if the file is missing. On Google
    Colab, kagglehub ignores output_dir and serves files from its own cache,
    so the CSV is copied into data/raw/ from wherever kagglehub put it.

    :return: Path to data/raw/creditcard.csv, which exists when this returns.
    """
    target = RAW_DATA_DIR / RAW_CSV_NAME

    if target.exists():
        return target

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_dir = kagglehub.dataset_download(
        handle="mlg-ulb/creditcardfraud",
        output_dir=str(RAW_DATA_DIR)
    )

    source = Path(downloaded_dir) / RAW_CSV_NAME

    if source.resolve() != target.resolve():
        shutil.copy2(source, target)

    return target

# Function for loading raw data into a pandas dataframe
def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw creditcard.csv into a DataFrame.

    :param path: CSV location. Defaults to data/raw/creditcard.csv
    :return: The dataset as a pandas DataFrame
    """
    if path is None:
        path = RAW_DATA_DIR / RAW_CSV_NAME
    return pd.read_csv(path)
