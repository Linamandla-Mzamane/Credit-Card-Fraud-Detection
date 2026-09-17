from pathlib import Path

import pandas as pd
import pytest

from credit_fraud_pack import data


# A tiny stand-in for creditcard.csv, shared by the download tests.
SAMPLE_CSV = "Time,Amount,Class\n0,100.0,0\n1,50.0,1\n"


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Point data.RAW_DATA_DIR at a temp folder that does not exist yet."""
    fake_raw_dir = tmp_path / "raw"
    monkeypatch.setattr(data, "RAW_DATA_DIR", fake_raw_dir)
    return fake_raw_dir


def test_load_raw_data_reads_csv(tmp_path):
    csv_path = tmp_path / "creditcard.csv"
    csv_path.write_text("Time,Amount,Class\n0,100.0,0\n1,50.0,1\n")

    df = data.load_raw_data(path=csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Time", "Amount", "Class"]
    assert len(df) == 2

def test_download_dataset_skips_download_when_file_exists(raw_dir, monkeypatch):
    # Arrange: data/raw/creditcard.csv is already there.
    raw_dir.mkdir()
    target = raw_dir / data.RAW_CSV_NAME
    target.write_text(SAMPLE_CSV)

    # Replace kagglehub with a fake that fails the test if it is ever called.
    def fail_if_called(**kwargs):
        raise AssertionError("kagglehub should not be called when the CSV exists")

    monkeypatch.setattr(data.kagglehub, "dataset_download", fail_if_called)

    # Act and assert: the existing file's path comes straight back.
    assert data.download_dataset() == target


def test_download_dataset_downloads_into_raw_dir(raw_dir, monkeypatch):
    # Arrange: data/raw/ does not exist yet, so a download is needed.
    calls = {}

    # Behave like kagglehub when output_dir is honoured: record the arguments,
    # write the CSV into output_dir, and return that folder.
    def fake_download(handle, output_dir):
        calls["handle"] = handle
        calls["output_dir"] = output_dir
        (Path(output_dir) / data.RAW_CSV_NAME).write_text(SAMPLE_CSV)
        return output_dir

    monkeypatch.setattr(data.kagglehub, "dataset_download", fake_download)

    # Act
    result = data.download_dataset()

    # Assert
    assert calls == {"handle": "mlg-ulb/creditcardfraud", "output_dir": str(raw_dir)}
    assert result == raw_dir / data.RAW_CSV_NAME
    assert result.exists()
def test_download_dataset_copies_csv_when_output_dir_is_ignored(tmp_path, raw_dir, monkeypatch):
    # Arrange: a separate folder standing in for kagglehub's own cache.
    cache_dir = tmp_path / "kaggle_cache"

    # Behave like kagglehub on Colab: ignore output_dir, put the CSV in
    # kagglehub's cache folder, and return that folder instead.
    def fake_colab_download(handle, output_dir):
        cache_dir.mkdir()
        (cache_dir / data.RAW_CSV_NAME).write_text(SAMPLE_CSV)
        return str(cache_dir)

    monkeypatch.setattr(data.kagglehub, "dataset_download", fake_colab_download)

    # Act
    result = data.download_dataset()

    # Assert: the CSV was copied into data/raw/ with the same contents...
    assert result == raw_dir / data.RAW_CSV_NAME
    assert result.read_text() == SAMPLE_CSV
    # ...and copied, not moved: kagglehub's cached file is still there.
    assert (cache_dir / data.RAW_CSV_NAME).exists()



def test_load_raw_data_uses_default_path_when_none_given(tmp_path, monkeypatch):
    fake_raw_dir = tmp_path / "raw"
    fake_raw_dir.mkdir()
    monkeypatch.setattr(data, "RAW_DATA_DIR", fake_raw_dir)

    csv_path = fake_raw_dir / data.RAW_CSV_NAME
    csv_path.write_text("Time,Amount,Class\n0,100.0,0\n1,50.0,1\n")

    df = data.load_raw_data()

    assert list(df.columns) == ["Time", "Amount", "Class"]
    assert len(df) == 2
