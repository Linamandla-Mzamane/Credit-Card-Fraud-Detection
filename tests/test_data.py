import pandas as pd

from credit_fraud_pack import data

def test_load_raw_data_reads_csv(tmp_path):
    csv_path = tmp_path / "creditcard.csv"
    csv_path.write_text("Time,Amount,Class\n0,100.0,0\n1,50.0,1\n")

    df = data.load_raw_data(path=csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Time", "Amount", "Class"]
    assert len(df) == 2

def test_download_dataset_creates_dir_and_calls_kagglehub(tmp_path, monkeypatch):
    fake_raw_dir = tmp_path / "raw"
    monkeypatch.setattr(data, "RAW_DATA_DIR", fake_raw_dir)

    calls = {}

    def fake_download(handle, output_dir):
        calls["handle"] = handle
        calls["output_dir"] = output_dir
        return output_dir

    monkeypatch.setattr(data.kagglehub, "dataset_download", fake_download)

    result = data.download_dataset()

    assert fake_raw_dir.exists()
    assert calls == {"handle": "mlg-ulb/creditcardfraud", "output_dir": str(fake_raw_dir)}
    assert result == str(fake_raw_dir)

def test_load_raw_data_uses_default_path_when_none_given(tmp_path, monkeypatch):
    fake_raw_dir = tmp_path / "raw"
    fake_raw_dir.mkdir()
    monkeypatch.setattr(data, "RAW_DATA_DIR", fake_raw_dir)

    csv_path = fake_raw_dir / data.RAW_CSV_NAME
    csv_path.write_text("Time,Amount,Class\n0,100.0,0\n1,50.0,1\n")

    df = data.load_raw_data()

    assert list(df.columns) == ["Time", "Amount", "Class"]
    assert len(df) == 2
