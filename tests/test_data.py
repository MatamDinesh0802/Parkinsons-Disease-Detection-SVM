"""Smoke tests for the data pipeline."""
from src.parkinsons.config import FEATURE_COLUMNS, RAW_DATA, TARGET_COLUMN
from src.parkinsons.data import load_raw, make_splits


def test_raw_data_exists():
    assert RAW_DATA.exists(), f"Missing dataset at {RAW_DATA}"


def test_raw_columns():
    df = load_raw()
    for c in FEATURE_COLUMNS + [TARGET_COLUMN, "name"]:
        assert c in df.columns, f"Missing column: {c}"


def test_split_shapes():
    df = load_raw()
    X_train, X_test, y_train, y_test, scaler = make_splits(df)
    assert X_train.shape[1] == len(FEATURE_COLUMNS)
    assert X_test.shape[1] == len(FEATURE_COLUMNS)
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert set(y_train).issubset({0, 1})
