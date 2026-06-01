"""Data loading and preprocessing."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_DATA,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA)


def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def make_splits(df: pd.DataFrame):
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train.values, y_test.values, scaler
