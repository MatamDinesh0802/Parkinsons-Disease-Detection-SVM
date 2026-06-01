"""Single-sample inference used by the Streamlit app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .config import CLASS_NAMES, FEATURE_COLUMNS, MODELS_DIR


@dataclass
class Prediction:
    label: str
    label_index: int
    probability: float
    proba_vector: np.ndarray


class ParkinsonsPredictor:
    """Loads the trained model + scaler once and predicts on new samples."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.model = joblib.load(models_dir / "best_model.joblib")
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.model_name = (models_dir / "best_model_name.txt").read_text().strip()

    def predict(self, features: dict | np.ndarray) -> Prediction:
        if isinstance(features, dict):
            x = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)
        else:
            x = np.asarray(features, dtype=float).reshape(1, -1)
        x_s = self.scaler.transform(x)
        proba = self.model.predict_proba(x_s)[0]
        idx = int(np.argmax(proba))
        return Prediction(
            label=CLASS_NAMES[idx],
            label_index=idx,
            probability=float(proba[idx]),
            proba_vector=proba,
        )
