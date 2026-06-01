"""Quick sanity check that the trained best model is loadable and predicts."""
from pathlib import Path

import numpy as np
import pytest

from src.parkinsons.config import FEATURE_COLUMNS, MODELS_DIR


@pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="Run training first (`python -m src.parkinsons.train`)",
)
def test_predictor_runs():
    from src.parkinsons.predict import ParkinsonsPredictor
    p = ParkinsonsPredictor()
    fake = np.zeros(len(FEATURE_COLUMNS))
    out = p.predict(fake)
    assert out.label in {"Healthy", "Parkinson's"}
    assert 0.0 <= out.probability <= 1.0
    assert out.proba_vector.shape == (2,)
