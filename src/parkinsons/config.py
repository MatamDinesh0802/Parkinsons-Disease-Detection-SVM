"""Paths, hyperparameters, and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "parkinsons.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]

TARGET_COLUMN = "status"

CLASS_NAMES = {0: "Healthy", 1: "Parkinson's"}
