from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = PROJECT_ROOT / "database"
MODEL_DIR = PROJECT_ROOT / "model"
RAW_DATA_DIR = DATA_DIR / "raw" / "creditcardfraud"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

DATABASE_PATH = DATA_DIR / "fraud_detection.db"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
MODEL_PATH = MODEL_DIR / "model.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
FALLBACK_DATASET_PATH = DATA_DIR / "fraud_transactions.csv"
KAGGLE_CSV_PATH = RAW_DATA_DIR / "creditcard.csv"
SAMPLE_PROFILE_CSV_PATH = SAMPLE_DATA_DIR / "sample_profile_dataset.csv"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

REQUIRED_PACKAGES = ["pandas", "numpy", "sklearn", "joblib", "kagglehub"]
MODEL_TRAINING_N_JOBS = 1


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
