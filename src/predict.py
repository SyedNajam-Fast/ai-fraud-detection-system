from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "model" / "model_metadata.json"
DEFAULT_PREDICTION_THRESHOLD = 0.5


def load_model():
	if not MODEL_PATH.exists():
		raise FileNotFoundError(
			f"Trained model not found at {MODEL_PATH}. Run model/train_model.py first."
		)
	return joblib.load(MODEL_PATH)


def load_model_metadata() -> Dict[str, object]:
	if not MODEL_METADATA_PATH.exists():
		return {}

	with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
		return json.load(metadata_file)


def get_prediction_threshold() -> float:
	metadata = load_model_metadata()
	threshold = metadata.get("selected_threshold", DEFAULT_PREDICTION_THRESHOLD)
	try:
		return float(threshold)
	except (TypeError, ValueError):
		return DEFAULT_PREDICTION_THRESHOLD


def predict_transaction(transaction: Dict[str, object]) -> Tuple[int, float]:
	model = load_model()
	feature_frame = pd.DataFrame([transaction])
	probability = float(model.predict_proba(feature_frame)[0][1])
	prediction = int(probability >= get_prediction_threshold())
	return prediction, probability
