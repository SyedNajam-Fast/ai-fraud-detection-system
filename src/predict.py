from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"


def load_model():
	if not MODEL_PATH.exists():
		raise FileNotFoundError(
			f"Trained model not found at {MODEL_PATH}. Run model/train_model.py first."
		)
	return joblib.load(MODEL_PATH)


def predict_transaction(transaction: Dict[str, object]) -> Tuple[int, float]:
	model = load_model()
	feature_frame = pd.DataFrame([transaction])
	probability = float(model.predict_proba(feature_frame)[0][1])
	prediction = int(probability >= 0.5)
	return prediction, probability
