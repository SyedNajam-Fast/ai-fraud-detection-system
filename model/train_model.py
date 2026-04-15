from __future__ import annotations

from pathlib import Path
from typing import Dict
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "fraud_transactions.csv"


def _ensure_project_root_on_path() -> None:
	if str(PROJECT_ROOT) not in sys.path:
		sys.path.insert(0, str(PROJECT_ROOT))


def _generate_synthetic_dataset(sample_size: int = 1500) -> pd.DataFrame:
	rng = np.random.default_rng(42)
	locations = np.array(["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Online"])
	merchants = np.array(["grocery_store", "electronics_store", "travel", "restaurant", "online_retail"])

	amounts = rng.gamma(shape=2.2, scale=120.0, size=sample_size).round(2)
	times = rng.integers(0, 24, size=sample_size)
	location_values = rng.choice(locations, size=sample_size)
	merchant_values = rng.choice(merchants, size=sample_size)

	fraud_score = (
		(amounts > 500).astype(int)
		+ np.isin(location_values, ["Online", "Rawalpindi"]).astype(int)
		+ np.isin(merchant_values, ["electronics_store", "online_retail", "travel"]).astype(int)
		+ np.isin(times, [0, 1, 2, 3, 23]).astype(int)
	)
	fraud_probability = np.clip(0.05 + 0.18 * fraud_score + rng.normal(0, 0.08, size=sample_size), 0, 1)
	labels = (fraud_probability >= 0.5).astype(int)

	return pd.DataFrame(
		{
			"amount": amounts,
			"time": times,
			"location": location_values,
			"merchant": merchant_values,
			"fraud": labels,
		}
	)


def _load_dataset() -> pd.DataFrame:
	if DATA_PATH.exists():
		dataset = pd.read_csv(DATA_PATH)
		expected_columns = ["amount", "time", "location", "merchant", "fraud"]
		if set(expected_columns).issubset(dataset.columns):
			return dataset[expected_columns]
	return _generate_synthetic_dataset()


def _build_pipeline() -> Pipeline:
	numeric_features = ["amount", "time"]
	categorical_features = ["location", "merchant"]

	preprocessor = ColumnTransformer(
		transformers=[
			("numeric", SimpleImputer(strategy="median"), numeric_features),
			(
				"categorical",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("encoder", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				categorical_features,
			),
		]
	)

	return Pipeline(
		steps=[
			("preprocessor", preprocessor),
			(
				"classifier",
				RandomForestClassifier(
					n_estimators=200,
					random_state=42,
					class_weight="balanced",
				),
			),
		]
	)


def train_and_save_model() -> Dict[str, object]:
	dataset = _load_dataset()
	features = dataset[["amount", "time", "location", "merchant"]]
	target = dataset["fraud"]

	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=42,
		stratify=target,
	)

	pipeline = _build_pipeline()
	pipeline.fit(x_train, y_train)

	predictions = pipeline.predict(x_test)
	accuracy = accuracy_score(y_test, predictions)
	matrix = confusion_matrix(y_test, predictions).tolist()

	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipeline, MODEL_PATH)

	return {
		"accuracy": accuracy,
		"confusion_matrix": matrix,
		"model_path": str(MODEL_PATH),
	}


if __name__ == "__main__":
	_ensure_project_root_on_path()
	results = train_and_save_model()
	print(f"Saved model to {results['model_path']}")
	print(f"Accuracy: {results['accuracy']:.4f}")
	print("Confusion matrix:")
	print(results["confusion_matrix"])
