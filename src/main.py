from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from model.train_model import train_and_save_model
from src.db import (  # noqa: E402
	create_fraud_alert,
	fetch_transaction,
	initialize_database,
	store_prediction,
)
from src.insert_data import insert_sample_transaction  # noqa: E402
from src.predict import MODEL_PATH, predict_transaction  # noqa: E402


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run end-to-end fraud detection workflow.")
	parser.add_argument(
		"--force-train",
		action="store_true",
		help="Retrain and overwrite the model before processing a transaction.",
	)
	return parser.parse_args()


def ensure_model_available(force_train: bool = False) -> dict[str, object] | None:
	"""Train and persist the model when required.

	Returns model metrics only when training happens in this run.
	"""
	if not force_train and MODEL_PATH.exists():
		return None
	return train_and_save_model()


def run_workflow(force_train: bool = False) -> None:
	initialize_database()
	model_metrics = ensure_model_available(force_train=force_train)

	# System workflow from context:
	# 1) Insert transaction 2) Fetch transaction 3) Predict
	# 4) Store prediction 5) Create alert if fraud
	transaction_id = insert_sample_transaction()
	transaction = fetch_transaction(transaction_id)

	if transaction is None:
		raise RuntimeError("Inserted transaction could not be retrieved from the database.")

	payload = {
		"amount": transaction["amount"],
		"time": transaction["time"],
		"location": transaction["location"],
		"merchant": transaction["merchant"],
	}
	prediction, probability = predict_transaction(payload)
	store_prediction(transaction_id, prediction, probability)

	alert_id = None
	if prediction == 1:
		alert_id = create_fraud_alert(transaction_id)

	if model_metrics is not None:
		print("Model training metrics (new model trained):")
		print(f"Accuracy: {model_metrics['accuracy']:.4f}")
		print("Confusion matrix:")
		print(model_metrics["confusion_matrix"])
	else:
		print(f"Using existing trained model: {MODEL_PATH}")

	print(f"Transaction ID: {transaction_id}")
	print(f"Prediction: {prediction}")
	print(f"Fraud probability: {probability:.4f}")
	if alert_id is not None:
		print(f"Fraud alert created with ID: {alert_id}")


if __name__ == "__main__":
	arguments = parse_args()
	run_workflow(force_train=arguments.force_train)
