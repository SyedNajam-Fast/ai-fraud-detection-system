from __future__ import annotations

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
from src.predict import predict_transaction  # noqa: E402


def run_workflow() -> None:
	initialize_database()
	metrics = train_and_save_model()
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

	print("Model training metrics:")
	print(f"Accuracy: {metrics['accuracy']:.4f}")
	print("Confusion matrix:")
	print(metrics["confusion_matrix"])
	print(f"Transaction ID: {transaction_id}")
	print(f"Prediction: {prediction}")
	print(f"Fraud probability: {probability:.4f}")
	if alert_id is not None:
		print(f"Fraud alert created with ID: {alert_id}")


if __name__ == "__main__":
	run_workflow()
