from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
REQUIRED_PACKAGES = ["pandas", "numpy", "sklearn", "joblib", "kagglehub"]


def _ensure_project_root_on_path() -> None:
	if str(PROJECT_ROOT) not in sys.path:
		sys.path.insert(0, str(PROJECT_ROOT))


def _missing_packages() -> list[str]:
	missing = []
	for package_name in REQUIRED_PACKAGES:
		if importlib.util.find_spec(package_name) is None:
			missing.append(package_name)
	return missing


def _install_requirements_if_needed() -> None:
	missing_packages = _missing_packages()
	if not missing_packages:
		return

	print("Missing dependencies detected. Installing requirements from requirements.txt...")
	install_command = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
	result = subprocess.run(install_command, check=False)
	if result.returncode != 0:
		raise SystemExit(f"Dependency installation failed with exit code {result.returncode}.")

	remaining_missing = _missing_packages()
	if remaining_missing:
		raise SystemExit(
			"Dependency installation completed, but these packages are still missing: "
			+ ", ".join(remaining_missing)
		)


def _load_runtime_modules():
	_ensure_project_root_on_path()

	from model.train_model import MODEL_METADATA_PATH, MODEL_PATH, train_and_save_model
	from src.db import create_fraud_alert, fetch_transaction, initialize_database, store_prediction
	from src.insert_data import insert_sample_transaction
	from src.predict import load_model_metadata, predict_transaction

	return {
		"MODEL_METADATA_PATH": MODEL_METADATA_PATH,
		"MODEL_PATH": MODEL_PATH,
		"train_and_save_model": train_and_save_model,
		"create_fraud_alert": create_fraud_alert,
		"fetch_transaction": fetch_transaction,
		"initialize_database": initialize_database,
		"store_prediction": store_prediction,
		"insert_sample_transaction": insert_sample_transaction,
		"load_model_metadata": load_model_metadata,
		"predict_transaction": predict_transaction,
	}


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
	modules = _load_runtime_modules()
	model_path = modules["MODEL_PATH"]
	model_metadata_path = modules["MODEL_METADATA_PATH"]
	train_and_save_model = modules["train_and_save_model"]

	if not force_train and model_path.exists() and model_metadata_path.exists():
		return None
	return train_and_save_model()


def run_workflow(force_train: bool = False) -> None:
	_install_requirements_if_needed()
	modules = _load_runtime_modules()

	initialize_database = modules["initialize_database"]
	insert_sample_transaction = modules["insert_sample_transaction"]
	fetch_transaction = modules["fetch_transaction"]
	predict_transaction = modules["predict_transaction"]
	store_prediction = modules["store_prediction"]
	create_fraud_alert = modules["create_fraud_alert"]
	model_path = modules["MODEL_PATH"]
	load_model_metadata = modules["load_model_metadata"]

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
		print(f"Dataset source: {model_metrics['dataset_source']}")
		print(f"Samples used: {model_metrics['sample_count']}")
		print(f"Selected model: {model_metrics['selected_model_name']}")
		print(f"Selection metric: {model_metrics['selection_metric']}")
		print(f"Decision threshold: {model_metrics['selected_threshold']:.2f}")
		print(f"Validation F1: {model_metrics['validation_metrics']['f1']:.4f}")
		print(f"Test F1: {model_metrics['test_metrics']['f1']:.4f}")
		print(f"Overfit check: {'flagged' if model_metrics['overfit_flag'] else 'passed'}")
		print(f"Underfit check: {'flagged' if model_metrics['underfit_flag'] else 'passed'}")
		print(f"Accuracy: {model_metrics['accuracy']:.4f}")
		print("Confusion matrix:")
		print(model_metrics["confusion_matrix"])
	else:
		print(f"Using existing trained model: {model_path}")
		metadata = load_model_metadata()
		if metadata:
			print(f"Selected model: {metadata.get('selected_model_name', 'unknown')}")
			print(f"Decision threshold: {float(metadata.get('selected_threshold', 0.5)):.2f}")

	print(f"Transaction ID: {transaction_id}")
	print(f"Prediction: {prediction}")
	print(f"Fraud probability: {probability:.4f}")
	if alert_id is not None:
		print(f"Fraud alert created with ID: {alert_id}")


if __name__ == "__main__":
	arguments = parse_args()
	run_workflow(force_train=arguments.force_train)
