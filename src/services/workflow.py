from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import MODEL_METADATA_PATH, MODEL_PATH, ensure_project_root_on_path


@dataclass
class WorkflowResult:
    transaction_id: int
    prediction: int
    probability: float
    alert_id: int | None
    model_metrics: dict[str, Any] | None
    model_path: Path
    model_metadata_path: Path


def _load_runtime_modules() -> dict[str, Any]:
    ensure_project_root_on_path()

    from model.train_model import train_and_save_model
    from src.db import create_fraud_alert, fetch_transaction, initialize_database, store_prediction
    from src.insert_data import insert_sample_transaction
    from src.predict import predict_transaction

    return {
        "train_and_save_model": train_and_save_model,
        "create_fraud_alert": create_fraud_alert,
        "fetch_transaction": fetch_transaction,
        "initialize_database": initialize_database,
        "store_prediction": store_prediction,
        "insert_sample_transaction": insert_sample_transaction,
        "predict_transaction": predict_transaction,
    }


def ensure_model_available(force_train: bool = False) -> dict[str, Any] | None:
    modules = _load_runtime_modules()
    train_and_save_model = modules["train_and_save_model"]

    if not force_train and MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
        return None
    return train_and_save_model()


def run_workflow(force_train: bool = False) -> WorkflowResult:
    modules = _load_runtime_modules()

    initialize_database = modules["initialize_database"]
    insert_sample_transaction = modules["insert_sample_transaction"]
    fetch_transaction = modules["fetch_transaction"]
    predict_transaction = modules["predict_transaction"]
    store_prediction = modules["store_prediction"]
    create_fraud_alert = modules["create_fraud_alert"]

    initialize_database()
    model_metrics = ensure_model_available(force_train=force_train)

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

    return WorkflowResult(
        transaction_id=transaction_id,
        prediction=prediction,
        probability=probability,
        alert_id=alert_id,
        model_metrics=model_metrics,
        model_path=MODEL_PATH,
        model_metadata_path=MODEL_METADATA_PATH,
    )
