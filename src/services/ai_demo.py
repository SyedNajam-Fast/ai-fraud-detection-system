from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from model.train_model import load_training_dataset
from src.predict import get_prediction_threshold, load_model_metadata, predict_transaction


FEATURE_COLUMNS = ["amount", "time", "location", "merchant"]
TARGET_COLUMN = "fraud"
DEFAULT_MANUAL_INPUT = {
    "amount": 12450.75,
    "time": 23,
    "location": "Lahore",
    "merchant": "electronics_store",
}
FEATURE_EXPLANATIONS = {
    "amount": "Transaction amount used to measure unusual spending size.",
    "time": "Hour of the day from 0 to 23 to capture suspicious timing patterns.",
    "location": "Transaction location category to highlight geographic risk patterns.",
    "merchant": "Merchant type to compare normal shopping with higher-risk categories.",
    "fraud": "Target column where 1 means fraud and 0 means normal transaction.",
}


def _native_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _native_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        records.append({key: _native_value(value) for key, value in record.items()})
    return records


def _load_project_dataset() -> tuple[pd.DataFrame, str]:
    dataset, dataset_source = load_training_dataset()
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    if not required_columns.issubset(dataset.columns):
        missing = sorted(required_columns.difference(dataset.columns))
        raise ValueError(f"Training dataset is missing required columns: {', '.join(missing)}")
    return dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].copy(), dataset_source


def _risk_signals(payload: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    amount = float(payload["amount"])
    hour = int(payload["time"])
    location = str(payload["location"])
    merchant = str(payload["merchant"])

    if amount >= 500:
        signals.append("High transaction amount compared with normal spending.")
    if hour in {0, 1, 2, 3, 23}:
        signals.append("Transaction time falls in a late-night risk window.")
    if location in {"Online", "Rawalpindi"}:
        signals.append("Location belongs to a higher-risk bucket used in training.")
    if merchant in {"electronics_store", "online_retail", "travel"}:
        signals.append("Merchant category is associated with higher fraud exposure.")

    if not signals:
        signals.append("This input does not trigger any strong demo risk signal.")
    return signals


def _confidence_band(probability: float, threshold: float) -> str:
    margin = abs(probability - threshold)
    if margin >= 0.25:
        return "High confidence"
    if margin >= 0.10:
        return "Moderate confidence"
    return "Borderline case"


def _prediction_message(prediction: int, probability: float, threshold: float) -> str:
    confidence = _confidence_band(probability, threshold)
    if prediction == 1:
        return (
            f"The model flags this transaction as likely fraud because the risk score is above the "
            f"decision threshold. Confidence: {confidence}."
        )
    return (
        f"The model treats this transaction as normal because the risk score stays below the "
        f"decision threshold. Confidence: {confidence}."
    )


def build_ai_dataset_preview(sample_rows: int = 6) -> dict[str, Any]:
    dataset, dataset_source = _load_project_dataset()
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    location_options = sorted({str(value) for value in features["location"].dropna().tolist()})
    merchant_options = sorted({str(value) for value in features["merchant"].dropna().tolist()})

    sample_frame = dataset.head(sample_rows).copy()
    sample_frame["amount"] = sample_frame["amount"].astype(float).round(2)

    return {
        "dataset_source": dataset_source,
        "sample_count": int(len(dataset)),
        "target_column": TARGET_COLUMN,
        "fraud_rate": float(target.mean()),
        "normal_rate": float(1.0 - target.mean()),
        "feature_cards": [
            {
                "name": column,
                "kind": "target" if column == TARGET_COLUMN else "feature",
                "description": FEATURE_EXPLANATIONS[column],
            }
            for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        ],
        "sample_rows": _native_records(sample_frame),
        "manual_input_options": {
            "defaults": DEFAULT_MANUAL_INPUT,
            "location_options": location_options,
            "merchant_options": merchant_options,
        },
        "numeric_summary": {
            "amount": {
                "min": float(features["amount"].min()),
                "mean": float(features["amount"].mean()),
                "max": float(features["amount"].max()),
            },
            "time": {
                "min": int(features["time"].min()),
                "mean": float(features["time"].mean()),
                "max": int(features["time"].max()),
            },
        },
    }


def run_manual_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    clean_payload = {
        "amount": round(float(payload["amount"]), 2),
        "time": int(payload["time"]),
        "location": str(payload["location"]).strip(),
        "merchant": str(payload["merchant"]).strip(),
    }

    if clean_payload["amount"] < 0:
        raise ValueError("Amount must be zero or positive.")
    if clean_payload["time"] < 0 or clean_payload["time"] > 23:
        raise ValueError("Time must be between 0 and 23.")
    if not clean_payload["location"]:
        raise ValueError("Location is required.")
    if not clean_payload["merchant"]:
        raise ValueError("Merchant is required.")

    prediction, probability = predict_transaction(clean_payload)
    metadata = load_model_metadata()
    threshold = float(metadata.get("selected_threshold", get_prediction_threshold()))

    return {
        "input": clean_payload,
        "prediction": int(prediction),
        "prediction_label": "Fraud" if prediction == 1 else "Normal",
        "probability": float(probability),
        "threshold": threshold,
        "confidence_band": _confidence_band(float(probability), threshold),
        "risk_signals": _risk_signals(clean_payload),
        "message": _prediction_message(int(prediction), float(probability), threshold),
        "model_name": metadata.get("selected_model_display_name") or metadata.get("selected_model_name") or "Saved model",
    }


def predict_holdout_test_sample(index: int = 0) -> dict[str, Any]:
    dataset, dataset_source = _load_project_dataset()
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    if x_test.empty:
        raise ValueError("The held-out test split is empty.")

    resolved_index = index % len(x_test)
    row = x_test.iloc[resolved_index]
    actual_label = int(y_test.iloc[resolved_index])
    sample_payload = {
        "amount": round(float(row["amount"]), 2),
        "time": int(row["time"]),
        "location": str(row["location"]),
        "merchant": str(row["merchant"]),
    }

    prediction_result = run_manual_prediction(sample_payload)
    predicted_label = int(prediction_result["prediction"])
    correct = predicted_label == actual_label

    review_text = "Correct fraud detection on unseen data."
    if actual_label == 0 and predicted_label == 0:
        review_text = "Correctly identified as a normal transaction."
    elif actual_label == 0 and predicted_label == 1:
        review_text = "False positive: the model raised fraud on a normal transaction."
    elif actual_label == 1 and predicted_label == 0:
        review_text = "False negative: the model missed a fraudulent transaction."

    return {
        "sample_index": resolved_index,
        "next_index": (resolved_index + 1) % len(x_test),
        "total_test_samples": int(len(x_test)),
        "dataset_source": dataset_source,
        "input": prediction_result["input"],
        "prediction": predicted_label,
        "prediction_label": prediction_result["prediction_label"],
        "probability": prediction_result["probability"],
        "threshold": prediction_result["threshold"],
        "confidence_band": prediction_result["confidence_band"],
        "actual_label": actual_label,
        "actual_label_text": "Fraud" if actual_label == 1 else "Normal",
        "correct": correct,
        "review_text": review_text,
        "risk_signals": prediction_result["risk_signals"],
        "message": prediction_result["message"],
    }
