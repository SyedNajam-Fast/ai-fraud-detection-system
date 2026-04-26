from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import sys
from datetime import UTC, datetime
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
	AdaBoostClassifier,
	ExtraTreesClassifier,
	GradientBoostingClassifier,
	HistGradientBoostingClassifier,
	RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	precision_recall_curve,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.core.config import (
	DATABASE_PATH,
	FALLBACK_DATASET_PATH,
	MODEL_METADATA_PATH,
	MODEL_PATH,
	MODEL_TRAINING_N_JOBS,
	ensure_project_root_on_path,
)
from src.services.model_recommendation import (
	MODEL_DISPLAY_NAMES,
	build_model_recommendation_summary,
)


DATA_PATH = FALLBACK_DATASET_PATH


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


def _map_kaggle_to_project_schema(kaggle_dataset: pd.DataFrame) -> pd.DataFrame:
	location_score = kaggle_dataset["v1"] + 0.5 * kaggle_dataset["v2"] - 0.25 * kaggle_dataset["v3"]
	merchant_score = kaggle_dataset["v4"] - kaggle_dataset["v5"]

	location_values = np.select(
		[
			location_score < -2.0,
			location_score < -0.5,
			location_score < 0.75,
			location_score < 2.0,
		],
		["Online", "Karachi", "Lahore", "Islamabad"],
		default="Rawalpindi",
	)

	merchant_values = np.select(
		[
			merchant_score < -1.2,
			merchant_score < -0.2,
			merchant_score < 0.6,
			merchant_score < 1.5,
		],
		["travel", "restaurant", "grocery_store", "online_retail"],
		default="electronics_store",
	)

	return pd.DataFrame(
		{
			"amount": kaggle_dataset["amount"].astype(float),
			"time": ((kaggle_dataset["time_seconds"] // 3600) % 24).astype(int),
			"location": location_values,
			"merchant": merchant_values,
			"fraud": kaggle_dataset["class_label"].astype(int),
		}
	)


def _load_dataset_from_database() -> pd.DataFrame | None:
	if not DATABASE_PATH.exists():
		return None

	query = """
	SELECT
		time_seconds,
		amount,
		v1,
		v2,
		v3,
		v4,
		v5,
		class_label
	FROM kaggle_transactions
	"""

	try:
		with sqlite3.connect(DATABASE_PATH) as connection:
			kaggle_dataset = pd.read_sql_query(query, connection)
	except (sqlite3.Error, pd.errors.DatabaseError):
		return None

	if kaggle_dataset.empty or kaggle_dataset["class_label"].nunique() < 2:
		return None

	return _map_kaggle_to_project_schema(kaggle_dataset)


def load_training_dataset() -> tuple[pd.DataFrame, str]:
	database_dataset = _load_dataset_from_database()
	if database_dataset is not None:
		return database_dataset, "database:kaggle_transactions"

	if DATA_PATH.exists():
		dataset = pd.read_csv(DATA_PATH)
		expected_columns = ["amount", "time", "location", "merchant", "fraud"]
		if set(expected_columns).issubset(dataset.columns):
			return dataset[expected_columns], f"csv:{DATA_PATH.name}"

	return _generate_synthetic_dataset(), "synthetic:generated"


def _build_preprocessor() -> ColumnTransformer:
	numeric_features = ["amount", "time"]
	categorical_features = ["location", "merchant"]

	return ColumnTransformer(
		transformers=[
			(
				"numeric",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="median")),
						("scaler", StandardScaler()),
					]
				),
				numeric_features,
			),
			(
				"categorical",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
					]
				),
				categorical_features,
			),
		],
		sparse_threshold=0.0,
	)


def _build_classifier(model_name: str):
	if model_name == "logistic_regression":
		return LogisticRegression(
			max_iter=2000,
			class_weight="balanced",
			solver="liblinear",
			random_state=42,
		)
	if model_name == "decision_tree":
		return DecisionTreeClassifier(
			max_depth=8,
			min_samples_leaf=5,
			class_weight="balanced",
			random_state=42,
		)
	if model_name == "random_forest":
		return RandomForestClassifier(
			n_estimators=120,
			max_depth=12,
			class_weight="balanced",
			random_state=42,
			n_jobs=MODEL_TRAINING_N_JOBS,
		)
	if model_name == "extra_trees":
		return ExtraTreesClassifier(
			n_estimators=180,
			max_depth=16,
			class_weight="balanced",
			random_state=42,
			n_jobs=MODEL_TRAINING_N_JOBS,
		)
	if model_name == "gradient_boosting":
		return GradientBoostingClassifier(
			n_estimators=120,
			learning_rate=0.08,
			max_depth=3,
			random_state=42,
		)
	if model_name == "hist_gradient_boosting":
		return HistGradientBoostingClassifier(
			max_depth=8,
			learning_rate=0.08,
			max_iter=180,
			random_state=42,
		)
	if model_name == "adaboost":
		return AdaBoostClassifier(
			n_estimators=140,
			learning_rate=0.7,
			random_state=42,
		)
	if model_name == "svm":
		return SVC(
			C=1.0,
			kernel="rbf",
			gamma="scale",
			class_weight="balanced",
			probability=True,
			random_state=42,
		)
	if model_name == "knn":
		return KNeighborsClassifier(
			n_neighbors=9,
			weights="distance",
		)
	if model_name == "naive_bayes":
		return GaussianNB()
	raise ValueError(f"Unsupported model name: {model_name}")


def _build_pipeline(model_name: str) -> Pipeline:
	return Pipeline(
		steps=[
			("preprocessor", _build_preprocessor()),
			("classifier", _build_classifier(model_name)),
		]
	)


def _threshold_from_probabilities(target: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
	precision, recall, thresholds = precision_recall_curve(target, probabilities)
	if len(thresholds) == 0:
		return 0.5, 0.0

	precision = precision[:-1]
	recall = recall[:-1]
	f1_scores = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
	best_index = int(np.nanargmax(f1_scores))
	return float(thresholds[best_index]), float(f1_scores[best_index])


def _classification_metrics(target: pd.Series, probabilities: np.ndarray, threshold: float) -> Dict[str, object]:
	predictions = (probabilities >= threshold).astype(int)
	return {
		"accuracy": float(accuracy_score(target, predictions)),
		"precision": float(precision_score(target, predictions, zero_division=0)),
		"recall": float(recall_score(target, predictions, zero_division=0)),
		"f1": float(f1_score(target, predictions, zero_division=0)),
		"average_precision": float(average_precision_score(target, probabilities)),
		"roc_auc": float(roc_auc_score(target, probabilities)),
		"confusion_matrix": confusion_matrix(target, predictions).tolist(),
	}


def _fit_and_score_candidate(
	model_name: str,
	pipeline: Pipeline,
	x_train: pd.DataFrame,
	y_train: pd.Series,
	x_validation: pd.DataFrame,
	y_validation: pd.Series,
) -> Dict[str, object]:
	pipeline.fit(x_train, y_train)

	train_probabilities = pipeline.predict_proba(x_train)[:, 1]
	validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
	selected_threshold, threshold_f1 = _threshold_from_probabilities(y_validation, validation_probabilities)

	train_metrics = _classification_metrics(y_train, train_probabilities, selected_threshold)
	validation_metrics = _classification_metrics(y_validation, validation_probabilities, selected_threshold)
	fit_gap = float(train_metrics["f1"] - validation_metrics["f1"])

	overfit_flag = int(
		fit_gap > 0.12
		or (train_metrics["average_precision"] - validation_metrics["average_precision"] > 0.15)
	)
	underfit_flag = int(validation_metrics["f1"] < 0.55 and validation_metrics["average_precision"] < 0.65)

	return {
		"model_name": model_name,
		"display_name": MODEL_DISPLAY_NAMES[model_name],
		"pipeline": pipeline,
		"selected_threshold": selected_threshold,
		"threshold_f1": threshold_f1,
		"train_metrics": train_metrics,
		"validation_metrics": validation_metrics,
		"fit_gap": fit_gap,
		"overfit_flag": overfit_flag,
		"underfit_flag": underfit_flag,
	}


def _summarize_candidate_result(candidate: Dict[str, object]) -> Dict[str, object]:
	train_metrics = candidate["train_metrics"]
	validation_metrics = candidate["validation_metrics"]
	return {
		"model_name": candidate["model_name"],
		"display_name": candidate["display_name"],
		"selected_threshold": float(candidate["selected_threshold"]),
		"threshold_f1": float(candidate["threshold_f1"]),
		"train_metrics": train_metrics,
		"validation_metrics": validation_metrics,
		"fit_gap": float(candidate["fit_gap"]),
		"overfit_flag": bool(candidate["overfit_flag"]),
		"underfit_flag": bool(candidate["underfit_flag"]),
	}


def recommend_models_for_current_dataset() -> Dict[str, Any]:
	dataset, dataset_source = load_training_dataset()
	features = dataset[["amount", "time", "location", "merchant"]]
	target = dataset["fraud"]
	recommendation_summary = build_model_recommendation_summary(features, target, dataset_source)
	return {
		"dataset_source": dataset_source,
		"recommendation_summary": recommendation_summary,
	}


def train_and_save_model() -> Dict[str, object]:
	ensure_project_root_on_path()
	dataset, dataset_source = load_training_dataset()
	features = dataset[["amount", "time", "location", "merchant"]]
	target = dataset["fraud"]
	started_at = datetime.now(UTC).isoformat(timespec="seconds")

	if target.nunique() < 2:
		raise ValueError("Training dataset must include both fraud and non-fraud labels.")

	recommendation_summary = build_model_recommendation_summary(features, target, dataset_source)
	shortlisted_model_names = [item.model_name for item in recommendation_summary.shortlisted_models]

	x_train_full, x_test, y_train_full, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=42,
		stratify=target,
	)
	x_train, x_validation, y_train, y_validation = train_test_split(
		x_train_full,
		y_train_full,
		test_size=0.25,
		random_state=42,
		stratify=y_train_full,
	)

	candidate_results = []
	for model_name in shortlisted_model_names:
		candidate_results.append(
			_fit_and_score_candidate(
				model_name=model_name,
				pipeline=_build_pipeline(model_name),
				x_train=x_train,
				y_train=y_train,
				x_validation=x_validation,
				y_validation=y_validation,
			)
		)

	selected_candidate = max(
		candidate_results,
		key=lambda item: (
			float(item["validation_metrics"]["average_precision"]),
			float(item["validation_metrics"]["f1"]),
			float(item["validation_metrics"]["recall"]),
		),
	)

	selected_model_name = str(selected_candidate["model_name"])
	selected_threshold = float(selected_candidate["selected_threshold"])
	selection_metric = "validation_average_precision"

	final_pipeline = _build_pipeline(selected_model_name)
	final_features = pd.concat([x_train, x_validation], axis=0)
	final_target = pd.concat([y_train, y_validation], axis=0)
	final_pipeline.fit(final_features, final_target)

	test_probabilities = final_pipeline.predict_proba(x_test)[:, 1]
	test_metrics = _classification_metrics(y_test, test_probabilities, selected_threshold)

	selected_train_metrics = selected_candidate["train_metrics"]
	selected_validation_metrics = selected_candidate["validation_metrics"]
	overfit_flag = int(selected_candidate["overfit_flag"])
	underfit_flag = int(selected_candidate["underfit_flag"])
	finished_at = datetime.now(UTC).isoformat(timespec="seconds")
	status = "completed"
	notes = (
		"Selected by validation_average_precision after rule-based shortlist; "
		f"shortlisted={', '.join(shortlisted_model_names)}; "
		f"winner={selected_model_name}; "
		f"val_f1={selected_validation_metrics['f1']:.4f}; "
		f"test_f1={test_metrics['f1']:.4f}"
	)

	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(final_pipeline, MODEL_PATH)

	metadata = {
		"dataset_source": dataset_source,
		"selected_model_name": selected_model_name,
		"selected_model_display_name": MODEL_DISPLAY_NAMES[selected_model_name],
		"selection_metric": selection_metric,
		"selected_threshold": selected_threshold,
		"sample_count": int(len(dataset)),
		"train_count": int(len(x_train)),
		"validation_count": int(len(x_validation)),
		"test_count": int(len(x_test)),
		"train_metrics": selected_train_metrics,
		"validation_metrics": selected_validation_metrics,
		"test_metrics": test_metrics,
		"overfit_flag": bool(overfit_flag),
		"underfit_flag": bool(underfit_flag),
		"created_at": finished_at,
		"shortlisted_models": [
			{
				"model_name": item.model_name,
				"display_name": item.display_name,
				"score": item.score,
				"shortlist_rank": item.shortlist_rank,
				"rationale": item.rationale,
			}
			for item in recommendation_summary.shortlisted_models
		],
		"recommendation_strategy": recommendation_summary.recommendation_strategy,
		"dataset_characteristics": recommendation_summary.dataset_characteristics,
		"full_model_pool": [
			{
				"model_name": item.model_name,
				"display_name": item.display_name,
				"score": item.score,
				"shortlisted": item.shortlisted,
				"shortlist_rank": item.shortlist_rank,
				"rationale": item.rationale,
			}
			for item in recommendation_summary.all_models
		],
		"candidates": [_summarize_candidate_result(candidate) for candidate in candidate_results],
	}
	MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

	try:
		from src.db import (
			insert_model_candidate_metric,
			insert_model_recommendations,
			insert_model_training_run,
		)

		run_id = insert_model_training_run(
			{
				"dataset_source": dataset_source,
				"selection_metric": selection_metric,
				"selected_model_name": selected_model_name,
				"selected_threshold": selected_threshold,
				"sample_count": int(len(dataset)),
				"train_count": int(len(x_train)),
				"validation_count": int(len(x_validation)),
				"test_count": int(len(x_test)),
				"train_f1": float(selected_train_metrics["f1"]),
				"validation_f1": float(selected_validation_metrics["f1"]),
				"test_f1": float(test_metrics["f1"]),
				"validation_average_precision": float(selected_validation_metrics["average_precision"]),
				"test_average_precision": float(test_metrics["average_precision"]),
				"overfit_flag": overfit_flag,
				"underfit_flag": underfit_flag,
				"status": status,
				"started_at": started_at,
				"finished_at": finished_at,
				"notes": notes,
			},
		)

		insert_model_recommendations(
			[
				{
					"run_id": run_id,
					"model_name": item.model_name,
					"recommendation_rank": int(item.shortlist_rank or 0),
					"recommendation_score": float(item.score),
					"rationale_text": item.rationale,
					"selected_for_training": 1,
					"final_winner": 1 if item.model_name == selected_model_name else 0,
					"created_at": finished_at,
				}
				for item in recommendation_summary.shortlisted_models
			]
		)

		for candidate in candidate_results:
			validation_metrics = candidate["validation_metrics"]
			train_metrics = candidate["train_metrics"]
			insert_model_candidate_metric(
				{
					"run_id": run_id,
					"model_name": candidate["model_name"],
					"cv_f1_mean": None,
					"cv_f1_std": None,
					"train_precision": float(train_metrics["precision"]),
					"train_recall": float(train_metrics["recall"]),
					"train_f1": float(train_metrics["f1"]),
					"validation_precision": float(validation_metrics["precision"]),
					"validation_recall": float(validation_metrics["recall"]),
					"validation_f1": float(validation_metrics["f1"]),
					"validation_average_precision": float(validation_metrics["average_precision"]),
					"train_average_precision": float(train_metrics["average_precision"]),
					"validation_threshold": float(candidate["selected_threshold"]),
					"fit_gap": float(candidate["fit_gap"]),
					"overfit_flag": int(candidate["overfit_flag"]),
					"underfit_flag": int(candidate["underfit_flag"]),
					"confusion_matrix_json": json.dumps(validation_metrics["confusion_matrix"]),
					"selected": 1 if candidate["model_name"] == selected_model_name else 0,
				},
			)
	except Exception:
		pass

	return {
		"accuracy": float(test_metrics["accuracy"]),
		"confusion_matrix": test_metrics["confusion_matrix"],
		"model_path": str(MODEL_PATH),
		"dataset_source": dataset_source,
		"sample_count": int(len(dataset)),
		"selected_model_name": selected_model_name,
		"selected_model_display_name": MODEL_DISPLAY_NAMES[selected_model_name],
		"selected_threshold": selected_threshold,
		"selection_metric": selection_metric,
		"train_metrics": selected_train_metrics,
		"validation_metrics": selected_validation_metrics,
		"test_metrics": test_metrics,
		"overfit_flag": bool(overfit_flag),
		"underfit_flag": bool(underfit_flag),
		"shortlisted_models": metadata["shortlisted_models"],
		"recommendation_strategy": recommendation_summary.recommendation_strategy,
		"dataset_characteristics": recommendation_summary.dataset_characteristics,
		"full_model_pool": metadata["full_model_pool"],
		"candidates": [_summarize_candidate_result(candidate) for candidate in candidate_results],
		"training_run_started_at": started_at,
		"training_run_finished_at": finished_at,
	}


if __name__ == "__main__":
	ensure_project_root_on_path()
	results = train_and_save_model()
	print(f"Saved model to {results['model_path']}")
	print(f"Dataset source: {results['dataset_source']}")
	print(f"Selected model: {results['selected_model_name']}")
	print(f"Selection metric: {results['selection_metric']}")
	print(f"Decision threshold: {results['selected_threshold']:.2f}")
	print(f"Samples used: {results['sample_count']}")
	print(f"Shortlisted models: {[item['model_name'] for item in results['shortlisted_models']]}")
	print(f"Overfit check: {'flagged' if results['overfit_flag'] else 'passed'}")
	print(f"Underfit check: {'flagged' if results['underfit_flag'] else 'passed'}")
	print(f"Accuracy: {results['accuracy']:.4f}")
	print(f"Validation F1: {results['validation_metrics']['f1']:.4f}")
	print(f"Validation PR AUC: {results['validation_metrics']['average_precision']:.4f}")
	print(f"Validation ROC AUC: {results['validation_metrics']['roc_auc']:.4f}")
	print(f"Test F1: {results['test_metrics']['f1']:.4f}")
	print("Confusion matrix:")
	print(results["confusion_matrix"])
