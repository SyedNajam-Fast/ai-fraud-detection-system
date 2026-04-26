from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Dict, Iterator, Optional, Sequence

from src.core.config import DATABASE_PATH, SCHEMA_PATH


EXPECTED_TABLES = (
	"dataset_profiles",
	"feature_profiles",
	"fraud_alerts",
	"kaggle_transactions",
	"model_candidate_metrics",
	"model_training_runs",
	"predictions",
	"raw_dataset_uploads",
	"transactions",
	"users",
)
KAGGLE_VALUE_COLUMNS = tuple(f"v{index}" for index in range(1, 29))
KAGGLE_INSERT_COLUMNS = (
	"time_seconds",
	"amount",
	*KAGGLE_VALUE_COLUMNS,
	"class_label",
	"source_file",
	"imported_at",
)
KAGGLE_INSERT_SQL = (
	"INSERT INTO kaggle_transactions "
	f"({', '.join(KAGGLE_INSERT_COLUMNS)}) "
	f"VALUES ({', '.join(['?'] * len(KAGGLE_INSERT_COLUMNS))})"
)
MODEL_TRAINING_RUN_COLUMNS = (
	"dataset_source",
	"selection_metric",
	"selected_model_name",
	"selected_threshold",
	"sample_count",
	"train_count",
	"validation_count",
	"test_count",
	"train_f1",
	"validation_f1",
	"test_f1",
	"validation_average_precision",
	"test_average_precision",
	"overfit_flag",
	"underfit_flag",
	"status",
	"started_at",
	"finished_at",
	"notes",
)
MODEL_TRAINING_RUN_SQL = (
	"INSERT INTO model_training_runs "
	f"({', '.join(MODEL_TRAINING_RUN_COLUMNS)}) "
	f"VALUES ({', '.join(['?'] * len(MODEL_TRAINING_RUN_COLUMNS))})"
)
MODEL_CANDIDATE_COLUMNS = (
	"run_id",
	"model_name",
	"cv_f1_mean",
	"cv_f1_std",
	"train_precision",
	"train_recall",
	"train_f1",
	"validation_precision",
	"validation_recall",
	"validation_f1",
	"validation_average_precision",
	"train_average_precision",
	"validation_threshold",
	"fit_gap",
	"overfit_flag",
	"underfit_flag",
	"confusion_matrix_json",
	"selected",
)
MODEL_CANDIDATE_SQL = (
	"INSERT INTO model_candidate_metrics "
	f"({', '.join(MODEL_CANDIDATE_COLUMNS)}) "
	f"VALUES ({', '.join(['?'] * len(MODEL_CANDIDATE_COLUMNS))})"
)
RAW_DATASET_UPLOAD_COLUMNS = (
	"filename",
	"source_path",
	"file_size_bytes",
	"row_count",
	"column_count",
	"target_column",
	"status",
	"created_at",
)
RAW_DATASET_UPLOAD_SQL = (
	"INSERT INTO raw_dataset_uploads "
	f"({', '.join(RAW_DATASET_UPLOAD_COLUMNS)}) "
	f"VALUES ({', '.join(['?'] * len(RAW_DATASET_UPLOAD_COLUMNS))})"
)
DATASET_PROFILE_COLUMNS = (
	"upload_id",
	"row_count",
	"column_count",
	"duplicate_row_count",
	"missing_cell_count",
	"numeric_column_count",
	"categorical_column_count",
	"datetime_column_count",
	"target_column",
	"target_cardinality",
	"class_imbalance_ratio",
	"warnings_json",
	"summary_json",
	"created_at",
)
DATASET_PROFILE_SQL = (
	"INSERT INTO dataset_profiles "
	f"({', '.join(DATASET_PROFILE_COLUMNS)}) "
	f"VALUES ({', '.join(['?'] * len(DATASET_PROFILE_COLUMNS))})"
)
FEATURE_PROFILE_COLUMNS = (
	"upload_id",
	"dataset_profile_id",
	"column_name",
	"inferred_role",
	"inferred_dtype",
	"pandas_dtype",
	"non_null_count",
	"missing_count",
	"missing_ratio",
	"unique_count",
	"unique_ratio",
	"sample_values_json",
	"min_value",
	"max_value",
	"mean_value",
	"std_value",
	"simple_description",
	"technical_description",
	"target_candidate",
	"created_at",
)
FEATURE_PROFILE_SQL = (
	"INSERT INTO feature_profiles "
	f"({', '.join(FEATURE_PROFILE_COLUMNS)}) "
	f"VALUES ({', '.join(['?'] * len(FEATURE_PROFILE_COLUMNS))})"
)


def _ensure_database_directory() -> None:
	DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
	_ensure_database_directory()
	connection = sqlite3.connect(DATABASE_PATH)
	connection.row_factory = sqlite3.Row
	connection.execute("PRAGMA foreign_keys = ON")
	try:
		yield connection
		connection.commit()
	finally:
		connection.close()


def initialize_database() -> None:
	with get_connection() as connection:
		with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
			connection.executescript(schema_file.read())


def clear_kaggle_transactions() -> int:
	with get_connection() as connection:
		cursor = connection.execute("DELETE FROM kaggle_transactions")
		if cursor.rowcount is None or cursor.rowcount < 0:
			return 0
		return int(cursor.rowcount)


def insert_kaggle_transaction_rows(rows: Sequence[tuple[Any, ...]]) -> int:
	if not rows:
		return 0

	expected_length = len(KAGGLE_INSERT_COLUMNS)
	for row in rows:
		if len(row) != expected_length:
			raise ValueError(
				f"Each kaggle row must have {expected_length} values, got {len(row)}."
			)

	with get_connection() as connection:
		connection.executemany(KAGGLE_INSERT_SQL, rows)
	return len(rows)


def count_kaggle_transactions() -> int:
	with get_connection() as connection:
		cursor = connection.execute("SELECT COUNT(*) AS row_count FROM kaggle_transactions")
		row = cursor.fetchone()
		return int(row["row_count"]) if row else 0


def get_existing_tables() -> list[str]:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			SELECT name
			FROM sqlite_master
			WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
			ORDER BY name
			"""
		)
		return [str(row["name"]) for row in cursor.fetchall()]


def get_table_row_count(table_name: str) -> int:
	if table_name not in EXPECTED_TABLES:
		raise ValueError(f"Unsupported table name for count lookup: {table_name}")

	with get_connection() as connection:
		cursor = connection.execute(f"SELECT COUNT(*) AS row_count FROM {table_name}")
		row = cursor.fetchone()
		return int(row["row_count"]) if row else 0


def get_kaggle_label_distribution() -> Dict[int, int]:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			SELECT class_label, COUNT(*) AS sample_count
			FROM kaggle_transactions
			GROUP BY class_label
			ORDER BY class_label
			"""
		)
		return {
			int(row["class_label"]): int(row["sample_count"])
			for row in cursor.fetchall()
		}


def insert_model_training_run(run_data: Dict[str, Any]) -> int:
	values = [run_data[column] for column in MODEL_TRAINING_RUN_COLUMNS]
	with get_connection() as connection:
		cursor = connection.execute(MODEL_TRAINING_RUN_SQL, values)
		return int(cursor.lastrowid)


def insert_model_candidate_metric(metric_data: Dict[str, Any]) -> int:
	values = [metric_data[column] for column in MODEL_CANDIDATE_COLUMNS]
	with get_connection() as connection:
		cursor = connection.execute(MODEL_CANDIDATE_SQL, values)
		return int(cursor.lastrowid)


def insert_raw_dataset_upload(upload_data: Dict[str, Any]) -> int:
	values = [upload_data[column] for column in RAW_DATASET_UPLOAD_COLUMNS]
	with get_connection() as connection:
		cursor = connection.execute(RAW_DATASET_UPLOAD_SQL, values)
		return int(cursor.lastrowid)


def insert_dataset_profile(profile_data: Dict[str, Any]) -> int:
	values = [profile_data[column] for column in DATASET_PROFILE_COLUMNS]
	with get_connection() as connection:
		cursor = connection.execute(DATASET_PROFILE_SQL, values)
		return int(cursor.lastrowid)


def insert_feature_profiles(rows: Sequence[Dict[str, Any]]) -> int:
	if not rows:
		return 0

	values = [[row[column] for column in FEATURE_PROFILE_COLUMNS] for row in rows]
	with get_connection() as connection:
		connection.executemany(FEATURE_PROFILE_SQL, values)
	return len(rows)


def get_latest_dataset_upload() -> Optional[Dict[str, Any]]:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			SELECT *
			FROM raw_dataset_uploads
			ORDER BY id DESC
			LIMIT 1
			"""
		)
		row = cursor.fetchone()
		return dict(row) if row else None


def get_dataset_profile_by_upload_id(upload_id: int) -> Optional[Dict[str, Any]]:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			SELECT *
			FROM dataset_profiles
			WHERE upload_id = ?
			ORDER BY id DESC
			LIMIT 1
			""",
			(upload_id,),
		)
		row = cursor.fetchone()
		return dict(row) if row else None


def get_feature_profiles_by_upload_id(upload_id: int) -> list[Dict[str, Any]]:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			SELECT *
			FROM feature_profiles
			WHERE upload_id = ?
			ORDER BY id
			""",
			(upload_id,),
		)
		return [dict(row) for row in cursor.fetchall()]


def get_table_columns(table_name: str) -> list[Dict[str, Any]]:
	if table_name not in EXPECTED_TABLES:
		raise ValueError(f"Unsupported table name for schema lookup: {table_name}")

	with get_connection() as connection:
		cursor = connection.execute(f"PRAGMA table_info({table_name})")
		return [dict(row) for row in cursor.fetchall()]


def get_foreign_keys(table_name: str) -> list[Dict[str, Any]]:
	if table_name not in EXPECTED_TABLES:
		raise ValueError(f"Unsupported table name for foreign-key lookup: {table_name}")

	with get_connection() as connection:
		cursor = connection.execute(f"PRAGMA foreign_key_list({table_name})")
		return [dict(row) for row in cursor.fetchall()]


def get_indexes(table_name: str) -> list[Dict[str, Any]]:
	if table_name not in EXPECTED_TABLES:
		raise ValueError(f"Unsupported table name for index lookup: {table_name}")

	with get_connection() as connection:
		cursor = connection.execute(f"PRAGMA index_list({table_name})")
		return [dict(row) for row in cursor.fetchall()]


def get_or_create_user(name: str, email: str, card_number: str) -> int:
	with get_connection() as connection:
		cursor = connection.execute("SELECT id FROM users WHERE email = ?", (email,))
		row = cursor.fetchone()
		if row:
			return int(row["id"])

		cursor = connection.execute(
			"INSERT INTO users (name, email, card_number) VALUES (?, ?, ?)",
			(name, email, card_number),
		)
		return int(cursor.lastrowid)


def insert_transaction(user_id: int, amount: float, time: int, location: str, merchant: str) -> int:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			INSERT INTO transactions (user_id, amount, time, location, merchant)
			VALUES (?, ?, ?, ?, ?)
			""",
			(user_id, amount, time, location, merchant),
		)
		return int(cursor.lastrowid)


def fetch_transaction(transaction_id: int) -> Optional[Dict[str, Any]]:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			SELECT
				t.id,
				t.user_id,
				t.amount,
				t.time,
				t.location,
				t.merchant,
				u.name AS user_name,
				u.email AS user_email,
				u.card_number
			FROM transactions AS t
			JOIN users AS u ON u.id = t.user_id
			WHERE t.id = ?
			""",
			(transaction_id,),
		)
		row = cursor.fetchone()
		return dict(row) if row else None


def store_prediction(transaction_id: int, prediction: int, probability: float) -> int:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			INSERT INTO predictions (transaction_id, prediction, probability)
			VALUES (?, ?, ?)
			""",
			(transaction_id, prediction, probability),
		)
		return int(cursor.lastrowid)


def create_fraud_alert(transaction_id: int, status: str = "open") -> int:
	with get_connection() as connection:
		cursor = connection.execute(
			"""
			INSERT INTO fraud_alerts (transaction_id, alert_time, status)
			VALUES (?, ?, ?)
			""",
			(transaction_id, datetime.now(UTC).isoformat(timespec="seconds"), status),
		)
		return int(cursor.lastrowid)
