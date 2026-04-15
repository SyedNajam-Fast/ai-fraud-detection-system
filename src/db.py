from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"
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
