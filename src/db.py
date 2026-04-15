from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"


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
			(transaction_id, datetime.utcnow().isoformat(timespec="seconds"), status),
		)
		return int(cursor.lastrowid)
