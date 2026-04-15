from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.db import (  # noqa: E402
	clear_kaggle_transactions,
	count_kaggle_transactions,
	get_kaggle_label_distribution,
	initialize_database,
	insert_kaggle_transaction_rows,
)

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "creditcardfraud" / "creditcard.csv"
REQUIRED_COLUMNS = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)], "Class"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Import Kaggle credit card fraud dataset into SQLite.")
	parser.add_argument(
		"--csv-path",
		default=str(DEFAULT_CSV_PATH),
		help="Path to creditcard.csv downloaded from Kaggle.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=5000,
		help="Number of rows to insert per batch.",
	)
	parser.add_argument(
		"--append",
		action="store_true",
		help="Append to existing kaggle_transactions rows instead of clearing first.",
	)
	return parser.parse_args()


def _validate_csv_header(csv_path: Path) -> None:
	header = pd.read_csv(csv_path, nrows=0)
	missing_columns = [column for column in REQUIRED_COLUMNS if column not in header.columns]
	if missing_columns:
		raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")


def _build_rows(chunk: pd.DataFrame, source_file: str, imported_at: str) -> list[tuple[object, ...]]:
	chunk = chunk.copy()
	chunk["Class"] = chunk["Class"].astype(int)

	if not chunk["Class"].isin([0, 1]).all():
		raise ValueError("Class column must contain only 0 or 1 values.")

	rows: list[tuple[object, ...]] = []
	for record in chunk.itertuples(index=False, name=None):
		time_seconds = int(record[0])
		amount = float(record[1])
		v_values = [float(value) for value in record[2:30]]
		class_label = int(record[30])
		rows.append((time_seconds, amount, *v_values, class_label, source_file, imported_at))
	return rows


def import_kaggle_csv(csv_path: Path, batch_size: int, append: bool) -> None:
	if batch_size <= 0:
		raise ValueError("batch_size must be greater than zero.")
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	initialize_database()
	_validate_csv_header(csv_path)

	if not append:
		cleared_rows = clear_kaggle_transactions()
		print(f"Cleared existing kaggle_transactions rows: {cleared_rows}")

	total_inserted = 0
	source_file = csv_path.name
	for chunk_index, chunk in enumerate(
		pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, chunksize=batch_size),
		start=1,
	):
		imported_at = datetime.now(UTC).isoformat(timespec="seconds")
		rows = _build_rows(chunk=chunk, source_file=source_file, imported_at=imported_at)
		inserted = insert_kaggle_transaction_rows(rows)
		total_inserted += inserted

		if chunk_index == 1 or chunk_index % 10 == 0:
			print(f"Chunk {chunk_index}: inserted {inserted} rows (running total: {total_inserted})")

	label_distribution = get_kaggle_label_distribution()
	print("Kaggle import completed.")
	print(f"Rows imported this run: {total_inserted}")
	print(f"Total rows in kaggle_transactions: {count_kaggle_transactions()}")
	print(f"Class distribution: {label_distribution}")


def main() -> None:
	args = parse_args()
	try:
		import_kaggle_csv(
			csv_path=Path(args.csv_path),
			batch_size=args.batch_size,
			append=args.append,
		)
	except Exception as error:
		print(f"Kaggle DB import failed: {error}", file=sys.stderr)
		raise SystemExit(1) from error


if __name__ == "__main__":
	main()
