from __future__ import annotations

from pathlib import Path
import shutil
import sys

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "creditcardfraud"
DATASET_NAME = "mlg-ulb/creditcardfraud"
CSV_FILENAME = "creditcard.csv"


def _find_creditcard_csv(dataset_dir: Path) -> Path:
	direct_csv = dataset_dir / CSV_FILENAME
	if direct_csv.exists():
		return direct_csv

	matches = list(dataset_dir.rglob(CSV_FILENAME))
	if not matches:
		raise FileNotFoundError(
			f"Could not find {CSV_FILENAME} inside downloaded dataset directory: {dataset_dir}"
		)
	return matches[0]


def download_creditcardfraud_dataset() -> tuple[Path, Path]:
	"""Download dataset through kagglehub and copy creditcard.csv into project data folder."""
	cache_dir = Path(kagglehub.dataset_download(DATASET_NAME))
	source_csv = _find_creditcard_csv(cache_dir)

	RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
	destination_csv = RAW_DATA_DIR / CSV_FILENAME
	shutil.copy2(source_csv, destination_csv)

	return cache_dir, destination_csv


def main() -> None:
	try:
		cache_path, local_csv = download_creditcardfraud_dataset()
	except Exception as error:
		print(f"Dataset download failed: {error}", file=sys.stderr)
		raise SystemExit(1) from error

	print(f"Path to dataset files: {cache_path}")
	print(f"Copied dataset file to: {local_csv}")


if __name__ == "__main__":
	main()
