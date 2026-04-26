from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import SAMPLE_PROFILE_CSV_PATH, ensure_project_root_on_path
from src.core.console import print_info, print_ok, print_section, print_warning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a CSV dataset and store the summary in SQLite.")
    parser.add_argument(
        "--csv-path",
        default=str(SAMPLE_PROFILE_CSV_PATH),
        help="Path to the CSV file to profile.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional target column override.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=50,
        help="Maximum number of column explanations to print.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_project_root_on_path()

    from src.db import initialize_database
    from src.services.dataset_profiling import profile_csv_dataset

    args = parse_args()
    initialize_database()

    result = profile_csv_dataset(
        csv_path=Path(args.csv_path),
        target_column=args.target_column,
    )

    print_section("Dataset Summary")
    print_info(f"File: {result.filename}")
    print_info(f"Rows: {result.row_count}")
    print_info(f"Columns: {result.column_count}")
    print_info(f"Detected target column: {result.target_column or 'not detected'}")
    if result.target_candidates:
        print_info(f"Top target candidates: {', '.join(result.target_candidates)}")
    print_info(f"Duplicate rows: {result.duplicate_row_count}")
    print_info(f"Missing cells: {result.missing_cell_count}")
    print_info(
        f"Column types: numeric={result.numeric_column_count}, "
        f"categorical_or_binary={result.categorical_column_count}, datetime={result.datetime_column_count}"
    )
    if result.class_distribution:
        print_info(f"Class distribution: {result.class_distribution}")
    if result.class_imbalance_ratio is not None:
        print_info(f"Imbalance ratio: {result.class_imbalance_ratio:.2%}")

    if result.warnings:
        print_section("Warnings")
        for warning in result.warnings:
            print_warning(warning)
    else:
        print_ok("No major profiling warnings were detected.")

    print_section("Column Explanations")
    for feature in result.feature_profiles[: args.max_columns]:
        print_info(
            f"{feature.column_name} [{feature.inferred_role}/{feature.inferred_dtype}] "
            f"{feature.simple_description}"
        )
        print_info(f"Technical: {feature.technical_description}")

    print_section("Persistence")
    print_ok(f"raw_dataset_uploads ID: {result.upload_id}")
    print_ok(f"dataset_profiles ID: {result.dataset_profile_id}")
    print_ok(f"Stored feature profiles: {len(result.feature_profiles)}")


if __name__ == "__main__":
    main()
