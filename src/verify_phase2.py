from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import KAGGLE_CSV_PATH, SAMPLE_PROFILE_CSV_PATH, ensure_project_root_on_path
from src.core.console import print_info, print_ok, print_section, print_warning


def main() -> None:
    ensure_project_root_on_path()

    from src.db import (
        EXPECTED_TABLES,
        get_dataset_profile_by_upload_id,
        get_existing_tables,
        get_feature_profiles_by_upload_id,
        get_latest_dataset_upload,
        get_table_row_count,
        initialize_database,
    )
    from src.services.dataset_profiling import profile_csv_dataset

    print_section("Phase 2 Verification")
    initialize_database()

    existing_tables = set(get_existing_tables())
    missing_tables = sorted(set(EXPECTED_TABLES) - existing_tables)
    if missing_tables:
        raise SystemExit(f"Missing expected tables after schema initialization: {missing_tables}")
    print_ok("Schema contains all expected Phase 2 tables.")

    if not SAMPLE_PROFILE_CSV_PATH.exists():
        raise SystemExit(f"Missing sample profiling dataset: {SAMPLE_PROFILE_CSV_PATH}")

    sample_result = profile_csv_dataset(SAMPLE_PROFILE_CSV_PATH)
    if sample_result.row_count <= 0 or sample_result.column_count <= 0:
        raise SystemExit("Sample profile produced invalid row or column counts.")
    if not sample_result.feature_profiles:
        raise SystemExit("Sample profile did not create feature profiles.")
    if sample_result.target_column is None:
        raise SystemExit("Sample profile failed to detect a target column.")
    print_ok("Sample dataset profiling completed successfully.")

    latest_upload = get_latest_dataset_upload()
    if latest_upload is None:
        raise SystemExit("No dataset upload row found after profiling.")
    persisted_profile = get_dataset_profile_by_upload_id(int(latest_upload["id"]))
    if persisted_profile is None:
        raise SystemExit("No dataset profile row found after profiling.")
    persisted_features = get_feature_profiles_by_upload_id(int(latest_upload["id"]))
    if len(persisted_features) != sample_result.column_count:
        raise SystemExit(
            f"Expected {sample_result.column_count} persisted feature profiles, found {len(persisted_features)}."
        )
    print_ok("Profiling results were persisted to SQLite correctly.")

    for table_name in ("raw_dataset_uploads", "dataset_profiles", "feature_profiles"):
        print_info(f"{table_name} rows: {get_table_row_count(table_name)}")

    if KAGGLE_CSV_PATH.exists():
        kaggle_result = profile_csv_dataset(KAGGLE_CSV_PATH, target_column="Class")
        print_ok(
            f"Kaggle profiling path works. Rows: {kaggle_result.row_count}, columns: {kaggle_result.column_count}"
        )
    else:
        print_warning("Kaggle CSV not found locally, so Kaggle profiling verification was skipped.")

    print_ok("Phase 2 verification completed successfully.")


if __name__ == "__main__":
    main()
