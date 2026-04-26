from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import MODEL_METADATA_PATH, MODEL_PATH, ensure_project_root_on_path
from src.core.console import print_info, print_ok, print_section, print_warning


def main() -> None:
    ensure_project_root_on_path()

    from src.db import EXPECTED_TABLES, get_existing_tables, get_table_row_count, initialize_database
    from src.predict import get_prediction_threshold
    from src.services.workflow import run_workflow

    print_section("Phase 1 Verification")
    initialize_database()

    existing_tables = set(get_existing_tables())
    missing_tables = sorted(set(EXPECTED_TABLES) - existing_tables)
    if missing_tables:
        raise SystemExit(f"Missing database tables: {missing_tables}")
    print_ok("Database schema initialized and all expected tables are present.")

    print_info("Running forced training workflow to verify retraining path...")
    force_train_result = run_workflow(force_train=True)
    if force_train_result.model_metrics is None:
        raise SystemExit("Forced training did not produce model metrics.")
    print_ok("Forced training workflow completed.")

    print_info("Running standard workflow to verify reuse path...")
    standard_result = run_workflow(force_train=False)
    print_ok("Standard workflow completed.")

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing trained model artifact: {MODEL_PATH}")
    if not MODEL_METADATA_PATH.exists():
        raise SystemExit(f"Missing model metadata artifact: {MODEL_METADATA_PATH}")
    print_ok("Model artifact and metadata file are present.")

    threshold = get_prediction_threshold()
    print_ok(f"Prediction threshold resolved successfully: {threshold:.4f}")

    for table_name in ("users", "transactions", "predictions", "fraud_alerts", "model_training_runs"):
        row_count = get_table_row_count(table_name)
        print_info(f"{table_name} rows: {row_count}")
        if row_count <= 0:
            raise SystemExit(f"Expected rows in table {table_name}, found {row_count}.")

    if standard_result.alert_id is None:
        print_warning("Standard workflow did not create an alert. This is allowed if prediction is non-fraud.")
    else:
        print_ok(f"Fraud alert created successfully with ID: {standard_result.alert_id}")

    print_ok("Phase 1 verification completed successfully.")


if __name__ == "__main__":
    main()
