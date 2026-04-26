from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import ensure_project_root_on_path
from src.core.console import print_info, print_ok, print_section


def main() -> None:
    ensure_project_root_on_path()

    from src.db import EXPECTED_TABLES, get_existing_tables, initialize_database
    from src.services.schema_explainer import explain_database_schema

    print_section("Phase 3 Verification")
    initialize_database()

    explanation = explain_database_schema()
    explained_table_names = {table.table_name for table in explanation.tables}
    actual_table_names = set(get_existing_tables()) & set(EXPECTED_TABLES)

    if explained_table_names != actual_table_names:
        raise SystemExit(
            f"Explained tables do not match actual schema tables. explained={sorted(explained_table_names)} "
            f"actual={sorted(actual_table_names)}"
        )
    print_ok("Every expected application table is covered by the explanation layer.")

    required_relations = {
        ("transactions", "users"),
        ("predictions", "transactions"),
        ("fraud_alerts", "transactions"),
        ("dataset_profiles", "raw_dataset_uploads"),
        ("feature_profiles", "raw_dataset_uploads"),
        ("feature_profiles", "dataset_profiles"),
        ("model_recommendations", "model_training_runs"),
        ("model_candidate_metrics", "model_training_runs"),
    }

    actual_relations = {
        (table.table_name, foreign_key.reference_table)
        for table in explanation.tables
        for foreign_key in table.foreign_keys
    }

    missing_relations = sorted(required_relations - actual_relations)
    if missing_relations:
        raise SystemExit(f"Missing expected foreign-key explanations: {missing_relations}")
    print_ok("All critical foreign-key relationships are explained.")

    if not explanation.mermaid_er_diagram.startswith("erDiagram"):
        raise SystemExit("Mermaid ER diagram output is malformed.")
    if "..." in explanation.mermaid_er_diagram:
        raise SystemExit("Mermaid ER diagram contains an invalid placeholder token.")
    print_ok("Mermaid ER diagram text was generated.")

    if len(explanation.normalization_summary) < 3:
        raise SystemExit("Normalization summary is too small for Phase 3 requirements.")
    print_ok("Normalization summary is present and non-trivial.")

    transactions_table = next(table for table in explanation.tables if table.table_name == "transactions")
    if "location" not in {column.name for column in transactions_table.columns}:
        raise SystemExit("Transactions explanation is missing expected columns.")
    print_ok("Transactions table explanation matches the live schema.")

    print_info(f"Explained tables: {len(explanation.tables)}")
    print_info(f"Layer summaries: {len(explanation.layer_summaries)}")
    print_info(f"Normalization points: {len(explanation.normalization_summary)}")
    print_ok("Phase 3 verification completed successfully.")


if __name__ == "__main__":
    main()
