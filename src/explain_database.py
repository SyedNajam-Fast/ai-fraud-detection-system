from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.console import print_info, print_ok, print_section
from src.core.config import ensure_project_root_on_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain the current database schema and normalization design.")
    parser.add_argument(
        "--include-columns",
        action="store_true",
        help="Print column-by-column explanations for every table.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_project_root_on_path()

    from src.db import initialize_database
    from src.services.schema_explainer import explain_database_schema

    args = parse_args()
    initialize_database()
    explanation = explain_database_schema()

    print_section("Database Overview")
    print_info(explanation.simple_overview)
    print_info(explanation.technical_overview)

    print_section("Layer Summary")
    for line in explanation.layer_summaries:
        print_info(line)

    print_section("Normalization Summary")
    for line in explanation.normalization_summary:
        print_info(line)

    print_section("Table Explanations")
    for table in explanation.tables:
        print_info(f"Table: {table.table_name}")
        print_info(f"Layer: {table.layer}")
        print_info(f"Purpose: {table.purpose}")
        print_info(f"Primary key: {', '.join(table.primary_key_columns) if table.primary_key_columns else 'none'}")
        print_info(table.simple_relationship_summary)
        print_info(f"Normalization note: {table.normalization_note}")
        if table.index_names:
            print_info(f"Indexes: {', '.join(table.index_names)}")
        if table.foreign_keys:
            for foreign_key in table.foreign_keys:
                print_info(f"FK: {foreign_key.simple_description} ON DELETE {foreign_key.on_delete}.")
        if args.include_columns:
            for column in table.columns:
                print_info(f"Column `{column.name}`: {column.simple_description}")

    print_section("Mermaid ER Diagram")
    print(explanation.mermaid_er_diagram)
    print_ok("Database explanation generated successfully.")


if __name__ == "__main__":
    main()
