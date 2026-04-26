from __future__ import annotations

from dataclasses import dataclass

from src.db import EXPECTED_TABLES, get_existing_tables, get_foreign_keys, get_indexes, get_table_columns


TABLE_LAYER = {
    "users": "operational",
    "transactions": "operational",
    "predictions": "operational",
    "fraud_alerts": "operational",
    "kaggle_transactions": "raw_training",
    "raw_dataset_uploads": "raw_profiling",
    "dataset_profiles": "analytics",
    "feature_profiles": "analytics",
    "model_training_runs": "analytics",
    "model_recommendations": "analytics",
    "model_candidate_metrics": "analytics",
}

TABLE_PURPOSE = {
    "users": "Stores cardholder identity records so customer details are not repeated in every transaction row.",
    "transactions": "Stores the core transaction facts that the fraud workflow evaluates.",
    "predictions": "Stores the model decision and fraud probability for each processed transaction.",
    "fraud_alerts": "Stores alert records for transactions that were marked as suspicious.",
    "kaggle_transactions": "Stores imported raw Kaggle fraud records for database-first model training.",
    "raw_dataset_uploads": "Registers every uploaded dataset file that has been profiled by the system.",
    "dataset_profiles": "Stores dataset-level profiling summaries such as missing values, duplicates, and class imbalance.",
    "feature_profiles": "Stores column-level profiling details and plain-language explanations for every dataset field.",
    "model_training_runs": "Stores one summary row for each model training session and selected winner.",
    "model_recommendations": "Stores the three shortlisted models, their recommendation rank, and the reason they were chosen.",
    "model_candidate_metrics": "Stores evaluation details for every candidate model considered in a training run.",
}

NORMALIZATION_NOTES = {
    "users": "User identity is separated from transactions, which reduces repeated customer details and supports 3NF-style separation.",
    "transactions": "Transaction facts are stored once per event and link back to users through a foreign key.",
    "predictions": "Prediction results are separated from transactions because they describe model output, not original transaction facts.",
    "fraud_alerts": "Alerts are separated from predictions and transactions because they represent follow-up actions, not the base transaction itself.",
    "kaggle_transactions": "This table is intentionally denormalized because it preserves imported model-training rows in a wide raw format.",
    "raw_dataset_uploads": "This table keeps uploaded-file metadata separate from profile results so the source file can be tracked independently.",
    "dataset_profiles": "Dataset-level metrics are separated from file registration and column details to avoid mixing summary data with row-level feature explanations.",
    "feature_profiles": "Each feature explanation is stored in its own row, which avoids repeating dataset-level summary information for every column.",
    "model_training_runs": "Training-run summary data is separated from candidate-level metrics so the chosen winner and the compared models do not duplicate each other.",
    "model_recommendations": "Shortlisted models are stored separately from both training-run summary data and candidate metrics so recommendation logic is traceable.",
    "model_candidate_metrics": "Candidate metrics depend on a parent training run and are stored separately to preserve one-to-many comparison history.",
}

LAYER_DESCRIPTIONS = {
    "operational": "Operational layer: tables used by the live fraud transaction workflow.",
    "raw_training": "Raw training layer: imported source data kept in a wide form for model training.",
    "raw_profiling": "Raw profiling layer: uploaded file registration before deeper analysis.",
    "analytics": "Analytics and audit layer: profiling results, model history, and evaluation metadata.",
}


@dataclass
class ColumnExplanation:
    name: str
    data_type: str
    nullable: bool
    default_value: str | None
    is_primary_key: bool
    simple_description: str


@dataclass
class ForeignKeyExplanation:
    from_column: str
    reference_table: str
    reference_column: str
    on_delete: str
    simple_description: str


@dataclass
class TableExplanation:
    table_name: str
    layer: str
    purpose: str
    normalization_note: str
    primary_key_columns: list[str]
    columns: list[ColumnExplanation]
    foreign_keys: list[ForeignKeyExplanation]
    index_names: list[str]
    simple_relationship_summary: str


@dataclass
class DatabaseExplanation:
    tables: list[TableExplanation]
    layer_summaries: list[str]
    normalization_summary: list[str]
    simple_overview: str
    technical_overview: str
    mermaid_er_diagram: str


def _column_simple_description(table_name: str, column: dict[str, object]) -> str:
    column_name = str(column["name"])
    if int(column["pk"]) == 1:
        return f"`{column_name}` is the main unique key for the `{table_name}` table."
    if column_name.endswith("_id"):
        return f"`{column_name}` links this row to another table."
    if column_name in {"prediction", "class_label", "target_candidate", "selected", "overfit_flag", "underfit_flag"}:
        return f"`{column_name}` stores a yes/no style result used by the system."
    if "time" in column_name:
        return f"`{column_name}` records when something happened."
    if "count" in column_name:
        return f"`{column_name}` stores how many items were observed."
    if "ratio" in column_name or "probability" in column_name or "threshold" in column_name:
        return f"`{column_name}` stores a calculated scoring value."
    if "json" in column_name:
        return f"`{column_name}` stores structured summary details in JSON text."
    return f"`{column_name}` stores one attribute that belongs to the `{table_name}` table."


def _build_column_explanations(table_name: str) -> tuple[list[ColumnExplanation], list[str]]:
    columns = get_table_columns(table_name)
    column_explanations: list[ColumnExplanation] = []
    primary_key_columns: list[str] = []

    for column in columns:
        is_primary_key = int(column["pk"]) == 1
        if is_primary_key:
            primary_key_columns.append(str(column["name"]))

        column_explanations.append(
            ColumnExplanation(
                name=str(column["name"]),
                data_type=str(column["type"]),
                nullable=not bool(column["notnull"]),
                default_value=None if column["dflt_value"] is None else str(column["dflt_value"]),
                is_primary_key=is_primary_key,
                simple_description=_column_simple_description(table_name, column),
            )
        )

    return column_explanations, primary_key_columns


def _build_foreign_key_explanations(table_name: str) -> list[ForeignKeyExplanation]:
    explanations: list[ForeignKeyExplanation] = []
    for foreign_key in get_foreign_keys(table_name):
        from_column = str(foreign_key["from"])
        reference_table = str(foreign_key["table"])
        reference_column = str(foreign_key["to"])
        on_delete = str(foreign_key["on_delete"])
        explanations.append(
            ForeignKeyExplanation(
                from_column=from_column,
                reference_table=reference_table,
                reference_column=reference_column,
                on_delete=on_delete,
                simple_description=(
                    f"`{from_column}` in `{table_name}` points to `{reference_column}` in `{reference_table}` "
                    f"so the child row always belongs to a valid parent row."
                ),
            )
        )
    return explanations


def _relationship_summary(table_name: str, foreign_keys: list[ForeignKeyExplanation]) -> str:
    if not foreign_keys:
        return f"`{table_name}` does not depend on another table through foreign keys."

    relations = [
        f"`{item.from_column}` -> `{item.reference_table}.{item.reference_column}`"
        for item in foreign_keys
    ]
    return f"`{table_name}` connects to other tables through: {', '.join(relations)}."


def _build_table_explanation(table_name: str) -> TableExplanation:
    columns, primary_key_columns = _build_column_explanations(table_name)
    foreign_keys = _build_foreign_key_explanations(table_name)
    indexes = [str(index["name"]) for index in get_indexes(table_name)]

    return TableExplanation(
        table_name=table_name,
        layer=TABLE_LAYER.get(table_name, "uncategorized"),
        purpose=TABLE_PURPOSE.get(table_name, f"`{table_name}` is part of the system schema."),
        normalization_note=NORMALIZATION_NOTES.get(
            table_name,
            f"`{table_name}` should be explained according to how it reduces duplication or stores dependent data.",
        ),
        primary_key_columns=primary_key_columns,
        columns=columns,
        foreign_keys=foreign_keys,
        index_names=indexes,
        simple_relationship_summary=_relationship_summary(table_name, foreign_keys),
    )


def _build_layer_summaries(table_names: list[str]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for table_name in table_names:
        layer = TABLE_LAYER.get(table_name, "uncategorized")
        grouped.setdefault(layer, []).append(table_name)

    summaries: list[str] = []
    for layer_name, tables in sorted(grouped.items()):
        layer_description = LAYER_DESCRIPTIONS.get(layer_name, f"{layer_name} layer")
        summaries.append(f"{layer_description} Tables: {', '.join(sorted(tables))}.")
    return summaries


def _build_normalization_summary(table_names: list[str]) -> list[str]:
    summaries = [
        "The schema separates operational transactions from predictions and alerts, which avoids storing model output inside the base transaction row.",
        "The schema separates training-run summaries from candidate-level metrics, which preserves one-to-many model comparison history without duplication.",
        "The schema also stores recommendation rows separately so model selection reasoning is preserved before final winner evaluation.",
        "The profiling layer separates raw uploaded-file metadata, dataset-level summaries, and feature-level explanations into different tables.",
    ]

    if "transactions" in table_names:
        summaries.append(
            "The operational flow is close to 3NF because user identity is stored in `users` while transaction facts live in `transactions`."
        )
        summaries.append(
            "The schema is not fully normalized because `location` and `merchant` are still plain text in `transactions`; these could later move into lookup tables."
        )
    if "kaggle_transactions" in table_names:
        summaries.append(
            "`kaggle_transactions` is intentionally denormalized because it preserves imported training rows in the same wide shape as the source dataset."
        )
    return summaries


def _build_mermaid_er_diagram(table_explanations: list[TableExplanation]) -> str:
    lines = ["erDiagram"]
    seen_relations: set[str] = set()

    for table in table_explanations:
        table_name_upper = table.table_name.upper()
        for foreign_key in table.foreign_keys:
            relation = f"    {foreign_key.reference_table.upper()} ||--o{{ {table_name_upper} : references"
            if relation not in seen_relations:
                seen_relations.add(relation)
                lines.append(relation)

    for table in table_explanations:
        lines.append(f"    {table.table_name.upper()} {{")
        for column in table.columns[:8]:
            key_suffix = " PK" if column.is_primary_key else ""
            lines.append(f"        {column.data_type.lower()} {column.name}{key_suffix}")
        if len(table.columns) > 8:
            lines.append("        string ...")
        lines.append("    }")

    return "\n".join(lines)


def explain_database_schema() -> DatabaseExplanation:
    table_names = [name for name in get_existing_tables() if name in EXPECTED_TABLES]
    table_names.sort()
    table_explanations = [_build_table_explanation(table_name) for table_name in table_names]

    operational_tables = [table.table_name for table in table_explanations if table.layer == "operational"]
    analytics_tables = [table.table_name for table in table_explanations if table.layer == "analytics"]

    simple_overview = (
        "This database is split into operational tables for live fraud workflow, raw-data tables for imported files, "
        "and analytics tables for profiling and model history."
    )
    technical_overview = (
        f"The schema currently contains {len(table_explanations)} application tables. "
        f"Operational tables: {', '.join(sorted(operational_tables))}. "
        f"Analytics and audit tables: {', '.join(sorted(analytics_tables))}."
    )

    return DatabaseExplanation(
        tables=table_explanations,
        layer_summaries=_build_layer_summaries(table_names),
        normalization_summary=_build_normalization_summary(table_names),
        simple_overview=simple_overview,
        technical_overview=technical_overview,
        mermaid_er_diagram=_build_mermaid_er_diagram(table_explanations),
    )
