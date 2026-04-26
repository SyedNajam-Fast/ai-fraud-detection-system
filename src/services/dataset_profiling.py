from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from src.db import insert_dataset_profile, insert_feature_profiles, insert_raw_dataset_upload


TARGET_NAME_HINTS = {
    "class",
    "label",
    "target",
    "fraud",
    "is_fraud",
    "fraud_flag",
    "y",
}
ID_NAME_HINTS = {"id", "transaction_id", "user_id", "customer_id", "record_id"}
TIME_NAME_HINTS = {"time", "date", "datetime", "timestamp", "created_at", "event_time"}
AMOUNT_NAME_HINTS = {"amount", "value", "price", "balance", "cost", "payment"}


@dataclass
class FeatureProfileResult:
    column_name: str
    inferred_role: str
    inferred_dtype: str
    pandas_dtype: str
    non_null_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    sample_values: list[str]
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    std_value: float | None
    simple_description: str
    technical_description: str
    target_candidate: bool


@dataclass
class DatasetProfileResult:
    upload_id: int
    dataset_profile_id: int
    filename: str
    source_path: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    missing_cell_count: int
    numeric_column_count: int
    categorical_column_count: int
    datetime_column_count: int
    target_column: str | None
    target_candidates: list[str]
    target_cardinality: int | None
    class_distribution: dict[str, int]
    class_imbalance_ratio: float | None
    warnings: list[str]
    feature_profiles: list[FeatureProfileResult]
    created_at: str


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _sample_values(series: pd.Series, limit: int = 3) -> list[str]:
    values = []
    for value in series.dropna().astype(str).unique()[:limit]:
        values.append(value)
    return values


def _detect_datetime_like(series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series):
        return True
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    string_values = non_null.astype(str).str.strip()
    datetime_pattern_ratio = float(
        string_values.str.contains(r"[-/:T]", regex=True).mean()
    )
    if datetime_pattern_ratio < 0.6:
        return False

    parsed = pd.to_datetime(string_values, errors="coerce")
    success_ratio = float(parsed.notna().mean())
    return success_ratio >= 0.8


def _infer_dtype(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "binary"
    if _detect_datetime_like(series):
        return "datetime"
    if is_numeric_dtype(series):
        unique_count = int(series.dropna().nunique())
        if unique_count <= 2:
            return "binary"
        return "numeric"
    return "categorical"


def _infer_role(column_name: str, inferred_dtype: str, unique_ratio: float, unique_count: int) -> str:
    lowered = column_name.strip().lower()
    if lowered in TARGET_NAME_HINTS:
        return "target"
    if lowered in ID_NAME_HINTS or lowered.endswith("_id") or (unique_ratio >= 0.98 and unique_count > 0):
        return "identifier"
    if lowered in TIME_NAME_HINTS or inferred_dtype == "datetime":
        return "time"
    if lowered in AMOUNT_NAME_HINTS:
        return "amount"
    return "feature"


def _detect_target_candidates(dataframe: pd.DataFrame) -> list[str]:
    scored_candidates: list[tuple[int, str]] = []
    row_count = len(dataframe)

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        inferred_dtype = _infer_dtype(series)
        unique_count = int(series.dropna().nunique())
        non_null_count = int(series.notna().sum())
        unique_ratio = float(unique_count / non_null_count) if non_null_count else 0.0

        score = 0
        lowered = column_name.strip().lower()
        if lowered in TARGET_NAME_HINTS:
            score += 100
        if inferred_dtype == "binary":
            score += 40
        if 2 <= unique_count <= min(10, max(row_count, 1)) and unique_ratio <= 0.5:
            score += 15
        if unique_ratio < 0.2:
            score += 10
        if lowered.endswith("_flag") or lowered.startswith("is_"):
            score += 15

        if score > 0:
            scored_candidates.append((score, column_name))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return [column_name for _, column_name in scored_candidates[:5]]


def _build_simple_description(column_name: str, inferred_role: str, inferred_dtype: str, missing_ratio: float) -> str:
    pretty_name = column_name.replace("_", " ")
    if inferred_role == "target":
        base = f"`{column_name}` looks like the answer column the model will try to predict."
    elif inferred_role == "identifier":
        base = f"`{column_name}` looks like an ID column used to uniquely recognize each record."
    elif inferred_role == "time":
        base = f"`{column_name}` appears to describe when an event or transaction happened."
    elif inferred_role == "amount":
        base = f"`{column_name}` appears to show a money-related value for each record."
    elif inferred_dtype == "numeric":
        base = f"`{column_name}` is a number-based column that can be used to measure or compare records."
    elif inferred_dtype == "datetime":
        base = f"`{column_name}` stores date or time information."
    else:
        base = f"`{column_name}` stores category-style information such as names, groups, or labels."

    if missing_ratio > 0:
        return base + f" Some values are missing in this column ({missing_ratio:.1%} missing)."
    return base


def _build_technical_description(
    column_name: str,
    inferred_role: str,
    inferred_dtype: str,
    pandas_dtype: str,
    unique_count: int,
    missing_count: int,
) -> str:
    return (
        f"`{column_name}` is profiled as a {inferred_role} column with inferred type "
        f"`{inferred_dtype}` and pandas dtype `{pandas_dtype}`. It has {unique_count} unique "
        f"non-null values and {missing_count} missing values."
    )


def _build_feature_profile(
    column_name: str,
    series: pd.Series,
    row_count: int,
    target_candidates: set[str],
    target_column: str | None,
) -> FeatureProfileResult:
    pandas_dtype = str(series.dtype)
    non_null_count = int(series.notna().sum())
    missing_count = int(series.isna().sum())
    missing_ratio = float(missing_count / row_count) if row_count else 0.0
    unique_count = int(series.dropna().nunique())
    unique_ratio = float(unique_count / non_null_count) if non_null_count else 0.0
    inferred_dtype = _infer_dtype(series)
    inferred_role = "target" if target_column == column_name else _infer_role(
        column_name,
        inferred_dtype,
        unique_ratio,
        unique_count,
    )

    min_value = max_value = mean_value = std_value = None
    if inferred_dtype in {"numeric", "binary"} and non_null_count > 0:
        numeric_series = pd.to_numeric(series, errors="coerce")
        min_value = _safe_float(numeric_series.min())
        max_value = _safe_float(numeric_series.max())
        mean_value = _safe_float(numeric_series.mean())
        std_value = _safe_float(numeric_series.std())

    return FeatureProfileResult(
        column_name=column_name,
        inferred_role=inferred_role,
        inferred_dtype=inferred_dtype,
        pandas_dtype=pandas_dtype,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        sample_values=_sample_values(series),
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        std_value=std_value,
        simple_description=_build_simple_description(column_name, inferred_role, inferred_dtype, missing_ratio),
        technical_description=_build_technical_description(
            column_name,
            inferred_role,
            inferred_dtype,
            pandas_dtype,
            unique_count,
            missing_count,
        ),
        target_candidate=column_name in target_candidates,
    )


def _class_distribution(dataframe: pd.DataFrame, target_column: str | None) -> dict[str, int]:
    if not target_column:
        return {}

    series = dataframe[target_column]
    counts = series.value_counts(dropna=False)
    distribution: dict[str, int] = {}
    for key, value in counts.items():
        if pd.isna(key):
            label = "missing"
        else:
            label = str(key)
        distribution[label] = int(value)
    return distribution


def _imbalance_ratio(distribution: dict[str, int], row_count: int) -> float | None:
    if not distribution or row_count <= 0:
        return None
    return max(distribution.values()) / row_count


def _dataset_warnings(
    row_count: int,
    duplicate_row_count: int,
    missing_cell_count: int,
    class_imbalance_ratio: float | None,
) -> list[str]:
    warnings: list[str] = []
    if row_count == 0:
        warnings.append("The dataset has no rows.")
    if duplicate_row_count > 0:
        warnings.append(f"The dataset contains {duplicate_row_count} duplicate rows.")
    if missing_cell_count > 0:
        warnings.append(f"The dataset contains {missing_cell_count} missing cells.")
    if class_imbalance_ratio is not None and class_imbalance_ratio >= 0.8:
        warnings.append(
            f"The target distribution is imbalanced because one class holds {class_imbalance_ratio:.1%} of the rows."
        )
    if row_count < 100:
        warnings.append("The dataset is small, so model evaluation may be unstable.")
    return warnings


def _serialize_feature_profiles(
    upload_id: int,
    dataset_profile_id: int,
    feature_profiles: list[FeatureProfileResult],
    created_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in feature_profiles:
        rows.append(
            {
                "upload_id": upload_id,
                "dataset_profile_id": dataset_profile_id,
                "column_name": feature.column_name,
                "inferred_role": feature.inferred_role,
                "inferred_dtype": feature.inferred_dtype,
                "pandas_dtype": feature.pandas_dtype,
                "non_null_count": feature.non_null_count,
                "missing_count": feature.missing_count,
                "missing_ratio": feature.missing_ratio,
                "unique_count": feature.unique_count,
                "unique_ratio": feature.unique_ratio,
                "sample_values_json": json.dumps(feature.sample_values),
                "min_value": feature.min_value,
                "max_value": feature.max_value,
                "mean_value": feature.mean_value,
                "std_value": feature.std_value,
                "simple_description": feature.simple_description,
                "technical_description": feature.technical_description,
                "target_candidate": 1 if feature.target_candidate else 0,
                "created_at": created_at,
            }
        )
    return rows


def profile_csv_dataset(csv_path: Path, target_column: str | None = None) -> DatasetProfileResult:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    duplicate_row_count = int(dataframe.duplicated().sum())
    missing_cell_count = int(dataframe.isna().sum().sum())
    detected_candidates = _detect_target_candidates(dataframe)

    if target_column is not None and target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found in {csv_path}.")

    resolved_target_column = target_column or (detected_candidates[0] if detected_candidates else None)
    candidate_set = set(detected_candidates)
    feature_profiles = [
        _build_feature_profile(
            column_name=column_name,
            series=dataframe[column_name],
            row_count=row_count,
            target_candidates=candidate_set,
            target_column=resolved_target_column,
        )
        for column_name in dataframe.columns
    ]

    numeric_column_count = sum(1 for item in feature_profiles if item.inferred_dtype == "numeric")
    categorical_column_count = sum(
        1 for item in feature_profiles if item.inferred_dtype in {"categorical", "binary"}
    )
    datetime_column_count = sum(1 for item in feature_profiles if item.inferred_dtype == "datetime")

    class_distribution = _class_distribution(dataframe, resolved_target_column)
    class_imbalance_ratio = _imbalance_ratio(class_distribution, row_count)
    warnings = _dataset_warnings(row_count, duplicate_row_count, missing_cell_count, class_imbalance_ratio)
    target_cardinality = len(class_distribution) if class_distribution else None

    upload_id = insert_raw_dataset_upload(
        {
            "filename": csv_path.name,
            "source_path": str(csv_path.resolve()),
            "file_size_bytes": int(csv_path.stat().st_size),
            "row_count": row_count,
            "column_count": column_count,
            "target_column": resolved_target_column,
            "status": "profiled",
            "created_at": created_at,
        }
    )

    summary_payload = {
        "target_candidates": detected_candidates,
        "class_distribution": class_distribution,
        "warnings": warnings,
    }
    dataset_profile_id = insert_dataset_profile(
        {
            "upload_id": upload_id,
            "row_count": row_count,
            "column_count": column_count,
            "duplicate_row_count": duplicate_row_count,
            "missing_cell_count": missing_cell_count,
            "numeric_column_count": numeric_column_count,
            "categorical_column_count": categorical_column_count,
            "datetime_column_count": datetime_column_count,
            "target_column": resolved_target_column,
            "target_cardinality": target_cardinality,
            "class_imbalance_ratio": class_imbalance_ratio,
            "warnings_json": json.dumps(warnings),
            "summary_json": json.dumps(summary_payload),
            "created_at": created_at,
        }
    )

    insert_feature_profiles(
        _serialize_feature_profiles(
            upload_id=upload_id,
            dataset_profile_id=dataset_profile_id,
            feature_profiles=feature_profiles,
            created_at=created_at,
        )
    )

    return DatasetProfileResult(
        upload_id=upload_id,
        dataset_profile_id=dataset_profile_id,
        filename=csv_path.name,
        source_path=str(csv_path.resolve()),
        row_count=row_count,
        column_count=column_count,
        duplicate_row_count=duplicate_row_count,
        missing_cell_count=missing_cell_count,
        numeric_column_count=numeric_column_count,
        categorical_column_count=categorical_column_count,
        datetime_column_count=datetime_column_count,
        target_column=resolved_target_column,
        target_candidates=detected_candidates,
        target_cardinality=target_cardinality,
        class_distribution=class_distribution,
        class_imbalance_ratio=class_imbalance_ratio,
        warnings=warnings,
        feature_profiles=feature_profiles,
        created_at=created_at,
    )
