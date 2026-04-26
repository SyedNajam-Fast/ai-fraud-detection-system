from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.core.config import MODEL_SHORTLIST_SIZE


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradientBoosting",
    "adaboost": "AdaBoost",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
}

MODEL_PRIORITY_ORDER = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "gradient_boosting",
    "adaboost",
    "decision_tree",
    "svm",
    "knn",
    "naive_bayes",
]

TREE_ENSEMBLES = {
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "adaboost",
}


@dataclass
class RecommendedModel:
    model_name: str
    display_name: str
    score: float
    rationale: str
    shortlist_rank: int | None
    shortlisted: bool


@dataclass
class RecommendationSummary:
    dataset_characteristics: dict[str, Any]
    shortlisted_models: list[RecommendedModel]
    all_models: list[RecommendedModel]
    recommendation_strategy: str


def _dataset_characteristics(features: pd.DataFrame, target: pd.Series, dataset_source: str) -> dict[str, Any]:
    sample_count = int(len(features))
    numeric_columns = [column for column in features.columns if is_numeric_dtype(features[column])]
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    missing_ratio = float(features.isna().sum().sum() / max(features.size, 1))
    encoded_feature_estimate = len(numeric_columns) + sum(
        min(int(features[column].dropna().nunique()), 25) for column in categorical_columns
    )
    fraud_rate = float(target.mean())
    majority_class_ratio = max(fraud_rate, 1.0 - fraud_rate)
    class_imbalance = majority_class_ratio >= 0.75

    return {
        "dataset_source": dataset_source,
        "sample_count": sample_count,
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "missing_ratio": missing_ratio,
        "encoded_feature_estimate": int(encoded_feature_estimate),
        "fraud_rate": fraud_rate,
        "majority_class_ratio": majority_class_ratio,
        "class_imbalance_detected": class_imbalance,
    }


def _add_score(score_map: dict[str, float], reason_map: dict[str, list[str]], model_name: str, points: float, reason: str) -> None:
    score_map[model_name] += points
    reason_map[model_name].append(reason)


def build_model_recommendation_summary(
    features: pd.DataFrame,
    target: pd.Series,
    dataset_source: str,
) -> RecommendationSummary:
    characteristics = _dataset_characteristics(features, target, dataset_source)
    sample_count = int(characteristics["sample_count"])
    categorical_feature_count = int(characteristics["categorical_feature_count"])
    missing_ratio = float(characteristics["missing_ratio"])
    encoded_feature_estimate = int(characteristics["encoded_feature_estimate"])
    class_imbalance_detected = bool(characteristics["class_imbalance_detected"])
    numeric_feature_count = int(characteristics["numeric_feature_count"])

    scores = {model_name: 0.0 for model_name in MODEL_DISPLAY_NAMES}
    reasons = {model_name: [] for model_name in MODEL_DISPLAY_NAMES}

    _add_score(scores, reasons, "logistic_regression", 10, "It provides a strong interpretable baseline for fraud classification.")
    _add_score(scores, reasons, "random_forest", 8, "It handles mixed feature interactions well in tabular fraud data.")
    _add_score(scores, reasons, "extra_trees", 7, "It is useful for non-linear tabular patterns and fast ensemble comparisons.")

    if sample_count <= 5000:
        _add_score(scores, reasons, "logistic_regression", 10, "The dataset is small enough for a reliable linear baseline.")
        _add_score(scores, reasons, "svm", 11, "The dataset is small enough for an SVM to be computationally realistic.")
        _add_score(scores, reasons, "knn", 9, "The dataset is small enough for distance-based comparison to remain practical.")
        _add_score(scores, reasons, "decision_tree", 8, "A small dataset benefits from an easy-to-explain tree baseline.")
        _add_score(scores, reasons, "naive_bayes", 6, "A fast probabilistic baseline is practical on a small dataset.")
    elif sample_count <= 30000:
        _add_score(scores, reasons, "random_forest", 9, "The dataset size fits ensemble tree training well.")
        _add_score(scores, reasons, "extra_trees", 9, "The dataset size fits a randomized tree ensemble well.")
        _add_score(scores, reasons, "gradient_boosting", 8, "Boosting is feasible at this dataset size.")
        _add_score(scores, reasons, "hist_gradient_boosting", 8, "Histogram boosting is efficient on medium-sized datasets.")
        _add_score(scores, reasons, "adaboost", 6, "AdaBoost remains practical at this scale.")
    else:
        _add_score(scores, reasons, "hist_gradient_boosting", 14, "Large datasets favor boosting methods with efficient training behavior.")
        _add_score(scores, reasons, "logistic_regression", 8, "Large datasets still benefit from a stable linear baseline.")
        _add_score(scores, reasons, "naive_bayes", 8, "A very fast probabilistic baseline stays practical on large datasets.")
        _add_score(scores, reasons, "random_forest", 5, "Random forest remains relevant, but tree ensembles must stay computationally reasonable.")

    if categorical_feature_count > 0:
        _add_score(scores, reasons, "random_forest", 8, "Categorical feature interactions often favor tree ensembles.")
        _add_score(scores, reasons, "extra_trees", 8, "Categorical feature interactions often favor randomized tree ensembles.")
        _add_score(scores, reasons, "gradient_boosting", 5, "Boosting can capture category-driven non-linear behavior after encoding.")
        _add_score(scores, reasons, "hist_gradient_boosting", 5, "Boosting can benefit from mixed numeric and encoded categorical features.")
        _add_score(scores, reasons, "logistic_regression", 4, "A linear model remains a good benchmark after one-hot encoding.")

    if encoded_feature_estimate >= 10:
        _add_score(scores, reasons, "random_forest", 5, "The encoded feature space is rich enough to justify tree ensembles.")
        _add_score(scores, reasons, "extra_trees", 5, "The encoded feature space suggests non-linear interactions worth testing.")
        _add_score(scores, reasons, "hist_gradient_boosting", 4, "A richer feature space can reward gradient-based tree boosting.")

    if class_imbalance_detected:
        for model_name in ("logistic_regression", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting"):
            _add_score(scores, reasons, model_name, 7, "The target distribution is imbalanced, so robust probability ranking matters.")
        _add_score(scores, reasons, "adaboost", 5, "Boosting can still be useful when the fraud class is rarer than the normal class.")

    if missing_ratio > 0:
        for model_name in ("logistic_regression", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "naive_bayes"):
            _add_score(scores, reasons, model_name, 2, "The pipeline includes imputation, so this model can tolerate missing values.")

    if numeric_feature_count >= categorical_feature_count and sample_count >= 1000:
        _add_score(scores, reasons, "hist_gradient_boosting", 5, "The dataset has enough numeric signal for gradient-based tree methods.")
        _add_score(scores, reasons, "gradient_boosting", 4, "Numeric signal makes boosting a reasonable candidate.")

    ranked_model_names = sorted(
        MODEL_DISPLAY_NAMES,
        key=lambda model_name: (-scores[model_name], MODEL_PRIORITY_ORDER.index(model_name)),
    )

    shortlist_names: list[str] = []
    for model_name in ranked_model_names:
        if model_name not in shortlist_names:
            shortlist_names.append(model_name)
        if len(shortlist_names) == MODEL_SHORTLIST_SIZE:
            break

    if "logistic_regression" not in shortlist_names and sample_count < 100000:
        shortlist_names[-1] = "logistic_regression"

    shortlist_names = list(dict.fromkeys(shortlist_names))[:MODEL_SHORTLIST_SIZE]
    while len(shortlist_names) < MODEL_SHORTLIST_SIZE:
        fallback_name = MODEL_PRIORITY_ORDER[len(shortlist_names)]
        if fallback_name not in shortlist_names:
            shortlist_names.append(fallback_name)

    all_models: list[RecommendedModel] = []
    shortlisted_models: list[RecommendedModel] = []

    for model_name in ranked_model_names:
        shortlisted = model_name in shortlist_names
        shortlist_rank = shortlist_names.index(model_name) + 1 if shortlisted else None
        rationale_lines = reasons[model_name] or ["This model stays in the pool as a general fraud-detection candidate."]
        recommendation = RecommendedModel(
            model_name=model_name,
            display_name=MODEL_DISPLAY_NAMES[model_name],
            score=float(scores[model_name]),
            rationale=" ".join(rationale_lines),
            shortlist_rank=shortlist_rank,
            shortlisted=shortlisted,
        )
        all_models.append(recommendation)
        if shortlisted:
            shortlisted_models.append(recommendation)

    shortlisted_models.sort(key=lambda item: item.shortlist_rank or 99)

    return RecommendationSummary(
        dataset_characteristics=characteristics,
        shortlisted_models=shortlisted_models,
        all_models=all_models,
        recommendation_strategy=(
            "Rule-based shortlist using dataset size, imbalance, feature mix, missingness, and encoded feature estimate."
        ),
    )
