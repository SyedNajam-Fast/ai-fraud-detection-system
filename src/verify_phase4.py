from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import MODEL_SHORTLIST_SIZE, ensure_project_root_on_path
from src.core.console import print_info, print_ok, print_section


def main() -> None:
    ensure_project_root_on_path()

    from src.db import (
        get_latest_model_training_run,
        get_model_recommendations_by_run_id,
        get_table_row_count,
        initialize_database,
    )
    from model.train_model import recommend_models_for_current_dataset, train_and_save_model

    print_section("Phase 4 Verification")
    initialize_database()

    recommendation_result = recommend_models_for_current_dataset()
    recommendation_summary = recommendation_result["recommendation_summary"]
    if len(recommendation_summary.shortlisted_models) != MODEL_SHORTLIST_SIZE:
        raise SystemExit(
            f"Expected {MODEL_SHORTLIST_SIZE} shortlisted models, got {len(recommendation_summary.shortlisted_models)}."
        )
    print_ok("Recommendation engine produced exactly three shortlisted models.")

    train_result = train_and_save_model()
    shortlisted_models = train_result["shortlisted_models"]
    if len(shortlisted_models) != MODEL_SHORTLIST_SIZE:
        raise SystemExit(f"Training result expected {MODEL_SHORTLIST_SIZE} shortlisted models, got {len(shortlisted_models)}.")
    if len(train_result["full_model_pool"]) != 10:
        raise SystemExit(f"Expected a 10-model pool, got {len(train_result['full_model_pool'])}.")
    if "roc_auc" not in train_result["validation_metrics"] or "roc_auc" not in train_result["test_metrics"]:
        raise SystemExit("ROC AUC metrics are missing from training output.")
    print_ok("Expanded training pipeline ran and returned the expected metadata.")

    latest_run = get_latest_model_training_run()
    if latest_run is None:
        raise SystemExit("No latest training run found after Phase 4 training.")
    recommendations = get_model_recommendations_by_run_id(int(latest_run["id"]))
    if len(recommendations) != MODEL_SHORTLIST_SIZE:
        raise SystemExit(
            f"Expected {MODEL_SHORTLIST_SIZE} stored model recommendations, found {len(recommendations)}."
        )
    winner_count = sum(int(row["final_winner"]) for row in recommendations)
    if winner_count != 1:
        raise SystemExit(f"Expected exactly one final winner in model_recommendations, found {winner_count}.")
    print_ok("Shortlist recommendations were persisted to SQLite correctly.")

    print_info(f"model_training_runs rows: {get_table_row_count('model_training_runs')}")
    print_info(f"model_recommendations rows: {get_table_row_count('model_recommendations')}")
    print_info(f"model_candidate_metrics rows: {get_table_row_count('model_candidate_metrics')}")
    print_ok("Phase 4 verification completed successfully.")


if __name__ == "__main__":
    main()
