from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.console import print_info, print_ok, print_section
from src.core.config import ensure_project_root_on_path


def main() -> None:
    ensure_project_root_on_path()

    from model.train_model import recommend_models_for_current_dataset

    result = recommend_models_for_current_dataset()
    summary = result["recommendation_summary"]

    print_section("Dataset Characteristics")
    for key, value in summary.dataset_characteristics.items():
        print_info(f"{key}: {value}")

    print_section("Shortlist")
    for item in summary.shortlisted_models:
        print_info(
            f"Rank {item.shortlist_rank}: {item.model_name} (score={item.score:.1f})"
        )
        print_info(f"Reason: {item.rationale}")

    print_section("Full Model Pool")
    for item in summary.all_models:
        status = "shortlisted" if item.shortlisted else "not shortlisted"
        print_info(f"{item.model_name}: score={item.score:.1f} [{status}]")

    print_ok("Model recommendation summary generated successfully.")


if __name__ == "__main__":
    main()
