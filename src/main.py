from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import REQUIRED_PACKAGES, REQUIREMENTS_PATH, ensure_project_root_on_path
from src.core.console import print_info, print_ok, print_section
from src.services.workflow import run_workflow


def _missing_packages() -> list[str]:
	missing = []
	for package_name in REQUIRED_PACKAGES:
		if importlib.util.find_spec(package_name) is None:
			missing.append(package_name)
	return missing


def _install_requirements_if_needed() -> None:
	missing_packages = _missing_packages()
	if not missing_packages:
		return

	print_info("Missing dependencies detected. Installing requirements from requirements.txt...")
	install_command = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
	result = subprocess.run(install_command, check=False)
	if result.returncode != 0:
		raise SystemExit(f"Dependency installation failed with exit code {result.returncode}.")

	remaining_missing = _missing_packages()
	if remaining_missing:
		raise SystemExit(
			"Dependency installation completed, but these packages are still missing: "
			+ ", ".join(remaining_missing)
		)
	print_ok("Dependencies are ready.")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run end-to-end fraud detection workflow.")
	parser.add_argument(
		"--force-train",
		action="store_true",
		help="Retrain and overwrite the model before processing a transaction.",
	)
	return parser.parse_args()


def _print_training_summary(model_metrics: dict[str, object] | None) -> None:
	if model_metrics is None:
		from src.predict import load_model_metadata

		print_info("Using existing trained model artifact.")
		metadata = load_model_metadata()
		if metadata:
			print_info(f"Selected model: {metadata.get('selected_model_name', 'unknown')}")
			print_info(f"Decision threshold: {float(metadata.get('selected_threshold', 0.5)):.2f}")
			shortlisted_models = metadata.get("shortlisted_models", [])
			if shortlisted_models:
				print_info(
					"Shortlisted models: "
					+ ", ".join(str(item.get("model_name", "unknown")) for item in shortlisted_models)
				)
		return

	print_section("Training Summary")
	print_info(f"Dataset source: {model_metrics['dataset_source']}")
	print_info(f"Samples used: {model_metrics['sample_count']}")
	print_info(f"Selected model: {model_metrics['selected_model_name']}")
	print_info(f"Selection metric: {model_metrics['selection_metric']}")
	print_info(f"Decision threshold: {model_metrics['selected_threshold']:.2f}")
	print_info(f"Recommendation strategy: {model_metrics['recommendation_strategy']}")
	print_info(
		"Shortlisted models: "
		+ ", ".join(str(item["model_name"]) for item in model_metrics["shortlisted_models"])
	)
	print_info(f"Validation F1: {model_metrics['validation_metrics']['f1']:.4f}")
	print_info(f"Validation PR AUC: {model_metrics['validation_metrics']['average_precision']:.4f}")
	print_info(f"Validation ROC AUC: {model_metrics['validation_metrics']['roc_auc']:.4f}")
	print_info(f"Test F1: {model_metrics['test_metrics']['f1']:.4f}")
	print_info(f"Test PR AUC: {model_metrics['test_metrics']['average_precision']:.4f}")
	print_info(f"Test ROC AUC: {model_metrics['test_metrics']['roc_auc']:.4f}")
	print_info(f"Overfit check: {'flagged' if model_metrics['overfit_flag'] else 'passed'}")
	print_info(f"Underfit check: {'flagged' if model_metrics['underfit_flag'] else 'passed'}")
	print_info(f"Accuracy: {model_metrics['accuracy']:.4f}")
	print("Confusion matrix:")
	print(model_metrics["confusion_matrix"])
	for item in model_metrics["shortlisted_models"]:
		print_info(
			f"Reason for {item['model_name']}: {item['rationale']}"
		)


def _print_workflow_summary(result) -> None:
	print_section("Workflow Summary")
	print_info(f"Model artifact: {result.model_path}")
	print_info(f"Metadata file: {result.model_metadata_path}")
	print_info(f"Transaction ID: {result.transaction_id}")
	print_info(f"Prediction: {result.prediction}")
	print_info(f"Fraud probability: {result.probability:.4f}")
	if result.alert_id is not None:
		print_ok(f"Fraud alert created with ID: {result.alert_id}")
	else:
		print_info("No fraud alert was created for this transaction.")


if __name__ == "__main__":
	ensure_project_root_on_path()
	arguments = parse_args()
	_install_requirements_if_needed()
	result = run_workflow(force_train=arguments.force_train)
	_print_training_summary(result.model_metrics)
	_print_workflow_summary(result)
