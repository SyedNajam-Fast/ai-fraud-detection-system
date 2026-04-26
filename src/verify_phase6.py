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

    from src.db import initialize_database
    from src.services.presentation_support import (
        build_presentation_export_bundle,
        build_presentation_support_payload,
    )

    print_section("Phase 6 Verification")
    initialize_database()

    payload = build_presentation_support_payload()
    diagrams = payload["diagrams"]
    report_sections = payload["report_sections"]
    viva_notes = payload["viva_notes"]
    demo_readiness = payload["demo_readiness"]
    markdown_report = payload["markdown_report"]

    if len(diagrams) < 7:
        raise SystemExit(f"Expected at least 7 diagrams, found {len(diagrams)}.")
    if any(not item["mermaid"].strip() for item in diagrams):
        raise SystemExit("One or more diagrams has empty Mermaid content.")
    if any(len(item.get("talking_points", [])) == 0 for item in diagrams):
        raise SystemExit("One or more diagrams is missing presenter talking points.")
    print_ok("Presentation payload contains the expected Mermaid diagrams.")

    if len(report_sections) < 4:
        raise SystemExit(f"Expected at least 4 report sections, found {len(report_sections)}.")
    if len(viva_notes) < 5:
        raise SystemExit(f"Expected at least 5 viva notes, found {len(viva_notes)}.")
    if len(demo_readiness.get("checks", [])) < 4:
        raise SystemExit("Expected presentation readiness checks to be present.")
    print_ok("Report sections and viva notes are present.")

    if "## Mermaid Diagrams" not in markdown_report or "```mermaid" not in markdown_report:
        raise SystemExit("Markdown export is missing Mermaid report content.")

    markdown_export = build_presentation_export_bundle("markdown")
    json_export = build_presentation_export_bundle("json")
    if markdown_export["format"] != "markdown" or not markdown_export["content"].strip():
        raise SystemExit("Markdown export bundle is invalid.")
    if json_export["format"] != "json" or not json_export["content"].strip():
        raise SystemExit("JSON export bundle is invalid.")
    print_ok("Presentation export bundles are available.")

    print_info(f"Diagrams: {len(diagrams)}")
    print_info(f"Report sections: {len(report_sections)}")
    print_info(f"Viva notes: {len(viva_notes)}")
    print_info(f"Readiness checks: {len(demo_readiness['checks'])}")
    print_ok("Phase 6 verification completed successfully.")


if __name__ == "__main__":
    main()
