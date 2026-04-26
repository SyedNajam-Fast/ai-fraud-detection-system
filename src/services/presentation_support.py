from __future__ import annotations

import json
from typing import Any

from src.core.config import KAGGLE_CSV_PATH, MODEL_METADATA_PATH, MODEL_PATH, SAMPLE_PROFILE_CSV_PATH
from src.db import (
    get_dataset_profile_by_upload_id,
    get_feature_profiles_by_upload_id,
    get_latest_dataset_upload,
    get_latest_model_training_run,
    get_model_recommendations_by_run_id,
    get_table_row_count,
)
from src.predict import load_model_metadata
from src.services.schema_explainer import explain_database_schema


def _diagram_course_focus(diagram_id: str) -> str:
    mapping = {
        "use_case": "SDA",
        "activity": "SDA",
        "sequence": "SDA",
        "component": "SDA",
        "deployment": "SDA",
        "dfd": "AI + SDA",
        "erd": "DBS",
    }
    return mapping.get(diagram_id, "Presentation")


def _diagram_talking_points(diagram_id: str) -> list[str]:
    mapping = {
        "use_case": [
            "Use this first to explain the full user journey in one picture.",
            "It shows how dataset understanding, database design, AI training, and fraud workflow connect in the demo.",
        ],
        "activity": [
            "Use this to explain the step-by-step operational flow from upload to alert creation.",
            "It helps show that the project is a process, not just a static model file.",
        ],
        "sequence": [
            "Use this to show how the frontend, API, services, and database exchange messages.",
            "It is the strongest diagram for the SDA discussion on modularity and interaction.",
        ],
        "component": [
            "Use this to explain the software modules and the boundaries between frontend, backend, and storage.",
            "It shows that training, profiling, schema explanation, and workflow logic are separated.",
        ],
        "deployment": [
            "Use this to explain that the project is intentionally local and demo-ready on one laptop.",
            "It also helps justify why SQLite and local artifact files are reasonable for the semester scope.",
        ],
        "dfd": [
            "Use this to explain how data moves between profiling, recommendation, training, and runtime fraud detection.",
            "It links the AI pipeline with the database and presentation layers.",
        ],
        "erd": [
            "Use this when the DBS instructor asks about primary keys, foreign keys, and normalization.",
            "It reflects the live schema explanation output instead of a separate hand-drawn database story.",
        ],
    }
    return mapping.get(diagram_id, ["Use this diagram to explain the related part of the system."])


def _latest_profile_summary() -> dict[str, Any] | None:
    latest_upload = get_latest_dataset_upload()
    if latest_upload is None:
        return None

    upload_id = int(latest_upload["id"])
    dataset_profile = get_dataset_profile_by_upload_id(upload_id)
    feature_profiles = get_feature_profiles_by_upload_id(upload_id)
    warnings = []
    if dataset_profile and dataset_profile.get("warnings_json"):
        warnings = json.loads(str(dataset_profile["warnings_json"]))

    return {
        "upload": latest_upload,
        "dataset_profile": dataset_profile,
        "feature_profiles": feature_profiles,
        "warnings": warnings,
    }


def _build_diagrams(schema_mermaid: str) -> list[dict[str, Any]]:
    diagrams = [
        {
            "id": "use_case",
            "title": "Use Case Diagram",
            "description": "High-level user goals inside the semester project demo.",
            "mermaid": "\n".join(
                [
                    "flowchart LR",
                    "    User([Instructor / Student])",
                    "    Upload[Upload or choose dataset]",
                    "    Profile[Profile data and explain columns]",
                    "    ExplainDB[Explain schema, keys, and normalization]",
                    "    Recommend[Shortlist 3 models]",
                    "    Train[Train shortlisted models]",
                    "    Compare[Compare results and select winner]",
                    "    RunWorkflow[Run fraud workflow and store alert]",
                    "    Review[Review diagrams and viva notes]",
                    "    User --> Upload --> Profile --> ExplainDB --> Recommend --> Train --> Compare --> RunWorkflow --> Review",
                ]
            ),
        },
        {
            "id": "activity",
            "title": "Activity Diagram",
            "description": "End-to-end activity flow from data intake to final presentation output.",
            "mermaid": "\n".join(
                [
                    "flowchart TD",
                    "    A[Choose or upload dataset] --> B[Run dataset profiling]",
                    "    B --> C[Store profile results in SQLite]",
                    "    C --> D[Explain schema and normalization]",
                    "    D --> E[Inspect dataset characteristics]",
                    "    E --> F[Recommend top 3 models]",
                    "    F --> G[Train shortlisted models]",
                    "    G --> H[Evaluate validation and test metrics]",
                    "    H --> I[Select final winner and threshold]",
                    "    I --> J[Run transaction workflow]",
                    "    J --> K[Store prediction and fraud alert]",
                    "    K --> L[Show diagrams, summaries, and viva notes]",
                ]
            ),
        },
        {
            "id": "sequence",
            "title": "Sequence Diagram",
            "description": "Message flow across frontend, backend API, services, model layer, and database.",
            "mermaid": "\n".join(
                [
                    "sequenceDiagram",
                    "    participant UI as React Dashboard",
                    "    participant API as FastAPI",
                    "    participant Profile as Profiling Service",
                    "    participant Model as Training Service",
                    "    participant DB as SQLite",
                    "    UI->>API: Request dataset profiling",
                    "    API->>Profile: Profile CSV and explain columns",
                    "    Profile->>DB: Store raw_dataset_uploads / profiles",
                    "    UI->>API: Request schema explanation",
                    "    API->>DB: Inspect live schema",
                    "    UI->>API: Request model recommendation",
                    "    API->>Model: Analyze dataset characteristics",
                    "    Model-->>API: Return shortlisted models",
                    "    UI->>API: Start training",
                    "    API->>Model: Train shortlisted models",
                    "    Model->>DB: Store model runs, shortlist, metrics",
                    "    UI->>API: Run workflow",
                    "    API->>DB: Insert transaction / prediction / alert",
                    "    API-->>UI: Return final result",
                ]
            ),
        },
        {
            "id": "component",
            "title": "Component Diagram",
            "description": "Core software components used in the local demonstration environment.",
            "mermaid": "\n".join(
                [
                    "flowchart LR",
                    "    subgraph Frontend",
                    "        Dashboard[React dashboard]",
                    "    end",
                    "    subgraph Backend",
                    "        API[FastAPI app]",
                    "        Profiling[Profiling service]",
                    "        Schema[Schema explainer]",
                    "        Recommendation[Recommendation engine]",
                    "        Training[Training pipeline]",
                    "        Workflow[Workflow service]",
                    "    end",
                    "    subgraph Storage",
                    "        DB[(SQLite database)]",
                    "        Artifacts[(Model + metadata files)]",
                    "    end",
                    "    Dashboard --> API",
                    "    API --> Profiling",
                    "    API --> Schema",
                    "    API --> Recommendation",
                    "    API --> Training",
                    "    API --> Workflow",
                    "    Profiling --> DB",
                    "    Schema --> DB",
                    "    Recommendation --> Training",
                    "    Training --> DB",
                    "    Training --> Artifacts",
                    "    Workflow --> DB",
                    "    Workflow --> Artifacts",
                ]
            ),
        },
        {
            "id": "deployment",
            "title": "Deployment Diagram",
            "description": "Local laptop deployment view for the semester-project presentation.",
            "mermaid": "\n".join(
                [
                    "flowchart TD",
                    "    Laptop[Local presentation laptop]",
                    "    Browser[Browser at 127.0.0.1:5173]",
                    "    Frontend[React dev server]",
                    "    Backend[FastAPI server at 127.0.0.1:8000]",
                    "    Database[(SQLite file)]",
                    "    ModelFiles[(model.pkl + model_metadata.json)]",
                    "    Browser --> Frontend",
                    "    Frontend --> Backend",
                    "    Backend --> Database",
                    "    Backend --> ModelFiles",
                    "    Laptop --> Browser",
                ]
            ),
        },
        {
            "id": "dfd",
            "title": "Data Flow Diagram",
            "description": "Information movement across profiling, training, and runtime fraud detection.",
            "mermaid": "\n".join(
                [
                    "flowchart LR",
                    "    Dataset[CSV dataset] --> Profile[Dataset profiling]",
                    "    Profile --> ProfileStore[(Profile tables)]",
                    "    Profile --> Recommend[Model recommendation]",
                    "    Recommend --> Train[Shortlist training]",
                    "    Train --> Metrics[(Training audit tables)]",
                    "    Train --> SavedModel[(Saved model + metadata)]",
                    "    Transaction[Runtime transaction] --> Workflow[Fraud workflow]",
                    "    SavedModel --> Workflow",
                    "    Workflow --> PredictionStore[(Predictions + alerts)]",
                    "    Metrics --> Presentation[Presentation dashboard]",
                    "    ProfileStore --> Presentation",
                    "    PredictionStore --> Presentation",
                ]
            ),
        },
        {
            "id": "erd",
            "title": "ER Diagram",
            "description": "Database relationships taken from the live schema explanation layer.",
            "mermaid": schema_mermaid,
        },
    ]

    for diagram in diagrams:
        diagram["course_focus"] = _diagram_course_focus(str(diagram["id"]))
        diagram["talking_points"] = _diagram_talking_points(str(diagram["id"]))

    return diagrams


def _build_demo_readiness(
    latest_profile: dict[str, Any] | None,
    latest_model_metadata: dict[str, Any],
    counts: dict[str, int],
    diagram_count: int,
) -> dict[str, Any]:
    checks = [
        {
            "id": "sample_dataset",
            "label": "Sample dataset available",
            "status": "ready" if SAMPLE_PROFILE_CSV_PATH.exists() else "warning",
            "detail": str(SAMPLE_PROFILE_CSV_PATH),
        },
        {
            "id": "kaggle_dataset",
            "label": "Kaggle dataset local copy",
            "status": "ready" if KAGGLE_CSV_PATH.exists() else "warning",
            "detail": str(KAGGLE_CSV_PATH),
        },
        {
            "id": "profile_history",
            "label": "Latest profiled dataset",
            "status": "ready" if latest_profile else "warning",
            "detail": (
                f"{latest_profile['upload']['filename']} with "
                f"{latest_profile['dataset_profile']['row_count']} rows"
                if latest_profile and latest_profile.get("upload") and latest_profile.get("dataset_profile")
                else "No dataset profile has been created yet."
            ),
        },
        {
            "id": "trained_model",
            "label": "Saved model artifact and metadata",
            "status": "ready" if MODEL_PATH.exists() and MODEL_METADATA_PATH.exists() else "warning",
            "detail": latest_model_metadata.get("selected_model_name", "No model metadata is available."),
        },
        {
            "id": "audit_history",
            "label": "Training audit trail",
            "status": (
                "ready"
                if counts["model_training_runs"] > 0 and counts["model_recommendations"] > 0
                else "warning"
            ),
            "detail": (
                f"{counts['model_training_runs']} training runs, "
                f"{counts['model_recommendations']} recommendation rows."
            ),
        },
        {
            "id": "diagram_pack",
            "label": "Mermaid diagram pack",
            "status": "ready" if diagram_count >= 7 else "warning",
            "detail": f"{diagram_count} diagrams prepared for local presentation.",
        },
    ]

    warnings = [check["label"] for check in checks if check["status"] != "ready"]
    overall_status = "ready" if not warnings else "attention"
    summary = (
        "The app is presentation-ready for a local demo."
        if overall_status == "ready"
        else "The app is mostly ready, but a few demo-prep items still need attention."
    )

    return {
        "overall_status": overall_status,
        "summary": summary,
        "checks": checks,
        "warnings": warnings,
    }


def _build_markdown_report(payload: dict[str, Any]) -> str:
    readiness = payload["demo_readiness"]
    sections = payload["report_sections"]
    diagrams = payload["diagrams"]
    viva_notes = payload["viva_notes"]
    presentation_tips = payload["presentation_tips"]
    counts = payload["counts"]

    lines = [
        "# Phase 6 Presentation Pack",
        "",
        "## Demo Readiness",
        f"- Overall status: **{readiness['overall_status']}**",
        f"- Summary: {readiness['summary']}",
    ]

    for check in readiness["checks"]:
        lines.append(f"- {check['label']}: {check['status']} ({check['detail']})")

    lines.extend(
        [
            "",
            "## System Snapshot",
            f"- Transactions: {counts['transactions']}",
            f"- Predictions: {counts['predictions']}",
            f"- Fraud alerts: {counts['fraud_alerts']}",
            f"- Dataset profiles: {counts['dataset_profiles']}",
            f"- Training runs: {counts['model_training_runs']}",
            f"- Model recommendations: {counts['model_recommendations']}",
            "",
            "## Report Sections",
        ]
    )

    for section in sections:
        lines.extend(
            [
                f"### {section['title']}",
                section["technical_text"],
                "",
            ]
        )

    lines.extend(["## Demo Tips"])
    for tip in presentation_tips:
        lines.append(f"- {tip}")

    lines.extend(["", "## Viva Notes"])
    for note in viva_notes:
        lines.extend(
            [
                f"### {note['question']}",
                note["answer"],
                "",
            ]
        )

    lines.extend(["## Mermaid Diagrams"])
    for diagram in diagrams:
        lines.extend(
            [
                f"### {diagram['title']}",
                diagram["description"],
                "",
                "```mermaid",
                diagram["mermaid"],
                "```",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def build_presentation_export_bundle(export_format: str = "markdown") -> dict[str, Any]:
    payload = build_presentation_support_payload()
    normalized_format = export_format.strip().lower()

    if normalized_format == "markdown":
        return {
            "format": "markdown",
            "filename": "phase6-presentation-pack.md",
            "content_type": "text/markdown",
            "content": payload["markdown_report"],
        }
    if normalized_format == "json":
        return {
            "format": "json",
            "filename": "phase6-presentation-pack.json",
            "content_type": "application/json",
            "content": json.dumps(payload, indent=2),
        }

    raise ValueError(f"Unsupported export format: {export_format}")


def build_presentation_support_payload() -> dict[str, Any]:
    schema = explain_database_schema()
    latest_profile = _latest_profile_summary()
    latest_model_metadata = load_model_metadata()
    latest_run = get_latest_model_training_run()
    latest_recommendations = []
    if latest_run is not None:
        latest_recommendations = get_model_recommendations_by_run_id(int(latest_run["id"]))

    latest_target_column = None
    latest_dataset_rows = None
    latest_dataset_columns = None
    if latest_profile and latest_profile["dataset_profile"]:
        latest_dataset_rows = latest_profile["dataset_profile"]["row_count"]
        latest_dataset_columns = latest_profile["dataset_profile"]["column_count"]
        latest_target_column = latest_profile["dataset_profile"]["target_column"]

    selected_model_name = latest_model_metadata.get("selected_model_name", "not trained yet")
    selected_threshold = latest_model_metadata.get("selected_threshold")
    validation_f1 = latest_model_metadata.get("validation_metrics", {}).get("f1")
    test_f1 = latest_model_metadata.get("test_metrics", {}).get("f1")
    diagrams = _build_diagrams(schema.mermaid_er_diagram)

    report_sections = [
        {
            "title": "Project Overview",
            "simple_text": "This project combines data understanding, database explanation, model recommendation, fraud-model training, and transaction-level fraud workflow in one local system.",
            "technical_text": "The system integrates FastAPI, React, SQLite, and scikit-learn to support profiling, schema explanation, recommendation-based model selection, audited training runs, and prediction logging.",
        },
        {
            "title": "Dataset Summary",
            "simple_text": f"The latest profiled dataset has {latest_dataset_rows or 'unknown'} rows, {latest_dataset_columns or 'unknown'} columns, and target column `{latest_target_column or 'not detected'}`.",
            "technical_text": "The profiling layer stores raw file metadata, dataset-level summary metrics, and feature-level explanations inside `raw_dataset_uploads`, `dataset_profiles`, and `feature_profiles`.",
        },
        {
            "title": "Database Design Summary",
            "simple_text": "The database is split into operational tables, raw data tables, and analytics tables so each part of the system is easier to explain.",
            "technical_text": "Operational tables hold live fraud workflow records, raw tables preserve imported data, and analytics tables capture dataset profiles plus model audit history.",
        },
        {
            "title": "Model Summary",
            "simple_text": f"The current winning model is `{selected_model_name}` with threshold {selected_threshold if selected_threshold is not None else 'not available'}.",
            "technical_text": f"The latest run selected `{selected_model_name}` after rule-based top-3 recommendation. Validation F1={validation_f1 if validation_f1 is not None else 'N/A'}, Test F1={test_f1 if test_f1 is not None else 'N/A'}.",
        },
        {
            "title": "Demo Script",
            "simple_text": "Open the dashboard, profile a dataset, show schema and diagrams, recommend models, train the shortlist, run the fraud workflow, and finish with viva notes.",
            "technical_text": "The recommended demo order is dashboard -> data understanding -> schema explanation -> recommendation -> training -> workflow -> presentation tab.",
        },
    ]

    viva_notes = [
        {
            "question": "Why did you normalize this database design?",
            "answer": "I separated users, transactions, predictions, alerts, profiling summaries, and model history so repeated information would not be stored in one large table.",
        },
        {
            "question": "Why are location and merchant still text columns in transactions?",
            "answer": "For the current project scope I kept them simple to reduce implementation overhead, but they can be moved into lookup tables later for stronger normalization.",
        },
        {
            "question": "How does the system choose only 3 models out of 10?",
            "answer": "It uses dataset characteristics such as sample size, imbalance, feature mix, and encoded feature estimate to score the full model pool and shortlist the top three candidates.",
        },
        {
            "question": "Why do you tune the threshold instead of using 0.5?",
            "answer": "Fraud data is usually imbalanced, so the validation set is used to find a threshold that gives better fraud-class performance than a fixed 0.5 cutoff.",
        },
        {
            "question": "How do the database and AI parts connect?",
            "answer": "The database stores profiled datasets, transactions, predictions, alerts, training runs, shortlist recommendations, and candidate metrics, while the AI layer reads training data and writes audited results back into SQLite.",
        },
        {
            "question": "Why is the project suitable for SDA?",
            "answer": "Because it is a modular system with a frontend, backend API, service layer, database layer, model layer, and documented diagrams showing how all parts interact.",
        },
    ]

    presentation_tips = [
        "Start from the dashboard so the examiners see the whole system before details.",
        "Use Simple Mode first, then switch to Technical Mode when an instructor asks for depth.",
        "Profile the sample dataset if Kaggle data is not available locally.",
        "Show the ER diagram and then the sequence diagram to connect DBS and SDA discussion.",
        "Train the shortlist once before the demo if you want faster live interaction, then use workflow actions during the presentation.",
    ]

    counts = {
        "transactions": get_table_row_count("transactions"),
        "predictions": get_table_row_count("predictions"),
        "fraud_alerts": get_table_row_count("fraud_alerts"),
        "dataset_profiles": get_table_row_count("dataset_profiles"),
        "model_training_runs": get_table_row_count("model_training_runs"),
        "model_recommendations": get_table_row_count("model_recommendations"),
    }
    demo_readiness = _build_demo_readiness(
        latest_profile=latest_profile,
        latest_model_metadata=latest_model_metadata,
        counts=counts,
        diagram_count=len(diagrams),
    )

    payload = {
        "diagrams": diagrams,
        "report_sections": report_sections,
        "viva_notes": viva_notes,
        "presentation_tips": presentation_tips,
        "counts": counts,
        "demo_readiness": demo_readiness,
        "latest_profile": latest_profile,
        "latest_model_metadata": latest_model_metadata,
        "latest_model_recommendations": latest_recommendations,
    }
    payload["markdown_report"] = _build_markdown_report(payload)
    return payload
