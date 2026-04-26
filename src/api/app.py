from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import shutil
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import KAGGLE_CSV_PATH, SAMPLE_PROFILE_CSV_PATH, UPLOADS_DIR, ensure_project_root_on_path


ensure_project_root_on_path()

from model.train_model import recommend_models_for_current_dataset, train_and_save_model  # noqa: E402
from src.db import (  # noqa: E402
    get_dataset_profile_by_upload_id,
    get_feature_profiles_by_upload_id,
    get_latest_dataset_upload,
    get_latest_model_training_run,
    get_model_recommendations_by_run_id,
    get_table_row_count,
    initialize_database,
)
from src.predict import load_model_metadata  # noqa: E402
from src.services.dataset_profiling import profile_csv_dataset  # noqa: E402
from src.services.presentation_support import (  # noqa: E402
    build_presentation_export_bundle,
    build_presentation_support_payload,
)
from src.services.schema_explainer import explain_database_schema  # noqa: E402
from src.services.workflow import run_workflow  # noqa: E402


class ProfilePathRequest(BaseModel):
    csv_path: str
    target_column: str | None = None


class WorkflowRequest(BaseModel):
    force_train: bool = False


app = FastAPI(title="Fraud Detection Project API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _latest_profile_payload() -> dict[str, Any] | None:
    latest_upload = get_latest_dataset_upload()
    if latest_upload is None:
        return None

    upload_id = int(latest_upload["id"])
    dataset_profile = get_dataset_profile_by_upload_id(upload_id)
    feature_profiles = get_feature_profiles_by_upload_id(upload_id)
    return {
        "upload": latest_upload,
        "dataset_profile": dataset_profile,
        "feature_profiles": feature_profiles,
    }


def _latest_model_payload() -> dict[str, Any]:
    metadata = load_model_metadata()
    latest_run = get_latest_model_training_run()
    recommendations = []
    if latest_run is not None:
        recommendations = get_model_recommendations_by_run_id(int(latest_run["id"]))

    return {
        "metadata": metadata,
        "latest_run": latest_run,
        "recommendations": recommendations,
    }


def _dashboard_payload() -> dict[str, Any]:
    counts = {
        "users": get_table_row_count("users"),
        "transactions": get_table_row_count("transactions"),
        "predictions": get_table_row_count("predictions"),
        "fraud_alerts": get_table_row_count("fraud_alerts"),
        "raw_dataset_uploads": get_table_row_count("raw_dataset_uploads"),
        "dataset_profiles": get_table_row_count("dataset_profiles"),
        "feature_profiles": get_table_row_count("feature_profiles"),
        "model_training_runs": get_table_row_count("model_training_runs"),
        "model_recommendations": get_table_row_count("model_recommendations"),
        "model_candidate_metrics": get_table_row_count("model_candidate_metrics"),
    }
    return {
        "counts": counts,
        "latest_profile": _latest_profile_payload(),
        "latest_model": _latest_model_payload(),
    }


@app.on_event("startup")
def startup() -> None:
    initialize_database()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return _serialize(_dashboard_payload())


@app.get("/api/datasets/options")
def dataset_options() -> dict[str, Any]:
    return {
        "datasets": [
            {
                "label": "Sample Profile Dataset",
                "path": str(SAMPLE_PROFILE_CSV_PATH),
                "available": SAMPLE_PROFILE_CSV_PATH.exists(),
                "defaultTargetColumn": "Class",
            },
            {
                "label": "Kaggle Fraud Dataset",
                "path": str(KAGGLE_CSV_PATH),
                "available": KAGGLE_CSV_PATH.exists(),
                "defaultTargetColumn": "Class",
            },
        ]
    }


@app.get("/api/profiles/latest")
def latest_profile() -> dict[str, Any]:
    return {"profile": _serialize(_latest_profile_payload())}


@app.post("/api/profile/path")
def profile_path(request: ProfilePathRequest) -> dict[str, Any]:
    csv_path = Path(request.csv_path)
    try:
        result = profile_csv_dataset(csv_path, target_column=request.target_column)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"profile": _serialize(result), "dashboard": _serialize(_dashboard_payload())}


@app.post("/api/profile/upload")
async def profile_upload(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = UPLOADS_DIR / Path(file.filename).name
    try:
        with destination_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
        result = profile_csv_dataset(destination_path, target_column=target_column)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        file.file.close()

    return {"profile": _serialize(result), "dashboard": _serialize(_dashboard_payload())}


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    return {"schema": _serialize(explain_database_schema())}


@app.get("/api/presentation")
def presentation() -> dict[str, Any]:
    return {"presentation": _serialize(build_presentation_support_payload())}


@app.get("/api/presentation/export")
def presentation_export(export_format: str = Query(default="markdown", alias="format")) -> dict[str, Any]:
    try:
        payload = build_presentation_export_bundle(export_format)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"export": _serialize(payload)}


@app.get("/api/recommendations/current")
def recommendations_current() -> dict[str, Any]:
    try:
        result = recommend_models_for_current_dataset()
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _serialize(result)


@app.get("/api/model/latest")
def latest_model() -> dict[str, Any]:
    return _serialize(_latest_model_payload())


@app.post("/api/train")
def train() -> dict[str, Any]:
    try:
        result = train_and_save_model()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"training": _serialize(result), "dashboard": _serialize(_dashboard_payload())}


@app.post("/api/workflow/run")
def workflow_run(request: WorkflowRequest) -> dict[str, Any]:
    try:
        result = run_workflow(force_train=request.force_train)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"workflow": _serialize(result), "dashboard": _serialize(_dashboard_payload())}
