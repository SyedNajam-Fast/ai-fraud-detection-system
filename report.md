# Project Report

## Executive Summary

This repository now implements a broader academic fraud-detection platform than the original backend-only demo. The current system combines:

1. A **SQLite database layer** for transactions, predictions, fraud alerts, dataset profiling history, and model audit history.
2. A **machine-learning layer** that recommends a shortlist of three models from a pool of ten candidates, trains the shortlisted models, tunes the decision threshold, and saves the final winner.
3. A **FastAPI backend** that exposes profiling, schema explanation, model, prediction, and workflow endpoints.
4. A **React/Vite frontend** designed for AI, DBS, and SDA presentation, with offline fallback support when the backend is not running.

The project is a local, presentation-oriented semester system rather than a production banking platform.

## Current Architecture

The implemented architecture is organized into five layers:

- **Frontend layer**: `frontend/src/App.jsx` presents three academic tabs: AI, DB, and SDA.
- **API layer**: `src/api/app.py` provides REST endpoints under `/api`.
- **Service layer**: `src/services/` contains workflow, dataset profiling, schema explanation, AI demo, model recommendation, and presentation support logic.
- **Persistence layer**: `src/db.py` manages SQLite access and schema-driven table initialization.
- **Model layer**: `model/train_model.py` performs data loading, model recommendation, training, threshold tuning, evaluation, artifact persistence, and DB audit logging.

## Implemented Functional Coverage

### 1. Dataset understanding

The system can profile a CSV dataset from a file path or uploaded file. It records:

- row count and column count,
- duplicate rows and missing cells,
- inferred target candidates,
- inferred column role and simplified data type,
- class distribution and class imbalance,
- plain-language and technical explanations for each column.

These results are stored in:

- `raw_dataset_uploads`
- `dataset_profiles`
- `feature_profiles`

### 2. Database explanation

The schema explanation service reads the live SQLite database and explains:

- table purpose,
- primary keys and foreign keys,
- indexes,
- schema layers,
- normalization story,
- Mermaid ER diagram output.

This makes the DBS discussion dynamic instead of static.

### 3. Model recommendation and training

The AI pipeline now:

- evaluates dataset characteristics first,
- scores a **10-model candidate pool** using rule-based heuristics,
- selects a **top-3 shortlist** for actual training,
- compares shortlisted models on validation metrics,
- tunes the fraud decision threshold from validation probabilities,
- checks overfit and underfit heuristically,
- saves the final pipeline and metadata,
- writes training history and recommendation history to SQLite.

### 4. Prediction and runtime workflow

The system supports:

- manual prediction from the frontend or API,
- held-out test-sample prediction for live demo,
- end-to-end transaction insertion -> prediction -> fraud alert creation through `src/main.py` and `src/services/workflow.py`.

### 5. Presentation support

The project includes a presentation-support layer that prepares:

- Mermaid diagrams,
- viva notes,
- report sections,
- readiness checks,
- export bundles in Markdown or JSON.

The frontend can still present useful content in offline mode using built-in demo data.

## Database Design Summary

The live schema currently contains **11 application tables** grouped into four layers:

- **Raw training layer**: `kaggle_transactions`
- **Raw profiling layer**: `raw_dataset_uploads`
- **Operational layer**: `users`, `transactions`, `predictions`, `fraud_alerts`
- **Analytics and audit layer**: `dataset_profiles`, `feature_profiles`, `model_training_runs`, `model_recommendations`, `model_candidate_metrics`

Key design strengths:

- foreign-key relationships are enforced,
- major child tables use `ON DELETE CASCADE`,
- validation constraints protect core fields,
- indexes support lookup and audit queries,
- runtime data, profiling data, and model history are separated cleanly.

The operational design is mostly close to 3NF, but `location` and `merchant` remain text fields in `transactions` for simplicity.

## AI Pipeline Summary

### Training data priority

The training pipeline uses this order:

1. `kaggle_transactions` from SQLite
2. `data/fraud_transactions.csv`
3. synthetic generated fallback data

### Runtime feature schema

The final training and inference schema uses:

- `amount`
- `time`
- `location`
- `merchant`

Target:

- `fraud`

### Candidate pool

The full recommendation pool includes ten models:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- AdaBoost
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes

Only the top three recommended models are trained in the main comparison stage.

### Current saved model state

Based on the current local metadata:

- dataset source: `synthetic:generated`
- selected model: `logistic_regression`
- shortlist size: `3`
- full model pool size: `10`
- selected threshold: `0.6762725380991143`
- validation F1: `0.6336633663366337`
- test F1: `0.48484848484848486`

This means the current local environment is functional, but it is still using synthetic fallback data because no Kaggle CSV is currently available locally.

## Current Verified Local State

At the time of this documentation update, the local repository state includes:

- `data/fraud_detection.db` exists
- `model/model.pkl` exists
- `model/model_metadata.json` exists
- `data/samples/sample_profile_dataset.csv` exists
- `data/raw/creditcardfraud/creditcard.csv` is **not** currently present locally
- frontend build output exists in `frontend/dist/`

Current database row counts:

- `users`: 1
- `transactions`: 22
- `predictions`: 22
- `fraud_alerts`: 22
- `kaggle_transactions`: 0
- `raw_dataset_uploads`: 7
- `dataset_profiles`: 7
- `feature_profiles`: 70
- `model_training_runs`: 10
- `model_recommendations`: 18
- `model_candidate_metrics`: 30

## Strengths

- Stronger than the original backend-only concept
- Clear modular separation across frontend, API, services, database, and model logic
- Good academic traceability through profiling tables and model audit tables
- Live schema explanation and Mermaid ER generation
- Local demo resilience through frontend offline mode
- Useful AI presentation flow: dataset -> shortlist -> training -> prediction -> test sample

## Current Limitations

- No authentication or role-based access control
- Local deployment only
- No production security model
- Kaggle feature mapping into `location` and `merchant` is heuristic rather than semantically real
- Frontend does not yet expose every backend action, such as full dataset upload/profile workflow
- Runtime transaction workflow still uses a deterministic demo transaction in `src/insert_data.py`

## Conclusion

The project has evolved from a simple database-plus-model demo into a multi-layer academic platform that supports dataset understanding, schema explanation, shortlist-based model training, prediction, alert logging, and viva-focused presentation. Its current implementation is strong for semester evaluation because it connects DBS, AI, and SDA concerns in one locally runnable system while still remaining honest about its scope and limitations.
