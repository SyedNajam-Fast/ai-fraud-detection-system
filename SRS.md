# Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose

This document defines the software requirements for the current repository implementation of the **AI-Powered Fraud Detection and Transaction Management System**. It is written for academic use and aligns with the codebase as it exists today: a local fraud-detection platform that combines dataset profiling, database explanation, model recommendation, machine-learning training, prediction, alert logging, and presentation support.

### 1.2 Scope

The system is a local, single-machine application that:

- stores fraud-related operational and analytical data in SQLite,
- profiles CSV datasets and persists dataset-level plus feature-level summaries,
- explains the live database schema, keys, indexes, and normalization story,
- recommends a shortlist of three candidate models from a ten-model pool,
- trains shortlisted models and selects the final winner using validation metrics,
- performs fraud prediction for manual or workflow-generated transactions,
- stores predictions, fraud alerts, and model audit history,
- exposes a FastAPI backend for project actions,
- provides a React/Vite presentation UI with AI, DB, and SDA views,
- supports an offline frontend presentation mode when the backend is unavailable.

The system is **not** a production banking platform. It is a semester-project system focused on explainability, traceability, and demo readiness.

### 1.3 Intended Audience

This SRS is intended for:

- student developers maintaining or extending the repository,
- course instructors evaluating the DBS, AI, and SDA aspects,
- teammates who need a clear view of system boundaries and implemented requirements.

### 1.4 Definitions and Acronyms

| Term | Meaning |
| --- | --- |
| AI | Artificial Intelligence / machine-learning layer |
| DBS | Database Systems course perspective |
| SDA | Software Design and Analysis course perspective |
| API | Application Programming Interface exposed by FastAPI |
| PR AUC | Average precision over the precision-recall curve |
| ROC AUC | Area under the ROC curve |
| Offline mode | Frontend mode that uses local demo snapshots instead of live backend data |
| Shortlist | Top 3 recommended models selected from the full 10-model pool |

### 1.5 Source Basis

This SRS is based on direct inspection of the repository structure, schema, backend services, frontend code, model pipeline, startup scripts, and helper utilities. The schema explanation and presentation-support layers were also verified through the local verification scripts.

## 2. Overall Description

### 2.1 Product Perspective

The product is composed of five cooperating layers:

1. **Frontend presentation layer**  
   React + Vite UI in `frontend/` with three academic tabs: AI, DB, and SDA.
2. **API layer**  
   FastAPI application in `src/api/app.py`.
3. **Service layer**  
   Workflow, dataset profiling, schema explanation, model recommendation, AI demo, and presentation support services under `src/services/`.
4. **Persistence layer**  
   SQLite database initialized from `database/schema.sql`.
5. **Model/artifact layer**  
   Training pipeline in `model/train_model.py` and saved artifacts in `model/model.pkl` and `model/model_metadata.json`.

### 2.2 Product Goals

The system shall support three academic goals at the same time:

- explain how fraud-related source data is understood and profiled,
- explain how the relational schema is structured and normalized,
- explain how model recommendation, training, evaluation, and prediction work end to end.

### 2.3 User Classes

| User Class | Description | Expected Interaction |
| --- | --- | --- |
| Presenter / Student | Main operator of the system during demo or viva | Runs backend/frontend, triggers training, shows diagrams, performs predictions |
| Instructor / Examiner | Observer who asks academic questions | Reviews UI output, diagrams, schema, metrics, and explanations |
| Developer / Teammate | Maintains and extends the project | Uses CLI scripts, API, codebase, and database schema |

The current system does not implement login, role-based permissions, or separate account-specific workflows.

### 2.4 Operating Environment

The system is designed for a local development or presentation machine with:

- Python runtime for backend, ML, and CLI utilities,
- Node.js and npm for the frontend,
- SQLite as the local embedded database,
- a modern browser for the React frontend,
- Windows-friendly launch scripts (`start_backend.bat`, `start_backend.ps1`).

### 2.5 Constraints

- The database engine shall be SQLite.
- The backend shall run locally on `127.0.0.1:8000` by default.
- The frontend dev server shall run locally on `127.0.0.1:5173` by default.
- The primary fraud model input schema shall remain limited to `amount`, `time`, `location`, and `merchant`.
- The current project is single-user and local; there is no authentication or remote multi-user coordination.
- The Kaggle dataset mapping into the project schema is heuristic and presentation-oriented, not a real semantic reconstruction of the anonymized PCA features.

### 2.6 Assumptions and Dependencies

- Python dependencies are installed from `requirements.txt`.
- Frontend dependencies are installed from `frontend/package.json`.
- Kaggle download support requires valid Kaggle credentials if the dataset is downloaded.
- The backend should be started before the frontend when live mode is required.
- If real training data is unavailable, the system may fall back to synthetic data to preserve end-to-end operability.

## 3. External Interface Requirements

### 3.1 User Interface Requirements

The frontend shall provide the following subject-wise views:

| View | Purpose | Live Actions |
| --- | --- | --- |
| AI | Dataset understanding, model recommendation, training, manual prediction, held-out test prediction | Train model, predict manual transaction, predict next test sample |
| DB | Explain source dataset profile, table separation, normalization, and ER diagram | Refresh backend-provided schema/presentation data |
| SDA | Explain methodology, architecture, requirements, testing, and diagrams | View built-in presentation notes and diagrams |

Additional UI behavior:

- The UI shall display a clear message when offline presentation mode is active.
- The UI shall continue working with local demo snapshots if the backend cannot be reached.
- The UI shall show diagrams through Mermaid rendering.
- The manual-prediction form shall restrict input to structured fields rather than free-form text.

### 3.2 CLI Interface Requirements

The repository shall expose command-line entry points for major workflows:

| Script | Purpose |
| --- | --- |
| `src/main.py` | Run the end-to-end transaction-to-prediction workflow |
| `src/run_api.py` | Start the FastAPI backend |
| `src/download_dataset.py` | Download the Kaggle fraud dataset into the project data directory |
| `src/import_kaggle_to_db.py` | Import Kaggle CSV rows into `kaggle_transactions` |
| `src/profile_dataset.py` | Profile a CSV dataset and persist profiling results |
| `src/recommend_models.py` | Show the current dataset-driven shortlist |
| `src/explain_database.py` | Print schema, relationship, and normalization explanations |

### 3.3 API Interface Requirements

The backend shall expose REST endpoints under `/api`:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `GET` | `/dashboard` | Dashboard counts plus latest profile/model summary |
| `GET` | `/datasets/options` | Known local dataset choices |
| `GET` | `/profiles/latest` | Latest stored dataset profile |
| `POST` | `/profile/path` | Profile a CSV by path |
| `POST` | `/profile/upload` | Upload and profile a CSV file |
| `GET` | `/schema` | Live schema explanation payload |
| `GET` | `/presentation` | Presentation-support payload |
| `GET` | `/presentation/export` | Export presentation bundle as Markdown or JSON |
| `GET` | `/recommendations/current` | Current shortlist recommendation |
| `GET` | `/ai/dataset-preview` | Preview current training dataset |
| `POST` | `/predict/manual` | Predict a manual transaction |
| `GET` | `/predict/test-sample` | Predict a held-out sample from the test split |
| `GET` | `/model/latest` | Latest model metadata and latest DB run |
| `POST` | `/train` | Train the shortlisted models and persist the winner |
| `POST` | `/workflow/run` | Run the full end-to-end fraud workflow |

### 3.4 Software Interfaces

The system depends on:

- `pandas`, `numpy`, `scikit-learn`, `joblib`,
- `fastapi`, `uvicorn`, `python-multipart`,
- `react`, `react-dom`, `vite`, `lucide-react`, `mermaid`,
- `kagglehub` for optional dataset acquisition.

### 3.5 Communication Interfaces

- Frontend-backend communication shall occur over local HTTP.
- CORS shall allow localhost/127.0.0.1 development origins on ports `5173` and `5174`.
- No message queue, websocket channel, or external broker is required.

## 4. Functional Requirements

### FR-1. Database Initialization

- The system shall create all required application tables from `database/schema.sql` if they do not exist.
- The system shall enable SQLite foreign-key enforcement for application connections.
- The system shall create required data directories before writing the database or uploads.

### FR-2. Operational Fraud Workflow

- The system shall insert or reuse a demo user.
- The system shall insert a transaction containing `amount`, `time`, `location`, and `merchant`.
- The system shall fetch the inserted transaction from SQLite before prediction.
- The system shall predict the fraud class and probability for that transaction.
- The system shall store the prediction in the `predictions` table.
- The system shall create a row in `fraud_alerts` when the prediction result is fraud.

### FR-3. Dataset Acquisition and Import

- The system shall support downloading `mlg-ulb/creditcardfraud` through `kagglehub`.
- The system shall copy the downloaded `creditcard.csv` into `data/raw/creditcardfraud/creditcard.csv`.
- The system shall validate Kaggle CSV headers before import.
- The system shall import Kaggle rows into `kaggle_transactions` in batches.
- The import utility shall support both replace and append modes.

### FR-4. Dataset Profiling

- The system shall accept a CSV file path or uploaded CSV file for profiling.
- The system shall compute row count, column count, duplicate rows, and missing-cell count.
- The system shall infer likely target columns.
- The system shall infer column roles such as target, identifier, time, amount, and feature.
- The system shall infer simplified data types such as numeric, categorical, binary, and datetime.
- The system shall generate plain-language and technical descriptions for each column.
- The system shall compute class distribution and imbalance ratio when a target column is known.
- The system shall store upload metadata, dataset summary, and feature-level profiles in SQLite.

### FR-5. Database Explanation

- The system shall inspect the live SQLite schema rather than using a hardcoded schema description.
- The system shall classify tables into layers such as raw training, raw profiling, operational, and analytics.
- The system shall explain table purpose, primary keys, foreign keys, indexes, and normalization notes.
- The system shall generate a Mermaid ER diagram from the live schema metadata.

### FR-6. Model Recommendation

- The system shall analyze the current training dataset before training starts.
- The system shall derive dataset characteristics including sample count, feature mix, missingness, encoded feature estimate, and class imbalance.
- The system shall score a pool of ten candidate models using rule-based heuristics.
- The system shall shortlist exactly three candidate models for actual training.
- The system shall provide rationale text for each shortlisted model.

### FR-7. Training Data Selection

- The training pipeline shall use data from `kaggle_transactions` when available and valid.
- If database training data is not available, the pipeline shall check for `data/fraud_transactions.csv`.
- If neither source is available, the pipeline shall generate a synthetic dataset.
- The final training dataset shall expose the project schema: `amount`, `time`, `location`, `merchant`, `fraud`.

### FR-8. Model Training and Selection

- The training pipeline shall preprocess numeric and categorical features using a scikit-learn pipeline.
- Numeric preprocessing shall include median imputation and standard scaling.
- Categorical preprocessing shall include most-frequent imputation and one-hot encoding.
- The pipeline shall train the three shortlisted models.
- The pipeline shall split data into train, validation, and test subsets.
- The pipeline shall select a decision threshold from validation probabilities rather than using a fixed `0.5` cutoff.
- The winning model shall be selected primarily by validation average precision, with validation F1 and recall as tiebreakers.
- The pipeline shall compute train, validation, and test metrics including accuracy, precision, recall, F1, PR AUC, ROC AUC, and confusion matrix.
- The pipeline shall flag heuristic overfitting and underfitting conditions.

### FR-9. Model Persistence and Audit Trail

- The system shall save the final trained model to `model/model.pkl`.
- The system shall save detailed model metadata to `model/model_metadata.json`.
- The system shall persist training-run summaries in `model_training_runs`.
- The system shall persist candidate metrics in `model_candidate_metrics`.
- The system shall persist shortlist recommendations in `model_recommendations`.

### FR-10. Prediction Services

- The system shall support manual fraud prediction for a user-supplied transaction.
- Manual prediction input shall validate that amount is non-negative, time is between `0` and `23`, and location/merchant are non-empty.
- The system shall return prediction label, probability, threshold, confidence band, and plain-language message.
- The system shall return rule-based risk-signal text for manual predictions.
- The system shall support held-out sample prediction from the test split for demo purposes.

### FR-11. Presentation Support

- The system shall generate a presentation payload containing diagrams, report sections, viva notes, counts, and readiness checks.
- The system shall export the presentation payload as Markdown or JSON.
- The system shall provide at least the following diagram types: use case, activity, sequence, component, deployment, DFD, and ER diagram.

### FR-12. Frontend Offline Mode

- The frontend shall check backend health before relying on live API data.
- If the backend is unavailable, the frontend shall load local demo snapshots for AI and DB content.
- The SDA tab shall remain presentation-ready through built-in content even without the backend.

### FR-13. Dependency Bootstrap

- The CLI workflow entry point shall detect missing Python dependencies.
- If dependencies are missing, the workflow shall attempt installation from `requirements.txt`.
- The API launcher shall perform similar dependency checks for backend packages.

## 5. Data Requirements

### 5.1 Primary Persistent Files

| Path | Purpose |
| --- | --- |
| `data/fraud_detection.db` | Main SQLite database |
| `model/model.pkl` | Saved trained model pipeline |
| `model/model_metadata.json` | Selected threshold, metrics, shortlist, and dataset characteristics |
| `data/raw/creditcardfraud/creditcard.csv` | Optional local copy of Kaggle dataset |
| `data/uploads/` | Uploaded CSV files for profiling |
| `data/samples/sample_profile_dataset.csv` | Sample profiling dataset |

### 5.2 Database Tables

| Table | Role | Key Relationships |
| --- | --- | --- |
| `users` | Cardholder master data | Parent of `transactions` |
| `transactions` | Runtime transaction facts | FK to `users` |
| `predictions` | Stored model decisions | FK to `transactions` |
| `fraud_alerts` | Alerts for suspicious transactions | FK to `transactions` |
| `kaggle_transactions` | Imported raw training rows | Standalone raw table |
| `raw_dataset_uploads` | Uploaded-file registry | Parent of dataset/profile analytics |
| `dataset_profiles` | Dataset-level profile summary | FK to `raw_dataset_uploads` |
| `feature_profiles` | Column-level profile records | FK to both `raw_dataset_uploads` and `dataset_profiles` |
| `model_training_runs` | Training-run summary | Parent of recommendations and candidate metrics |
| `model_recommendations` | Stored top-3 shortlist | FK to `model_training_runs` |
| `model_candidate_metrics` | Metrics for compared models | FK to `model_training_runs` |

### 5.3 Data Integrity Rules

- `users.email` and `users.card_number` shall be unique.
- `transactions.amount` shall be greater than or equal to `0`.
- `transactions.time` shall be between `0` and `23`.
- `predictions.prediction` shall be `0` or `1`.
- `predictions.probability` shall be between `0` and `1`.
- `fraud_alerts.status` shall be one of `open`, `investigating`, `resolved`, or `dismissed`.
- `kaggle_transactions.class_label` shall be `0` or `1`.
- Child records shall cascade on delete from their parent entities where defined in the schema.

### 5.4 Indexing Requirements

The schema shall index the major foreign-key and analytics lookup paths, including:

- transactions by `user_id`,
- predictions by `transaction_id`,
- fraud alerts by `transaction_id`,
- Kaggle rows by `class_label` and `source_file`,
- training runs by `dataset_source` and `selected_model_name`,
- candidate metrics and recommendations by `run_id`,
- profiling rows by upload/profile identifiers.

## 6. Non-Functional Requirements

### 6.1 Usability

- The system should be easy to present in an academic viva.
- The UI should explain outputs in both simple and technical language.
- Error states should be visible to the user rather than failing silently.
- Diagram rendering should support the DB and SDA explanations directly in the browser.

### 6.2 Reliability

- The system should recover gracefully from missing datasets by falling back to available alternatives.
- The backend should initialize required schema objects automatically on startup.
- The frontend should remain usable in offline presentation mode when the backend is unavailable.

### 6.3 Maintainability

- The codebase shall separate orchestration, persistence, training, prediction, profiling, schema explanation, and presentation logic into distinct modules.
- Configuration paths should be centralized.
- Verification scripts should support quick regression checks for major project phases.

### 6.4 Performance

- The system should remain responsive for semester-project-scale local datasets and demo workflows.
- Kaggle import should process large CSV files in batches rather than loading the full file into SQLite row by row.
- Training should be scoped to a shortlist of three models to keep local execution practical.

### 6.5 Security

- The current system shall be treated as a local academic application, not a hardened production system.
- Cross-origin access shall be restricted to expected local frontend origins.
- Production-grade authentication, authorization, encryption at rest, and secrets management are outside current scope.

### 6.6 Portability

- The Python backend code should remain largely cross-platform.
- The provided launch scripts are optimized for Windows-based usage.
- The frontend should run in standard modern browsers.

### 6.7 Auditability and Explainability

- Profiling results, training runs, recommendations, and candidate metrics shall be persisted for later explanation.
- The saved model threshold shall be stored separately from the model file so inference behavior remains traceable.
- The schema explanation output shall be generated from the live database structure.

## 7. Representative Use Cases

### UC-1. Profile a Dataset

- **Actor:** Presenter or developer
- **Preconditions:** A CSV file exists locally or is uploaded through the API
- **Main Flow:** Select or provide the CSV -> run profiling -> infer target/features -> store upload, dataset summary, and feature profiles -> review warnings and feature explanations
- **Postconditions:** New rows exist in `raw_dataset_uploads`, `dataset_profiles`, and `feature_profiles`

### UC-2. Train and Select a Fraud Model

- **Actor:** Presenter or developer
- **Preconditions:** A valid training dataset can be loaded from DB, CSV, or synthetic fallback
- **Main Flow:** Analyze dataset characteristics -> shortlist 3 models -> train candidates -> tune threshold -> compare metrics -> save final model and metadata -> persist audit rows
- **Postconditions:** `model.pkl`, `model_metadata.json`, and training audit tables are updated

### UC-3. Predict a Manual Transaction

- **Actor:** Presenter
- **Preconditions:** A trained model is available
- **Main Flow:** Enter amount/time/location/merchant -> validate input -> predict fraud probability -> classify using saved threshold -> display confidence and risk signals
- **Postconditions:** Prediction result is shown in the UI or API response

### UC-4. Run the End-to-End Fraud Workflow

- **Actor:** Developer or backend client
- **Preconditions:** Database is available; model is available or can be trained
- **Main Flow:** Insert sample user and transaction -> fetch stored row -> predict -> store prediction -> create alert if fraud
- **Postconditions:** Transaction, prediction, and optional alert rows are persisted

### UC-5. Present the System Architecture

- **Actor:** Presenter
- **Preconditions:** Frontend is open; backend is optional for SDA/offline mode
- **Main Flow:** Open AI/DB/SDA tabs -> review metrics, schema, normalization, and diagrams -> export presentation pack if needed
- **Postconditions:** The system state and architecture can be explained to instructors using live or snapshot-backed content

## 8. Limitations and Out-of-Scope Items

- No authentication, user sessions, or role-based access control
- No remote deployment or multi-user collaboration
- No real payment gateway or banking-system integration
- No streaming transaction ingestion
- No human review workflow for fraud alerts beyond status fields
- No production-grade security or privacy controls
- No direct frontend UI for dataset upload/profile execution, despite backend and CLI support for profiling
- No direct frontend action for `/workflow/run`, even though the backend supports it

## 9. Acceptance Criteria Summary

The system shall be considered aligned with this SRS when:

- the schema initializes successfully with all required tables,
- dataset profiling persists upload, profile, and feature rows,
- the schema explanation layer returns live table, FK, normalization, and ER data,
- the model recommendation layer returns exactly three shortlisted models from the ten-model pool,
- the training pipeline saves a model, metadata, shortlist, and candidate metrics,
- manual prediction returns class, probability, threshold, and explanation text,
- the presentation layer returns diagrams, viva notes, readiness checks, and export content,
- the frontend remains usable in offline presentation mode when the backend is unavailable.

## 10. Conclusion

This repository implements a modular academic fraud-detection system whose requirements span database design, machine-learning workflow, and software-architecture presentation. The SRS above reflects the current implemented system, including both its strengths and its present scope limits, and is intended to support report writing, viva preparation, and future extension work.
