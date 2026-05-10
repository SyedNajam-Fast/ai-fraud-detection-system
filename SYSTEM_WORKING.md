# System Working Overview

## 1. What This System Is Now

This project is no longer only a backend fraud-detection script. It is now a local academic platform with four connected parts:

1. **SQLite database layer** for runtime records, profiling history, and model audit history
2. **Machine-learning layer** for recommendation-based model selection, training, and prediction
3. **FastAPI backend** for live system actions
4. **React/Vite frontend** for AI, DBS, and SDA presentation

The project is designed to be explainable in a viva, easy to run locally, and strong enough to show how database design, AI workflow, and software architecture fit together in one system.

## 2. Recommended Startup Flow

### 2.1 Backend

Use the project launcher:

```powershell
.\start_backend.bat
```

This uses the project virtual environment and starts the FastAPI server through `src/run_api.py`.

Default backend address:

- `http://127.0.0.1:8000/api`

Health check:

- `http://127.0.0.1:8000/api/health`

### 2.2 Frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Default frontend address:

- `http://127.0.0.1:5173/`

### 2.3 Important behavior

- Start the backend first if you want live API data.
- The frontend can still open without the backend.
- When the backend is unavailable, the frontend switches to **offline presentation mode** and uses built-in demo snapshots.

## 3. What the User Can Do

### 3.1 AI tab

The AI tab supports:

- viewing the current training dataset preview,
- seeing the dataset signals used for recommendation,
- understanding why the shortlist was chosen,
- training or retraining the model,
- trying manual transaction prediction,
- predicting a held-out test sample live.

### 3.2 DB tab

The DB tab supports:

- showing the latest profiled dataset,
- showing raw column examples,
- explaining table separation by layer,
- explaining normalization,
- rendering the ER diagram from live schema data.

### 3.3 SDA tab

The SDA tab supports:

- introduction and problem statement discussion,
- workflow and methodology explanation,
- architecture and module explanation,
- testing, requirements, and diagram discussion.

This tab is largely backed by presentation-ready content so it remains useful even when the backend is not live.

## 4. Main Runtime Paths

## 4.1 End-to-end CLI workflow

If someone runs:

```powershell
python src\main.py
```

the system performs this sequence:

1. checks Python dependencies,
2. installs missing packages if needed,
3. initializes the SQLite schema,
4. ensures a valid model and metadata file exist,
5. trains the model if needed,
6. inserts a sample user and sample transaction,
7. fetches the stored transaction,
8. predicts fraud probability,
9. applies the saved threshold,
10. stores the prediction,
11. creates a fraud alert if fraud is predicted,
12. prints a workflow summary.

## 4.2 Dataset profiling flow

The profiling flow works through CLI or API:

1. a CSV path is selected or a file is uploaded,
2. the dataset is loaded into pandas,
3. row count, column count, duplicates, missing cells, and target candidates are calculated,
4. each column gets a role, data-type classification, and simple explanation,
5. the profiling results are written into SQLite.

The main persistence tables are:

- `raw_dataset_uploads`
- `dataset_profiles`
- `feature_profiles`

## 4.3 Model recommendation and training flow

The training pipeline does more than simply fit one model:

1. load the best available dataset source,
2. inspect dataset characteristics,
3. score a 10-model pool,
4. shortlist the top 3 models,
5. preprocess the data with a shared pipeline,
6. train the shortlisted models,
7. tune the threshold from validation probabilities,
8. choose the final winner using validation metrics,
9. save the model and metadata,
10. store training audit rows in SQLite.

## 4.4 Manual prediction flow

The manual prediction path works like this:

1. user enters amount, hour, location, and merchant,
2. the backend validates the inputs,
3. `src/predict.py` loads the saved model and metadata,
4. the probability is computed,
5. the saved threshold converts probability into class,
6. the response includes:
   - prediction label,
   - probability,
   - threshold,
   - confidence band,
   - risk signals,
   - simple explanation text.

## 5. Current Backend Structure

The current backend is split into focused modules.

### 5.1 Core entrypoints

- `src/main.py` for CLI end-to-end workflow
- `src/run_api.py` for backend startup
- `src/api/app.py` for REST endpoints

### 5.2 Services

- `src/services/workflow.py` for end-to-end transaction workflow
- `src/services/dataset_profiling.py` for profiling and explanation of datasets
- `src/services/schema_explainer.py` for live schema explanation and ER output
- `src/services/model_recommendation.py` for top-3 shortlist logic
- `src/services/ai_demo.py` for AI-tab demo features
- `src/services/presentation_support.py` for diagrams, viva notes, and export bundles

### 5.3 Data and persistence

- `src/db.py` manages all SQLite interaction
- `database/schema.sql` owns the relational schema

### 5.4 Model logic

- `model/train_model.py` handles training, recommendation, evaluation, and persistence
- `src/predict.py` handles inference with the saved threshold

## 6. Current Database Design

The live schema currently contains 11 application tables.

### 6.1 Raw training layer

- `kaggle_transactions`

This stores imported Kaggle records in wide form for database-first training.

### 6.2 Raw profiling layer

- `raw_dataset_uploads`

This tracks which dataset file was selected or uploaded.

### 6.3 Operational layer

- `users`
- `transactions`
- `predictions`
- `fraud_alerts`

These are the main runtime workflow tables.

### 6.4 Analytics and audit layer

- `dataset_profiles`
- `feature_profiles`
- `model_training_runs`
- `model_recommendations`
- `model_candidate_metrics`

These keep the system explainable and auditable.

## 7. Current AI Design

### 7.1 Project feature schema

The final project-level training and runtime features are:

- `amount`
- `time`
- `location`
- `merchant`

Target:

- `fraud`

### 7.2 Data source fallback order

The system uses:

1. SQLite `kaggle_transactions`
2. `data/fraud_transactions.csv`
3. synthetic fallback data

### 7.3 Candidate model pool

The recommendation layer scores these ten models:

- logistic regression
- decision tree
- random forest
- extra trees
- gradient boosting
- hist gradient boosting
- adaboost
- svm
- knn
- naive bayes

Only the top 3 shortlisted models are trained in the main comparison stage.

### 7.4 Model selection logic

The winner is chosen mainly by:

1. validation average precision
2. validation F1
3. validation recall

### 7.5 Threshold logic

The system does not rely on a fixed threshold of 0.5. It selects the threshold from validation probabilities using precision-recall behavior.

## 8. API Endpoints That Matter

### 8.1 Health and dashboard

- `GET /api/health`
- `GET /api/dashboard`

### 8.2 Dataset and profiling

- `GET /api/datasets/options`
- `GET /api/profiles/latest`
- `POST /api/profile/path`
- `POST /api/profile/upload`

### 8.3 Schema and presentation

- `GET /api/schema`
- `GET /api/presentation`
- `GET /api/presentation/export`

### 8.4 AI features

- `GET /api/recommendations/current`
- `GET /api/ai/dataset-preview`
- `POST /api/predict/manual`
- `GET /api/predict/test-sample`
- `GET /api/model/latest`
- `POST /api/train`
- `POST /api/workflow/run`

## 9. Current Local State

At the time of this update:

- database exists at `data/fraud_detection.db`
- model artifact exists at `model/model.pkl`
- metadata exists at `model/model_metadata.json`
- sample profile dataset exists at `data/samples/sample_profile_dataset.csv`
- Kaggle CSV is not currently present locally

Current local model metadata shows:

- dataset source: `synthetic:generated`
- selected model: `logistic_regression`
- shortlist size: `3`
- full model pool size: `10`
- selected threshold: `0.6762725380991143`

Current row counts:

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

## 10. Offline Mode Behavior

If the backend is not running:

- the frontend checks `/api/health`,
- AI and DB tabs load demo snapshots from `frontend/src/demoData.js`,
- SDA content remains presentation-ready,
- live retraining and live API actions are disabled,
- the UI shows a clear offline presentation message.

This is useful for demos when the backend is temporarily unavailable.

## 11. What Is Strong About the Current System

- It connects AI, DBS, and SDA in one repository.
- It supports both operational workflow and academic explanation.
- It persists model and dataset history instead of only printing console output.
- It has a presentation-oriented frontend instead of relying only on scripts.
- It remains runnable even without real Kaggle data.

## 12. Current Limits

- local only
- no authentication
- no production deployment story
- no real-time transaction stream
- no full frontend coverage for every backend action
- Kaggle mapping into project-level categories is heuristic

## 13. Best One-Line Description

If someone asks what the system does today, the best short answer is:

> It is a local fraud-detection presentation platform that profiles datasets, explains the database, recommends and trains fraud models, predicts suspicious transactions, stores audit history in SQLite, and presents the whole system through AI, DB, and SDA views.
