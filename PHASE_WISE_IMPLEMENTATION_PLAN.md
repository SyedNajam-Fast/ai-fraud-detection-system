# Phase-Wise Implementation Plan

## 1. Purpose

This file is the execution plan for building the final semester project step by step.

The goal is:

1. build the system in phases,
2. verify each phase locally,
3. fix anything broken before moving forward,
4. keep the project demo-ready for local presentation only.

This plan assumes:

- no deployment is needed,
- the project will be shown locally on your laptop,
- stability and presentation quality matter more than production-scale infrastructure.

## 2. Build Strategy

The correct build order is:

1. strengthen backend foundation,
2. add data understanding,
3. add database explanation layer,
4. add intelligent model recommendation and training flow,
5. add UI,
6. polish demo and documentation.

Rule for every phase:

> Do not move to the next phase until the current phase runs correctly, is manually verified, and any broken behavior is fixed.

## 3. Phase Summary

### Phase 1

Backend foundation cleanup and architecture stabilization

### Phase 2

Dataset ingestion and profiling engine

### Phase 3

Database explanation and normalization module

### Phase 4

Model recommendation engine and expanded training pipeline

### Phase 5

Local UI dashboard and workflow integration

### Phase 6

Diagrams, reporting, demo polish, and viva support

## 4. Phase 1: Backend Foundation Cleanup and Stabilization

## Objective

Turn the current repository from a strong prototype into a stable base for future phases.

## Why this phase comes first

Right now the project already works, but later phases will become messy if the current backend is not cleaned up first. Before adding profiling, recommendation logic, and UI, the internal structure must be reliable.

## Main outcomes

- consistent project structure
- cleaner module boundaries
- better config/path handling
- stable local run commands
- stronger database helpers
- stable model artifact handling
- basic regression checks

## Detailed tasks

### 1. Audit and refactor the current backend flow

Tasks:

- verify each current script purpose
- remove duplication where needed
- make runtime flow easier to extend
- standardize naming across modules

### 2. Create a clear app-level structure

Recommended structure after Phase 1:

```text
database/
model/
src/
  core/
  db/
  ml/
  services/
  utils/
  main.py
```

This does not mean rewriting everything. It means gradually organizing code so later features have a proper place.

### 3. Centralize configuration

Tasks:

- create one config module for important paths
- store database path, model path, metadata path, raw data path in one place
- stop repeating path logic in many files

### 4. Strengthen database access layer

Tasks:

- keep schema initialization stable
- group DB functions by responsibility
- add safe helper functions for counts and summaries
- prepare DB helper functions for later profiling/report modules

### 5. Strengthen model artifact management

Tasks:

- make model loading safer
- make metadata loading safer
- add clearer error messages for missing model or bad metadata
- ensure training and inference use the same expected schema

### 6. Improve console logging

Tasks:

- standardize log style
- print clearer run summaries
- separate warnings from normal informational output

### 7. Add local verification scripts or checks

Tasks:

- verify database initializes cleanly
- verify training runs cleanly
- verify prediction runs cleanly
- verify import script fails clearly when dataset is missing

## Deliverables

- cleaned backend structure
- stable local entrypoint
- improved DB and ML helpers
- updated documentation for local usage

## Acceptance criteria

Phase 1 is complete only if:

- `python src/main.py` runs successfully
- `python src/main.py --force-train` runs successfully
- database schema initializes without errors
- model and metadata are generated correctly
- prediction is stored correctly
- fraud alert logic still works
- no existing core workflow is broken by refactor

## Local test checklist

- run main workflow with existing model
- run forced retraining
- inspect database row creation
- inspect metadata file creation
- inspect that prediction threshold is loaded correctly

## Exit gate

Do not start Phase 2 until the backend is stable and verified.

## 5. Phase 2: Dataset Ingestion and Profiling Engine

## Objective

Add the ability to load a dataset and automatically understand it before modeling.

## Why this phase matters

This is one of the biggest differences between your current project and your final major-project idea.

## Main outcomes

- upload/import-ready data ingestion flow
- automatic dataset summary
- column typing
- missing value analysis
- target column detection support
- simple-language explanation output

## Detailed tasks

### 1. Build dataset profile module

The module should detect:

- row count
- column count
- column names
- data type guesses
- missing values
- duplicate rows
- unique counts
- target-like columns
- imbalance warnings

### 2. Build column explanation engine

For each column, generate:

- technical explanation
- simple explanation

Example:

- technical: "`Amount` is a numeric feature representing transaction value."
- simple: "`Amount` shows how much money was used in a transaction."

### 3. Build profiling persistence

Store profile results in database tables such as:

- `dataset_profiles`
- `feature_profiles`

### 4. Support raw dataset registration

Add a table such as:

- `raw_dataset_uploads`

Store:

- filename
- source path
- imported time
- row count
- column count

### 5. Create profiling CLI flow

Suggested command behavior:

- choose dataset file
- run profiling
- print simple summary
- store profile results

## Deliverables

- dataset profiling module
- column explanation module
- profile persistence tables
- local profiling flow

## Acceptance criteria

Phase 2 is complete only if:

- a CSV can be profiled locally
- profile results are printed and stored
- missing values and class imbalance are detected
- each column gets a readable explanation
- the system can identify candidate target columns or accept a configured target

## Local test checklist

- run profiling on Kaggle fraud CSV
- run profiling on a small custom CSV
- verify database rows are created for profile tables
- verify output is understandable for non-technical audience

## Exit gate

Do not start Phase 3 until profiling is correct and repeatable.

## 6. Phase 3: Database Explanation and Normalization Module

## Objective

Make the system explain database design decisions clearly for the DBS instructor.

## Main outcomes

- schema explanation engine
- PK/FK explanation output
- normalization explanation
- ER-structure summary

## Detailed tasks

### 1. Expand schema for raw, normalized, and AI layers

Add or refine table groups:

- raw layer
- operational normalized layer
- analytics/AI audit layer

### 2. Build schema explanation service

The service should explain:

- why each table exists
- what each primary key does
- what each foreign key does
- how tables relate

### 3. Build normalization explanation service

The system should explain:

- repeated data
- how tables reduce duplication
- what is in 1NF
- what is in 2NF
- what is in 3NF

### 4. Build ER summary output

Generate:

- textual ER explanation
- Mermaid ER diagram text

## Deliverables

- expanded schema
- schema explanation module
- normalization explanation module
- ER diagram text generation

## Acceptance criteria

Phase 3 is complete only if:

- the database design can be explained table by table
- PK/FK relationships are clearly described
- normalization explanation is readable and correct
- ER output matches the actual schema

## Local test checklist

- compare explanation output against schema.sql
- verify generated diagram text matches table relationships
- manually review explanation for academic correctness

## Exit gate

Do not start Phase 4 until database explanation is accurate.

## 7. Phase 4: Model Recommendation Engine and Expanded Training Pipeline

## Objective

Move from fixed 3-model comparison to intelligent top-3 selection from a larger model pool.

## Main outcomes

- 10-model candidate pool
- dataset-aware shortlist logic
- improved training orchestration
- better result comparison
- stronger audit history

## Detailed tasks

### 1. Expand model pool

Initial pool:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Gradient Boosting
6. HistGradientBoosting
7. AdaBoost
8. SVM
9. KNN
10. Naive Bayes

### 2. Build model recommendation rules

Inputs:

- dataset size
- class imbalance
- numeric/categorical ratio
- missingness
- encoded dimensionality estimate

Outputs:

- selected 3 models
- plain-language reason for each choice

### 3. Improve preprocessing and training pipeline

Tasks:

- support dataset-aware preprocessing
- keep train/validation/test split
- train shortlisted models one by one
- track metrics consistently

### 4. Improve evaluation and reporting

Metrics to support:

- precision
- recall
- F1
- ROC AUC
- PR AUC
- confusion matrix
- selected threshold
- overfit gap

### 5. Improve persistence

Store:

- full model pool considered
- shortlisted models
- selection reasons
- final winner

## Deliverables

- model recommendation engine
- expanded training module
- new audit fields/tables if required
- readable training summaries

## Acceptance criteria

Phase 4 is complete only if:

- the system can shortlist 3 models from a larger pool
- training runs end to end on the shortlisted models
- the best model is selected using explicit rules
- results are stored and reproducible
- explanations are understandable

## Local test checklist

- run against Kaggle dataset
- run against small synthetic/custom dataset
- verify shortlist changes appropriately when dataset characteristics change
- verify metadata and DB audit rows are consistent

## Exit gate

Do not start Phase 5 until the AI pipeline is stable and explainable.

## 8. Phase 5: Local UI Dashboard and Workflow Integration

## Objective

Build the interface you will actually show in the demo.

## Main outcomes

- local frontend
- local backend API
- interactive workflow
- readable logs and results

## Recommended stack

- Backend: FastAPI
- Frontend: React + Vite
- Styling: Tailwind CSS

This is local-only, not deployment-focused.

## Detailed tasks

### 1. Build backend API endpoints

Examples:

- upload/profile dataset
- fetch profile results
- fetch schema explanations
- start model recommendation
- start training
- fetch training logs
- fetch final results

### 2. Build frontend screens

Required screens:

- dashboard
- dataset upload
- data understanding
- database explanation
- model recommendation
- training monitor
- results view

### 3. Add simple mode and technical mode

This should affect:

- logs
- explanations
- metric descriptions

### 4. Integrate backend and frontend

The user should be able to go through the full flow without touching the terminal.

## Deliverables

- working local web app
- integrated backend API
- full demonstration workflow

## Acceptance criteria

Phase 5 is complete only if:

- the app starts locally
- the frontend can trigger dataset profiling
- the frontend can show DB explanation
- the frontend can trigger training
- the frontend can display results and logs
- the workflow is stable enough for demo use

## Local test checklist

- run frontend and backend together
- upload and profile a dataset through UI
- start training through UI
- inspect results page
- restart app and confirm it still works

## Exit gate

Do not start Phase 6 until the UI workflow is stable.

## 9. Phase 6: Diagrams, Reporting, Demo Polish, and Viva Support

## Objective

Turn the working system into a polished semester-project presentation package.

## Main outcomes

- diagrams
- report-ready summaries
- final demo stability
- viva notes

## Detailed tasks

### 1. Add diagram generation or curated diagram pages

Include:

- use case diagram
- activity diagram
- sequence diagram
- component diagram
- deployment diagram
- ER diagram
- DFD

### 2. Add report/export content

Prepare:

- project overview
- dataset summary
- database design summary
- model comparison summary
- final result summary

### 3. Add demo-safe polish

Tasks:

- remove confusing text
- improve labels
- ensure all buttons and flows work
- keep screenshots and prepared sample data ready

### 4. Prepare viva support notes

Prepare answers for:

- why this schema?
- why these models?
- why only 3 selected?
- what is normalization here?
- what is threshold tuning?
- how is the system modular?

## Deliverables

- diagram pages or diagram exports
- presentation-ready summaries
- final local demo flow
- viva support notes

## Acceptance criteria

Phase 6 is complete only if:

- all major diagrams are available
- the system looks presentable for local demo
- the explanation text is coherent
- you can perform a full demo without broken flow

## Local test checklist

- full end-to-end dry run before presentation
- confirm all diagrams load
- confirm profile, schema, training, and results views work
- prepare offline backup files and screenshots

## 10. Execution Rules Between Phases

For every phase, the working rule will be:

1. implement the phase,
2. run local checks,
3. identify anything broken,
4. fix the breakage,
5. re-run checks,
6. only then move to the next phase.

## 11. Recommended Order of Implementation in This Workspace

This is the exact order I recommend for this repository:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6

No skipping.

## 12. Best Working Style for This Project

For this semester project, the best working style is:

- keep every phase small enough to verify,
- avoid rewriting the whole codebase in one jump,
- keep local demo readiness in mind,
- verify after every meaningful change,
- keep documentation updated with implementation reality.

## 13. What Happens Next

Next action after this file:

Implement Phase 1 first.

Phase 1 will focus on:

- cleanup,
- structure,
- reliability,
- local verification.

Only after Phase 1 is fully stable should we begin dataset profiling in Phase 2.
