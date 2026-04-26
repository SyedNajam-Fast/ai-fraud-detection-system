# Semester Project Master Plan

## 1. Purpose of This File

This document turns your scattered idea into a structured major-project plan.

Your goal is not just to show a fraud model. Your goal is to present one integrated system that satisfies:

- Database course expectations
- AI course expectations
- SDA course expectations
- presentation quality expectations through a good UI

## 2. Project Vision

## Final project title

**Intelligent Fraud Detection, Data Understanding, and Database Design Explanation System**

Alternative academic title:

**AI-Powered Fraud Detection and Explainable Data Management Platform**

## 3. Core Idea in Simple Words

When you show your project to instructors, the system should behave like this:

1. You load a dataset in front of them.
2. The system studies the dataset automatically.
3. It explains every important column in simple language.
4. It tells what kind of data it found, what is missing, what looks suspicious, and what the target column means.
5. It explains how the data should be stored in database tables.
6. It explains normalization, primary keys, foreign keys, and relationships.
7. It intelligently chooses the best 3 candidate models from a larger model pool.
8. It trains the chosen models one by one.
9. It prints simple logs so even a non-technical person can follow what is happening.
10. It shows results, comparisons, and diagrams in a good interface.

That is the correct direction for a major semester project.

## 4. The Real Problem You Are Solving

Right now, most student projects only do one of these:

- only database design
- only model training
- only dashboard

Your project should combine all three into one coherent product:

- understand data,
- design and explain the database,
- select and train AI models,
- present the whole system clearly.

That combination is what will make the project stronger.

## 5. Course-Wise Mapping

## 5.1 Database course contribution

The database instructor should see:

- proper schema design
- staging tables and final normalized tables
- primary keys and foreign keys
- normalization explanation
- relationship mapping
- transaction storage and prediction storage
- audit tables for model runs
- optional data dictionary

## 5.2 AI course contribution

The AI instructor should see:

- automated data profiling
- feature typing
- missing value analysis
- imbalance detection
- model recommendation
- training of selected models
- evaluation metrics
- threshold tuning
- explainability and comparison

## 5.3 SDA course contribution

The SDA instructor should see:

- modular architecture
- system components and data flow
- UML/behavior diagrams
- activity and sequence diagrams
- layer separation
- input-process-output flow
- maintainable software structure
- user-oriented interface

## 6. Recommended Final Product

The final system should have six major modules.

### Module 1: Data Intake and Profiling

Inputs:

- CSV file
- optional database import file

System actions:

- detect column names
- detect numeric, categorical, datetime, ID, and target-like columns
- show row count, missing values, duplicate count, class distribution
- explain each column in simple words

Example output in simple language:

> The `Amount` column shows how much money was involved in each transaction. Most values are normal-sized, but a few transactions are much larger than the rest.

### Module 2: Database Design and Explanation

System actions:

- separate raw input data from normalized transactional design
- build or suggest tables
- explain why tables were split
- explain primary keys and foreign keys
- explain 1NF, 2NF, and 3NF in project language
- show ER diagram

Example explanation:

> User information is stored in one table and transaction records are stored in another table so the same customer details are not repeated again and again.

### Module 3: Model Recommendation Engine

System actions:

- inspect dataset size
- inspect class imbalance
- inspect feature types
- inspect missingness
- inspect number of categories
- inspect linear vs non-linear difficulty signals
- shortlist 3 models from a larger pool

The key academic value:

You are not manually selecting models every time. The system justifies why those 3 were chosen.

### Module 4: Training and Evaluation Engine

System actions:

- preprocess data
- split train, validation, and test sets
- train 3 selected models one by one
- print simple progress logs
- compare results side by side
- choose the best final model
- save model and metadata

### Module 5: Explanation and Reporting Engine

System actions:

- explain metrics in simple words
- explain false positives and false negatives
- explain threshold
- explain why one model was preferred
- generate summary cards and report-ready text

### Module 6: UI and Presentation Layer

System actions:

- upload data
- view profiling summary
- view schema and normalization
- run model recommendation
- start training
- inspect logs and results
- view diagrams
- export screenshots or report text

## 7. Recommended Technical Architecture

## Recommended stack

For the actual semester project, the strongest practical stack is:

- Backend: `FastAPI`
- Frontend: `React + Vite`
- Styling/UI: `Tailwind CSS` with a clean dashboard layout
- Database: `SQLite` first, with schema structured cleanly enough to move to PostgreSQL later
- ML: `scikit-learn`
- Diagrams: `Mermaid` for generated diagrams, plus exported PNGs for report slides

## Why this stack is better for marks

- FastAPI gives a real system feel for SDA
- React gives stronger UI marks than a console-only demo
- SQLite keeps the project lightweight and easy to run
- scikit-learn is enough for strong academic AI work

## If time becomes short

Fallback stack:

- Frontend + backend together in `Streamlit`

This is faster to build, but React + FastAPI will usually look more serious in presentation.

## 8. Recommended Database Design for the Final Version

The current schema is a good seed, but your final major project should expand it.

## 8.1 Proposed table groups

### Raw ingestion layer

- `raw_dataset_uploads`
- `raw_dataset_columns`
- `raw_dataset_rows` or staging import table

Purpose:

- keep original data details
- preserve provenance of uploaded files

### Business/normalized operational layer

- `users`
- `cards`
- `locations`
- `merchants`
- `transactions`
- `predictions`
- `fraud_alerts`

Purpose:

- explain normalized design cleanly to DBS instructor

### AI/analytics layer

- `dataset_profiles`
- `feature_profiles`
- `model_training_runs`
- `model_candidate_metrics`
- `model_recommendations`
- `evaluation_reports`

Purpose:

- preserve AI decisions and training history

## 8.2 Normalization story you should present

Your database explanation should be:

1. Raw uploaded data is stored first so the source is preserved.
2. Important entities are separated into their own tables.
3. Repeating information like merchant names and location details is moved to lookup tables.
4. Prediction and alert records are stored separately because they describe outcomes, not the original transaction itself.

That gives you a clear normalization narrative.

## 9. Recommended AI Design for the Final Version

## 9.1 Model pool of 10 candidates

A practical scikit-learn model pool:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Gradient Boosting
6. HistGradientBoosting
7. AdaBoost
8. Support Vector Machine
9. K-Nearest Neighbors
10. Naive Bayes

You can later replace one or two with XGBoost or LightGBM if allowed and if time permits.

## 9.2 How the system should choose 3 models

The model recommendation engine should not pick randomly. It should use rules like:

- if dataset is small and simple: include Logistic Regression
- if dataset is non-linear: include Random Forest or Extra Trees
- if there are many noisy interactions: include tree ensembles
- if classes are highly imbalanced: prefer models that work well with probability ranking and class weights
- if there are many categorical features after encoding: avoid models that become too expensive

## Example recommendation output

> The system selected Logistic Regression, Random Forest, and HistGradientBoosting because the dataset is moderately sized, class-imbalanced, and contains patterns that may not be fully linear.

## 9.3 Training outputs that should appear

For each selected model, show:

- preprocessing steps
- train/validation/test split
- training start and end
- precision
- recall
- F1 score
- ROC AUC
- PR AUC
- confusion matrix
- best threshold
- simple explanation

## 9.4 Best-model decision

The system should select the final winner based on a defined rule, for example:

- primary metric: PR AUC or F1 for fraud class
- secondary metric: recall
- tertiary check: overfit gap

That makes the model choice defensible.

## 10. Logging Style You Need

You specifically asked for logs that even a non-technical person can understand.

That means the system should print logs in two styles:

## 10.1 Simple logs

Examples:

- "The system is reading the dataset now."
- "This column looks like the transaction amount."
- "Some values are missing in this column."
- "The data has many more normal transactions than fraud transactions."
- "The system selected 3 models that fit this type of data."
- "Model 1 training has started."
- "This model is catching many fraud cases but also making some false alarms."

## 10.2 Technical logs

Examples:

- "Detected target column: Class"
- "Missing values found in 2 columns"
- "Stratified split applied: 60/20/20"
- "Selected threshold from validation PR curve: 0.41"
- "Overfit flag raised due to train-validation F1 gap"

Best practice:

Put a toggle in the UI:

- `Simple Mode`
- `Technical Mode`

That will impress instructors because the same system can explain itself at different levels.

## 11. Diagrams You Should Include

For the SDA instructor, include these diagrams in the application and in the report.

## Essential diagrams

1. Use case diagram
2. Activity diagram
3. Sequence diagram
4. Component diagram
5. Deployment diagram
6. ER diagram
7. Data flow diagram

## Suggested meanings

- Use case: user uploads data, profiles data, runs models, views results
- Activity: end-to-end flow from upload to alert/report
- Sequence: frontend, backend, database, ML engine interactions
- Component: UI, API, DB, model engine, reporting engine
- Deployment: browser, app server, DB file/model artifacts
- ERD: database tables and relationships
- DFD: data enters, transforms, stores, predicts, and reports

## 12. UI Strategy

## Screens you should build

1. Dashboard/Home
2. Dataset Upload
3. Data Understanding
4. Database Design View
5. Model Recommendation
6. Training Monitor
7. Results Comparison
8. Diagrams and Report View

## What the UI should show

### Dashboard/Home

- project title
- quick summary counts
- last model status
- recent dataset status

### Dataset Upload

- upload CSV
- detect target column
- preview rows
- show file summary

### Data Understanding

- column-by-column explanation
- missing values
- distributions
- imbalance warning
- duplicate check

### Database Design View

- final schema tables
- PK/FK explanation
- normalization explanation
- ER diagram

### Model Recommendation

- candidate pool
- why 3 models were selected
- expected strengths and weaknesses

### Training Monitor

- live logs
- model-by-model progress
- simple and technical view toggle

### Results Comparison

- table of metrics
- best model card
- confusion matrix charts
- threshold explanation

### Diagrams and Report View

- SDA diagrams
- download/export report content
- viva explanation notes

## 13. Development Phases

## Phase 1: Strengthen the backend foundation

Build first:

- proper data profiling module
- expanded schema
- model recommendation engine
- better training history tables

## Phase 2: Add explainability and report generation

Build next:

- plain-language explanation functions
- normalization explanation generator
- metric explanation generator
- Mermaid diagram generation

## Phase 3: Add the presentation layer

Build next:

- FastAPI endpoints
- React dashboard
- training monitor and results pages

## Phase 4: Polish for demo

Finish with:

- better styling
- demo dataset presets
- stable screenshots
- prepared viva script

## 14. Reuse Plan From Current Repository

You should not throw away the current project. Reuse it smartly.

## Keep and expand

- `database/schema.sql`
- `src/db.py`
- `model/train_model.py`
- `src/predict.py`
- training run registry idea
- prediction and alert storage idea

## Replace or expand heavily

- hardcoded demo transaction approach
- current simple runtime workflow
- current 3-model-only selection logic
- current artificial Kaggle feature mapping narrative
- console-only interaction

## 15. What the Final Demo Should Look Like

In the final demonstration, you should be able to do this sequence:

1. Open the web app.
2. Upload a dataset.
3. Let the system explain the columns in plain language.
4. Open the database tab and show normalized tables, PKs, FKs, and ERD.
5. Open the AI tab and show why the system shortlisted 3 models.
6. Start training and show readable logs.
7. Compare results side by side.
8. Show the selected best model and why it was chosen.
9. Show alerts, predictions, and stored records.
10. Show SDA diagrams and overall architecture.

If you can perform that flow smoothly, the project will look like one serious integrated system instead of three disconnected course assignments.

## 16. Risks and How to Control Them

## Risk 1: Project becomes too big

Control:

- focus on one solid end-to-end dataset flow
- avoid deep learning
- avoid too many advanced extras

## Risk 2: UI takes too much time

Control:

- backend first
- keep dashboard clean and academic
- use a component library instead of custom design from scratch

## Risk 3: Model selection becomes messy

Control:

- use rule-based shortlist logic first
- keep the first version deterministic and explainable

## Risk 4: Database explanation becomes weak

Control:

- explicitly separate raw, normalized, and AI audit tables
- prepare normalization notes and ERD early

## 17. Best Final Scope Recommendation

To keep the project strong but achievable, the best scope is:

- one complete fraud-detection platform
- one upload flow
- one normalized database story
- one 10-model pool
- one automatic 3-model shortlist engine
- one polished dashboard
- one report/diagram section

That is enough for a major semester project.

## 18. Direct Recommendation

Your best path is:

1. Keep the current repository as the backend seed.
2. Expand the database into raw layer + normalized layer + AI audit layer.
3. Upgrade the ML module from fixed 3-model comparison to intelligent 3-from-10 recommendation.
4. Add simple-language explanations for data, database, and model behavior.
5. Build a React + FastAPI dashboard for the final presentation.

## 19. Best Short Project Pitch

If you need one sharp project pitch:

> This project is an intelligent fraud detection platform that can study an uploaded dataset, explain its columns and structure in simple language, design and justify a normalized database schema, automatically shortlist and train the most suitable machine learning models, and present the full system with diagrams and readable results through a modern interface.

## 20. Final Verdict

Your idea is strong, but it needs structure. The correct structured version is:

- current repository = foundation
- final semester project = explainable data understanding + explainable database design + intelligent model recommendation + polished UI

That is the version you should now build.
