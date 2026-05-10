# Updated Project Proposal

## Project Title

AI-Powered Fraud Detection and Transaction Management System

## 1. Background

Fraud-detection projects are often shown either as machine-learning notebooks or as separate database assignments. That split makes it difficult to explain how real systems actually work end to end. This project proposes a unified local platform that combines data understanding, database design, model recommendation, fraud-model training, prediction, alert logging, and presentation support in one repository.

## 2. Problem Statement

Traditional semester projects usually focus on only one area:

- only the database,
- only the machine-learning model,
- or only static diagrams.

The problem is that instructors from DBS, AI, and SDA need to see how the whole system fits together. This project solves that gap by building one system that can:

- profile fraud-related datasets,
- explain the database schema and normalization,
- recommend and train fraud-detection models,
- predict suspicious transactions,
- store outcomes and audit history,
- present the architecture and workflow clearly through a frontend dashboard.

## 3. Proposed Solution

The proposed system is a local academic fraud-detection platform with five connected layers:

1. Frontend presentation layer using React and Vite
2. FastAPI backend for live system actions
3. Service layer for profiling, workflow, recommendation, schema explanation, and presentation support
4. SQLite database for operational and analytical storage
5. Machine-learning layer using scikit-learn pipelines

## 4. Main Objectives

The project aims to:

- build a relational database for transaction, prediction, alert, profiling, and audit data,
- support end-to-end transaction processing and fraud prediction,
- understand datasets before training by profiling structure and target behavior,
- shortlist the most suitable candidate models before full training,
- train the shortlisted models and select the strongest winner,
- preserve model decisions and dataset knowledge for explanation later,
- provide a presentation-friendly interface for AI, DBS, and SDA discussion.

## 5. Core Modules

### 5.1 Dataset profiling module

This module will:

- load CSV datasets,
- detect likely target columns,
- infer feature roles and simplified data types,
- calculate duplicates, missingness, and class imbalance,
- store profiling summaries and column explanations in SQLite.

### 5.2 Database explanation module

This module will:

- inspect the live SQLite schema,
- explain table purpose, keys, and indexes,
- discuss normalization honestly,
- generate Mermaid ER output for presentation.

### 5.3 Model recommendation module

This module will:

- inspect dataset characteristics,
- score a pool of candidate models,
- shortlist the top 3 models for actual training,
- explain why those models were selected.

### 5.4 Training and evaluation module

This module will:

- load the best available training dataset,
- preprocess numeric and categorical features,
- train shortlisted models,
- tune the decision threshold,
- evaluate validation and test performance,
- save the final winner and metadata,
- log training history into SQLite.

### 5.5 Prediction and alert module

This module will:

- accept transaction input,
- predict fraud probability,
- apply the saved threshold,
- store prediction records,
- create fraud alerts when necessary.

### 5.6 Presentation module

This module will:

- present AI, DB, and SDA views in one frontend,
- show diagrams and viva notes,
- support offline snapshot mode if the backend is unavailable.

## 6. Proposed Database Design

The proposed schema is divided into four layers:

- raw training data
- raw profiling registration
- operational fraud workflow
- analytics and audit history

Main tables include:

- users
- transactions
- predictions
- fraud_alerts
- kaggle_transactions
- raw_dataset_uploads
- dataset_profiles
- feature_profiles
- model_training_runs
- model_recommendations
- model_candidate_metrics

## 7. Proposed AI Strategy

The model pipeline will use a project-level schema with:

- amount
- time
- location
- merchant

Target:

- fraud

The training process will:

- prefer real data from SQLite or CSV when available,
- use synthetic fallback data if necessary,
- score a 10-model pool,
- shortlist 3 models,
- select the winner by validation performance,
- store both the model artifact and the supporting metadata.

## 8. Technology Stack

- Python
- pandas
- numpy
- scikit-learn
- joblib
- FastAPI
- Uvicorn
- SQLite
- React
- Vite
- Mermaid

## 9. Expected Outcomes

At the end of the project, the system should be able to:

- explain the source dataset,
- explain the relational schema,
- show how and why a shortlist of models was chosen,
- train and evaluate the winning fraud model,
- predict suspicious transactions,
- store alerts and model history,
- support a complete local presentation for DBS, AI, and SDA.

## 10. Scope and Limits

This proposal is for a local academic system, not a production banking platform. The project does not currently aim to provide:

- multi-user authentication,
- cloud deployment,
- real payment integration,
- production security hardening,
- real-time large-scale streaming.

The focus is explainability, modularity, and academic strength.

## 11. Conclusion

This proposal reflects a complete semester-project direction where database systems, machine learning, and software design are presented as one connected solution. The proposed platform is practical, locally runnable, and strong enough to support both implementation and viva discussion across multiple subjects.
