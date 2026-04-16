# Project Report

## Overview

This repository implements an academic fraud detection system with two connected layers:

1. A multi-model machine learning pipeline that compares Logistic Regression, Random Forest, and Extra Trees classifiers.
2. A relational database layer that stores users, transactions, predictions, fraud alerts, and model training history.

## Architecture

The workflow now includes model selection and validation checks:

1. Insert a transaction into the database.
2. Fetch the transaction data.
3. Train and compare multiple candidate models when needed.
4. Select the best model using validation metrics and threshold tuning.
5. Pass the transaction features into the trained model.
6. Produce a fraud prediction and probability.
7. Store the prediction in the database.
8. Create a fraud alert when the prediction is fraudulent.

## Implementation Details

- Database: SQLite, initialized from `database/schema.sql`
- Models: `LogisticRegression`, `RandomForestClassifier`, `ExtraTreesClassifier`
- Features: `amount`, `time`, `location`, `merchant`
- Saved artifact: `model/model.pkl`
- Model metadata: `model/model_metadata.json`
- Training registry: `model_training_runs` and `model_candidate_metrics`

## Evaluation

The training script reports:

- Selected model name and threshold
- Validation F1 and test F1
- Overfit and underfit checks
- Accuracy and confusion matrix

## Run Summary

The main entrypoint initializes the database, trains the model if needed, inserts a demo transaction, generates a prediction, stores the result, and prints the outcome to the console.
