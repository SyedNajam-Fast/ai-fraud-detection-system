# Project Report

## Overview

This repository implements an academic fraud detection system with two connected layers:

1. A Random Forest classifier that predicts whether a transaction is fraudulent.
2. A relational database layer that stores users, transactions, predictions, and fraud alerts.

## Architecture

The workflow is intentionally simple:

1. Insert a transaction into the database.
2. Fetch the transaction data.
3. Pass the transaction features into the trained model.
4. Produce a fraud prediction and probability.
5. Store the prediction in the database.
6. Create a fraud alert when the prediction is fraudulent.

## Implementation Details

- Database: SQLite, initialized from `database/schema.sql`
- Model: `RandomForestClassifier`
- Features: `amount`, `time`, `location`, `merchant`
- Saved artifact: `model/model.pkl`

## Evaluation

The training script reports:

- Accuracy
- Confusion matrix

## Run Summary

The main entrypoint initializes the database, trains the model if needed, inserts a demo transaction, generates a prediction, stores the result, and prints the outcome to the console.
