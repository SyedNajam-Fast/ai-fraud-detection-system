# AI-Powered Credit Card Fraud Detection and Transaction Management System

This project combines a Random Forest fraud detector with a lightweight SQLite-backed transaction workflow.

## What It Does

- Trains a fraud detection model and saves it as `model/model.pkl`
- Stores users, transactions, predictions, and fraud alerts in SQLite
- Runs an end-to-end demo transaction through the database and model pipeline

## Project Structure

- `database/schema.sql` - relational schema for the system
- `model/train_model.py` - trains and saves the model
- `src/db.py` - database helper functions
- `src/insert_data.py` - inserts demo transaction data
- `src/predict.py` - loads the saved model and produces predictions
- `src/main.py` - orchestrates the full workflow

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Train the model directly:

```bash
python model/train_model.py
```

Run the full database-to-model workflow:

```bash
python src/main.py
```

Force model retraining before workflow execution:

```bash
python src/main.py --force-train
```

## Notes

- The training script uses `data/fraud_transactions.csv` if present.
- If that file is absent, it generates a synthetic dataset so the project still runs end to end.
