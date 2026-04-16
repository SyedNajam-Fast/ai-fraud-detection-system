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

### Download Kaggle Dataset (Optional)

This project includes a helper script to download the `mlg-ulb/creditcardfraud` dataset using `kagglehub`.

Prerequisites:

- Kaggle credentials must be configured (`kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`).

Run:

```bash
python src/download_dataset.py
```

After download, `creditcard.csv` is copied to:

- `data/raw/creditcardfraud/creditcard.csv`

### Import Kaggle Data Into Database

Load the downloaded Kaggle records into SQLite (table: `kaggle_transactions`):

```bash
python src/import_kaggle_to_db.py
```

Useful options:

```bash
# Keep existing rows and append
python src/import_kaggle_to_db.py --append

# Tune batch size
python src/import_kaggle_to_db.py --batch-size 10000
```

## Run

The main entrypoint now bootstraps itself. On a fresh clone, run:

```bash
python src/main.py
```

If required packages are missing, `main.py` installs them from `requirements.txt`, then it initializes the database, trains the model if needed, and runs the prediction workflow.

The training pipeline now compares multiple models, tunes the decision threshold, and records model-run metadata in SQLite before prediction starts.

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

- `model/train_model.py` prefers database data from `kaggle_transactions` when available.
- If database training data is unavailable, it falls back to `data/fraud_transactions.csv`.
- If CSV data is also unavailable, it generates a synthetic dataset so the project still runs end to end.
- The Kaggle downloader fetches raw dataset files only and does not transform schema for training.
- The selected model, threshold, and validation/test checks are saved to `model/model_metadata.json` and the SQLite model registry tables.
