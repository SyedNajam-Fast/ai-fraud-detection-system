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

## New User Quick Start

If a new teammate wants to run the full system without confusion, follow this exact flow.

### Prerequisites

Install these first:

- Python 3
- Node.js and npm
- Git

### First-Time Setup Only

Open a terminal in the project root and run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### Daily Startup For Teammates

Use two terminals.

#### Terminal 1: Backend

From the project root:

```powershell
.\start_backend.bat
```

Keep this terminal open.

#### Terminal 2: Frontend

From the project root:

```powershell
cd frontend
npm run dev
```

Open the frontend URL shown by Vite, usually:

- `http://127.0.0.1:5173/`

### Important Rule

Always start the backend first, then the frontend.

### If Someone Only Wants The Presentation UI

The frontend can still open without the backend:

- `DB` tab works with built-in demo data
- `SDA` tab works with built-in demo data
- `AI` tab opens, but live training and live API actions need the backend

### Best Command To Use For Backend

Use:

```powershell
.\start_backend.bat
```

Do not rely on:

```powershell
python src\run_api.py
```

because another Python installation on the machine may be picked instead of the project environment.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Frontend Setup

Install frontend packages once:

```powershell
cd frontend
npm install
cd ..
```

### Recommended Windows Setup

If you are setting up the project on a fresh machine, use the project virtual environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
cd frontend
npm install
cd ..
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

### Recommended Startup Order For The UI

Use this exact order on Windows if you want the frontend and backend to start cleanly.

#### 1. Start the backend first

Open a terminal in the project root and run:

```powershell
.\start_backend.bat
```

Important:

- Run this from the project root only.
- Do not close this terminal while using the frontend.
- This launcher always uses `.venv\Scripts\python.exe`, so it avoids the wrong-Python crash problem.

Expected backend address:

- `http://127.0.0.1:8000/api`
- Health check: `http://127.0.0.1:8000/api/health`

#### 2. Start the frontend in a second terminal

```powershell
cd frontend
npm run dev
```

Open the URL shown by Vite. Normally it is:

- `http://127.0.0.1:5173/`

#### 3. Use the app

- `AI` tab: dataset, model choice, training, manual prediction
- `DB` tab: source data, normalization, ER diagram
- `SDA` tab: architecture, workflow, diagrams, requirements, testing

### If The Backend Is Not Running

The frontend now supports an offline presentation mode.

- The page will still open.
- `DB` and `SDA` sections will still work with built-in demo content.
- `AI` section will still show demo content, but live training requires the backend.

### Commands To Prefer

Use these commands:

```powershell
.\start_backend.bat
cd frontend
npm run dev
```

Avoid this if you are having startup issues:

```powershell
python src\run_api.py
```

Reason:

- `python` may point to the wrong interpreter on your machine.
- `start_backend.bat` always uses the project virtual environment.

### Troubleshooting

#### Backend window closes immediately

Run:

```powershell
.\start_backend.bat
```

and read the printed error. The launcher now reports common issues such as:

- missing dependencies
- wrong interpreter
- port `8000` already in use

#### Frontend shows old content

- Stop Vite with `Ctrl + C`
- Run `npm run dev` again inside `frontend`
- Hard refresh the browser with `Ctrl + F5`

#### Port 8000 is already in use

Close the other process using port `8000`, then run:

```powershell
.\start_backend.bat
```

#### Port 5173 is already in use

Vite may automatically choose another port such as `5174`.
Open the exact URL shown in the frontend terminal.

#### npm is not recognized

Install Node.js first, then run:

```powershell
cd frontend
npm install
npm run dev
```

### End-To-End Backend Workflow

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
