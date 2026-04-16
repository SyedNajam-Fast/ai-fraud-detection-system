PRAGMA foreign_keys = ON;

-- Core user account information tied to cardholders.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    card_number TEXT NOT NULL UNIQUE
);

-- Raw transactions used as ML model input.
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0),
    time INTEGER NOT NULL CHECK (time BETWEEN 0 AND 23),
    location TEXT NOT NULL,
    merchant TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Fraud predictions generated for each transaction.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    FOREIGN KEY (transaction_id) REFERENCES transactions (id) ON DELETE CASCADE
);

-- Alerts raised when a transaction is predicted as fraud.
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL,
    alert_time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'dismissed')),
    FOREIGN KEY (transaction_id) REFERENCES transactions (id) ON DELETE CASCADE
);

-- Raw Kaggle credit card records used for database-first training.
CREATE TABLE IF NOT EXISTS kaggle_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_seconds INTEGER NOT NULL CHECK (time_seconds >= 0),
    amount REAL NOT NULL,
    v1 REAL NOT NULL,
    v2 REAL NOT NULL,
    v3 REAL NOT NULL,
    v4 REAL NOT NULL,
    v5 REAL NOT NULL,
    v6 REAL NOT NULL,
    v7 REAL NOT NULL,
    v8 REAL NOT NULL,
    v9 REAL NOT NULL,
    v10 REAL NOT NULL,
    v11 REAL NOT NULL,
    v12 REAL NOT NULL,
    v13 REAL NOT NULL,
    v14 REAL NOT NULL,
    v15 REAL NOT NULL,
    v16 REAL NOT NULL,
    v17 REAL NOT NULL,
    v18 REAL NOT NULL,
    v19 REAL NOT NULL,
    v20 REAL NOT NULL,
    v21 REAL NOT NULL,
    v22 REAL NOT NULL,
    v23 REAL NOT NULL,
    v24 REAL NOT NULL,
    v25 REAL NOT NULL,
    v26 REAL NOT NULL,
    v27 REAL NOT NULL,
    v28 REAL NOT NULL,
    class_label INTEGER NOT NULL CHECK (class_label IN (0, 1)),
    source_file TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

-- Model registry for multi-model training and audit history.
CREATE TABLE IF NOT EXISTS model_training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_source TEXT NOT NULL,
    selection_metric TEXT NOT NULL DEFAULT 'validation_average_precision',
    selected_model_name TEXT NOT NULL,
    selected_threshold REAL NOT NULL CHECK (selected_threshold BETWEEN 0 AND 1),
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    train_count INTEGER NOT NULL CHECK (train_count > 0),
    validation_count INTEGER NOT NULL CHECK (validation_count > 0),
    test_count INTEGER NOT NULL CHECK (test_count > 0),
    train_f1 REAL NOT NULL CHECK (train_f1 BETWEEN 0 AND 1),
    validation_f1 REAL NOT NULL CHECK (validation_f1 BETWEEN 0 AND 1),
    test_f1 REAL NOT NULL CHECK (test_f1 BETWEEN 0 AND 1),
    validation_average_precision REAL NOT NULL CHECK (validation_average_precision BETWEEN 0 AND 1),
    test_average_precision REAL NOT NULL CHECK (test_average_precision BETWEEN 0 AND 1),
    overfit_flag INTEGER NOT NULL CHECK (overfit_flag IN (0, 1)),
    underfit_flag INTEGER NOT NULL CHECK (underfit_flag IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('completed', 'failed')),
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_candidate_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    cv_f1_mean REAL,
    cv_f1_std REAL,
    train_precision REAL NOT NULL CHECK (train_precision BETWEEN 0 AND 1),
    train_recall REAL NOT NULL CHECK (train_recall BETWEEN 0 AND 1),
    train_f1 REAL NOT NULL CHECK (train_f1 BETWEEN 0 AND 1),
    validation_precision REAL NOT NULL CHECK (validation_precision BETWEEN 0 AND 1),
    validation_recall REAL NOT NULL CHECK (validation_recall BETWEEN 0 AND 1),
    validation_f1 REAL NOT NULL CHECK (validation_f1 BETWEEN 0 AND 1),
    validation_average_precision REAL NOT NULL CHECK (validation_average_precision BETWEEN 0 AND 1),
    train_average_precision REAL NOT NULL CHECK (train_average_precision BETWEEN 0 AND 1),
    validation_threshold REAL NOT NULL CHECK (validation_threshold BETWEEN 0 AND 1),
    fit_gap REAL NOT NULL,
    overfit_flag INTEGER NOT NULL CHECK (overfit_flag IN (0, 1)),
    underfit_flag INTEGER NOT NULL CHECK (underfit_flag IN (0, 1)),
    confusion_matrix_json TEXT NOT NULL,
    selected INTEGER NOT NULL DEFAULT 0 CHECK (selected IN (0, 1)),
    FOREIGN KEY (run_id) REFERENCES model_training_runs (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_transaction_id ON fraud_alerts (transaction_id);
CREATE INDEX IF NOT EXISTS idx_kaggle_transactions_class_label ON kaggle_transactions (class_label);
CREATE INDEX IF NOT EXISTS idx_kaggle_transactions_source_file ON kaggle_transactions (source_file);
CREATE INDEX IF NOT EXISTS idx_model_training_runs_dataset_source ON model_training_runs (dataset_source);
CREATE INDEX IF NOT EXISTS idx_model_training_runs_selected_model ON model_training_runs (selected_model_name);
CREATE INDEX IF NOT EXISTS idx_model_candidate_metrics_run_id ON model_candidate_metrics (run_id);
CREATE INDEX IF NOT EXISTS idx_model_candidate_metrics_selected ON model_candidate_metrics (selected);
