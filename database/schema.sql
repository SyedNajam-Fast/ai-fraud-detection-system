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

CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_transaction_id ON fraud_alerts (transaction_id);
CREATE INDEX IF NOT EXISTS idx_kaggle_transactions_class_label ON kaggle_transactions (class_label);
CREATE INDEX IF NOT EXISTS idx_kaggle_transactions_source_file ON kaggle_transactions (source_file);
