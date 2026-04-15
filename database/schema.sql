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

CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_transaction_id ON fraud_alerts (transaction_id);
