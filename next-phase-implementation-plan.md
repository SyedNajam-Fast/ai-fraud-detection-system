# Next Phase Implementation Plan

## Implementation Status

- Status: Implemented and validated in current workspace session.
- Validation summary:
  - Imported Kaggle records into `kaggle_transactions` with total rows: 284,807.
  - Training now uses database source `database:kaggle_transactions` when available.
  - End-to-end workflow still runs successfully with `src/main.py --force-train`.

## 1) Progress Against Original Plan

### Completed
- Core relational schema exists for users, transactions, predictions, and fraud alerts.
- End-to-end workflow exists: insert transaction, fetch, predict, store prediction, create alert.
- Random Forest training and model persistence are implemented.
- Kaggle dataset download flow is implemented and documented.

### Partially Completed
- Database is used in runtime flow, but currently only for demo transaction records.
- Real dataset is downloaded, but not yet ingested into database tables.

### Not Completed
- Database-backed ingestion pipeline for real Kaggle data.
- Database-backed training data source.
- Schema-aligned mapping strategy between Kaggle creditcard.csv columns and project training/transaction schema.
- Validation reporting for real-data pipeline consistency.

## 2) Gap Summary

The project satisfies the context architecture for a demo transaction flow. The main gap is moving from demo data to real data while keeping the database as the source of truth.

## 3) Next Phase Objective

Implement a database-first real-data pipeline:
- Import Kaggle dataset into database staging/training tables.
- Train from database data (with controlled fallback behavior).
- Keep the current end-to-end transaction workflow intact.

## 4) Scope for Next Phase

### In Scope
- Schema extension for training/staging data.
- New ingestion script from data/raw/creditcardfraud/creditcard.csv into DB.
- Training script update to read from DB first.
- Documentation and verification commands.

### Out of Scope
- UI/frontend.
- Deep learning models.
- Full architecture rewrite.

## 5) Execution Plan

### Step 1: Add DB tables for real-data training
- Update database/schema.sql with one new table for imported Kaggle records.
- Include indexes for efficient loading/training reads.

Proposed table (conceptual):
- kaggle_transactions
  - id
  - time_seconds
  - amount
  - v1 ... v28
  - class_label
  - source_file
  - imported_at

### Step 2: Extend DB helpers
- Add functions in src/db.py for:
  - bulk insert of Kaggle rows
  - row counts and source checks
  - optional table reset for re-import

### Step 3: Build ingestion script
- Add src/import_kaggle_to_db.py:
  - read data/raw/creditcardfraud/creditcard.csv
  - validate required columns
  - normalize column names
  - batch insert into DB
  - print import summary

### Step 4: Enable training-from-database
- Update model/train_model.py to load from DB table when available.
- Keep existing CSV/synthetic fallback for resilience.
- Print dataset source used in each train run.

### Step 5: Keep runtime workflow stable
- Ensure src/main.py remains functional without requiring full Kaggle feature payload.
- Continue using existing demo transaction flow for end-to-end architecture demonstration.

### Step 6: Verification and acceptance
- Verify import counts in database.
- Verify training runs with DB source and saves model.
- Verify src/main.py still performs prediction storage and alert creation.

## 6) Acceptance Criteria

- Import script successfully loads Kaggle records into DB table.
- Training script can run from DB data without manual CSV path handling.
- Existing workflow still executes successfully from src/main.py.
- README includes concise run instructions for import plus training.

## 7) Estimated Effort

- Implementation: 1 focused coding session.
- Validation and cleanup: 0.5 session.
- Total: 1.5 sessions.

## 8) Risks and Mitigation

- Risk: Feature mismatch between runtime transaction schema and Kaggle anonymized features.
- Mitigation: Keep runtime workflow unchanged for architecture demo, and isolate Kaggle usage in training table/pipeline.

- Risk: Large insert operations may be slow.
- Mitigation: Use batched inserts and transaction commits per chunk.
