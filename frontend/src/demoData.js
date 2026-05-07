export const DEMO_AI_DATASET_PREVIEW = {
  dataset_source: "synthetic:generated",
  sample_count: 1500,
  target_column: "fraud",
  fraud_rate: 0.128,
  normal_rate: 0.872,
  feature_cards: [
    {
      name: "amount",
      kind: "feature",
      description: "Transaction amount used to measure unusual spending size."
    },
    {
      name: "time",
      kind: "feature",
      description: "Hour of the day from 0 to 23 to capture suspicious timing patterns."
    },
    {
      name: "location",
      kind: "feature",
      description: "Transaction location category to highlight geographic risk patterns."
    },
    {
      name: "merchant",
      kind: "feature",
      description: "Merchant type to compare normal shopping with higher-risk categories."
    },
    {
      name: "fraud",
      kind: "target",
      description: "Target column where 1 means fraud and 0 means normal transaction."
    }
  ],
  sample_rows: [
    { amount: 277.76, time: 5, location: "Islamabad", merchant: "electronics_store", fraud: 0 },
    { amount: 370.94, time: 4, location: "Karachi", merchant: "online_retail", fraud: 0 },
    { amount: 245.62, time: 1, location: "Online", merchant: "restaurant", fraud: 0 },
    { amount: 221.26, time: 23, location: "Karachi", merchant: "online_retail", fraud: 0 },
    { amount: 401.32, time: 16, location: "Online", merchant: "electronics_store", fraud: 0 },
    { amount: 235.0, time: 18, location: "Karachi", merchant: "restaurant", fraud: 0 }
  ],
  manual_input_options: {
    defaults: {
      amount: 12450.75,
      time: 23,
      location: "Lahore",
      merchant: "electronics_store"
    },
    location_options: ["Islamabad", "Karachi", "Lahore", "Online", "Rawalpindi"],
    merchant_options: ["electronics_store", "grocery_store", "online_retail", "restaurant", "travel"]
  }
};

export const DEMO_RECOMMENDATION = {
  dataset_source: "synthetic:generated",
  recommendation_summary: {
    dataset_characteristics: {
      dataset_source: "synthetic:generated",
      sample_count: 1500,
      numeric_feature_count: 2,
      categorical_feature_count: 2,
      missing_ratio: 0.0,
      encoded_feature_estimate: 12,
      fraud_rate: 0.128,
      majority_class_ratio: 0.872,
      class_imbalance_detected: true
    },
    shortlisted_models: [
      {
        model_name: "logistic_regression",
        display_name: "Logistic Regression",
        score: 31.0,
        rationale:
          "It provides a strong interpretable baseline for fraud classification. The target distribution is imbalanced, so robust probability ranking matters.",
        shortlist_rank: 1,
        shortlisted: true
      },
      {
        model_name: "random_forest",
        display_name: "Random Forest",
        score: 28.0,
        rationale:
          "It handles mixed feature interactions well in tabular fraud data and is useful for non-linear behavior.",
        shortlist_rank: 2,
        shortlisted: true
      },
      {
        model_name: "extra_trees",
        display_name: "Extra Trees",
        score: 27.0,
        rationale:
          "It is useful for non-linear tabular patterns and fast ensemble comparisons in encoded feature spaces.",
        shortlist_rank: 3,
        shortlisted: true
      }
    ],
    all_models: [
      {
        model_name: "logistic_regression",
        display_name: "Logistic Regression",
        score: 31.0,
        rationale:
          "It provides a strong interpretable baseline for fraud classification. The target distribution is imbalanced, so robust probability ranking matters.",
        shortlist_rank: 1,
        shortlisted: true
      },
      {
        model_name: "random_forest",
        display_name: "Random Forest",
        score: 28.0,
        rationale:
          "It handles mixed feature interactions well in tabular fraud data and is useful for non-linear behavior.",
        shortlist_rank: 2,
        shortlisted: true
      },
      {
        model_name: "extra_trees",
        display_name: "Extra Trees",
        score: 27.0,
        rationale:
          "It is useful for non-linear tabular patterns and fast ensemble comparisons in encoded feature spaces.",
        shortlist_rank: 3,
        shortlisted: true
      },
      {
        model_name: "hist_gradient_boosting",
        display_name: "HistGradientBoosting",
        score: 21.0,
        rationale: "Boosting can benefit from mixed numeric and encoded categorical features.",
        shortlist_rank: null,
        shortlisted: false
      },
      {
        model_name: "gradient_boosting",
        display_name: "Gradient Boosting",
        score: 16.0,
        rationale: "Boosting can capture category-driven non-linear behavior after encoding.",
        shortlist_rank: null,
        shortlisted: false
      },
      {
        model_name: "svm",
        display_name: "Support Vector Machine",
        score: 11.0,
        rationale: "The dataset is small enough for an SVM to be computationally realistic.",
        shortlist_rank: null,
        shortlisted: false
      }
    ]
  }
};

export const DEMO_MODEL_DATA = {
  metadata: {
    dataset_source: "synthetic:generated",
    selected_model_name: "logistic_regression",
    selected_model_display_name: "Logistic Regression",
    selection_metric: "validation_average_precision",
    selected_threshold: 0.6762725380991143,
    sample_count: 1500,
    train_count: 900,
    validation_count: 300,
    test_count: 300,
    validation_metrics: {
      accuracy: 0.8766666666666667,
      precision: 0.5079365079365079,
      recall: 0.8421052631578947,
      f1: 0.6336633663366337,
      average_precision: 0.7172523789376343,
      roc_auc: 0.9303937324226597,
      confusion_matrix: [
        [231, 31],
        [6, 32]
      ]
    },
    test_metrics: {
      accuracy: 0.83,
      precision: 0.39344262295081966,
      recall: 0.631578947368421,
      f1: 0.48484848484848486,
      average_precision: 0.4146553927805579,
      roc_auc: 0.8492366412213741,
      confusion_matrix: [
        [225, 37],
        [14, 24]
      ]
    },
    overfit_flag: false,
    underfit_flag: false,
    shortlisted_models: [
      {
        model_name: "logistic_regression",
        display_name: "Logistic Regression",
        score: 31.0,
        shortlist_rank: 1,
        rationale:
          "It provides a strong interpretable baseline for fraud classification. The target distribution is imbalanced, so robust probability ranking matters."
      },
      {
        model_name: "random_forest",
        display_name: "Random Forest",
        score: 28.0,
        shortlist_rank: 2,
        rationale:
          "It handles mixed feature interactions well in tabular fraud data and is useful for non-linear behavior."
      },
      {
        model_name: "extra_trees",
        display_name: "Extra Trees",
        score: 27.0,
        shortlist_rank: 3,
        rationale:
          "It is useful for non-linear tabular patterns and fast ensemble comparisons in encoded feature spaces."
      }
    ],
    dataset_characteristics: DEMO_RECOMMENDATION.recommendation_summary.dataset_characteristics,
    full_model_pool: DEMO_RECOMMENDATION.recommendation_summary.all_models
  },
  latest_run: {
    id: 10,
    dataset_source: "synthetic:generated",
    selection_metric: "validation_average_precision",
    selected_model_name: "logistic_regression",
    selected_threshold: 0.6762725380991143,
    sample_count: 1500,
    train_count: 900,
    validation_count: 300,
    test_count: 300,
    train_f1: 0.5047923322683706,
    validation_f1: 0.6336633663366337,
    test_f1: 0.48484848484848486,
    validation_average_precision: 0.7172523789376343,
    test_average_precision: 0.4146553927805579,
    overfit_flag: 0,
    underfit_flag: 0,
    status: "completed"
  },
  recommendations: [
    {
      id: 16,
      run_id: 10,
      model_name: "logistic_regression",
      recommendation_rank: 1,
      recommendation_score: 31.0,
      rationale_text:
        "It provides a strong interpretable baseline for fraud classification. The target distribution is imbalanced, so robust probability ranking matters.",
      selected_for_training: 1,
      final_winner: 1
    },
    {
      id: 17,
      run_id: 10,
      model_name: "random_forest",
      recommendation_rank: 2,
      recommendation_score: 28.0,
      rationale_text:
        "It handles mixed feature interactions well in tabular fraud data and is useful for non-linear behavior.",
      selected_for_training: 1,
      final_winner: 0
    },
    {
      id: 18,
      run_id: 10,
      model_name: "extra_trees",
      recommendation_rank: 3,
      recommendation_score: 27.0,
      rationale_text:
        "It is useful for non-linear tabular patterns and fast ensemble comparisons in encoded feature spaces.",
      selected_for_training: 1,
      final_winner: 0
    }
  ]
};

export const DEMO_SCHEMA_DATA = {
  tables: [
    {
      table_name: "kaggle_transactions",
      layer: "raw_training",
      purpose: "Stores imported raw Kaggle fraud records for database-first model training.",
      primary_key_columns: ["id"],
      columns: ["id", "time_seconds", "amount", "v1", "v2", "v3", "v4", "v5", "class_label"],
      foreign_keys: [],
      index_names: ["idx_kaggle_transactions_class_label", "idx_kaggle_transactions_source_file"],
      simple_relationship_summary: "`kaggle_transactions` does not depend on another table through foreign keys."
    },
    {
      table_name: "raw_dataset_uploads",
      layer: "raw_profiling",
      purpose: "Registers every uploaded dataset file that has been profiled by the system.",
      primary_key_columns: ["id"],
      columns: ["id", "filename", "source_path", "file_size_bytes", "row_count", "column_count", "target_column", "status", "created_at"],
      foreign_keys: [],
      index_names: ["idx_raw_dataset_uploads_filename"],
      simple_relationship_summary: "`raw_dataset_uploads` does not depend on another table through foreign keys."
    },
    {
      table_name: "dataset_profiles",
      layer: "analytics",
      purpose: "Stores dataset-level profiling summaries such as missing values, duplicates, and class imbalance.",
      primary_key_columns: ["id"],
      columns: ["id", "upload_id", "row_count", "column_count", "duplicate_row_count", "missing_cell_count", "numeric_column_count", "categorical_column_count", "warnings_json"],
      foreign_keys: [{ from_column: "upload_id", reference_table: "raw_dataset_uploads", reference_column: "id" }],
      index_names: ["idx_dataset_profiles_upload_id"],
      simple_relationship_summary: "`dataset_profiles` connects to other tables through: `upload_id` -> `raw_dataset_uploads.id`."
    },
    {
      table_name: "feature_profiles",
      layer: "analytics",
      purpose: "Stores column-level profiling details and plain-language explanations for every dataset field.",
      primary_key_columns: ["id"],
      columns: ["id", "upload_id", "dataset_profile_id", "column_name", "inferred_role", "pandas_dtype", "sample_values_json"],
      foreign_keys: [
        { from_column: "dataset_profile_id", reference_table: "dataset_profiles", reference_column: "id" },
        { from_column: "upload_id", reference_table: "raw_dataset_uploads", reference_column: "id" }
      ],
      index_names: ["idx_feature_profiles_upload_id", "idx_feature_profiles_profile_id", "idx_feature_profiles_column_name"],
      simple_relationship_summary:
        "`feature_profiles` connects to other tables through: `dataset_profile_id` -> `dataset_profiles.id`, `upload_id` -> `raw_dataset_uploads.id`."
    },
    {
      table_name: "users",
      layer: "operational",
      purpose: "Stores cardholder identity records so customer details are not repeated in every transaction row.",
      primary_key_columns: ["id"],
      columns: ["id", "name", "email", "card_number"],
      foreign_keys: [],
      index_names: ["sqlite_autoindex_users_1", "sqlite_autoindex_users_2"],
      simple_relationship_summary: "`users` does not depend on another table through foreign keys."
    },
    {
      table_name: "transactions",
      layer: "operational",
      purpose: "Stores the core transaction facts that the fraud workflow evaluates.",
      primary_key_columns: ["id"],
      columns: ["id", "user_id", "amount", "time", "location", "merchant"],
      foreign_keys: [{ from_column: "user_id", reference_table: "users", reference_column: "id" }],
      index_names: ["idx_transactions_user_id"],
      simple_relationship_summary: "`transactions` connects to other tables through: `user_id` -> `users.id`."
    },
    {
      table_name: "predictions",
      layer: "operational",
      purpose: "Stores the model decision and fraud probability for each processed transaction.",
      primary_key_columns: ["id"],
      columns: ["id", "transaction_id", "prediction", "probability"],
      foreign_keys: [{ from_column: "transaction_id", reference_table: "transactions", reference_column: "id" }],
      index_names: ["idx_predictions_transaction_id"],
      simple_relationship_summary: "`predictions` connects to other tables through: `transaction_id` -> `transactions.id`."
    },
    {
      table_name: "fraud_alerts",
      layer: "operational",
      purpose: "Stores alert records for transactions that were marked as suspicious.",
      primary_key_columns: ["id"],
      columns: ["id", "transaction_id", "alert_time", "status"],
      foreign_keys: [{ from_column: "transaction_id", reference_table: "transactions", reference_column: "id" }],
      index_names: ["idx_alerts_transaction_id"],
      simple_relationship_summary: "`fraud_alerts` connects to other tables through: `transaction_id` -> `transactions.id`."
    },
    {
      table_name: "model_training_runs",
      layer: "analytics",
      purpose: "Stores one summary row for each model training session and selected winner.",
      primary_key_columns: ["id"],
      columns: ["id", "dataset_source", "selected_model_name", "selected_threshold", "sample_count", "validation_f1", "test_f1"],
      foreign_keys: [],
      index_names: ["idx_model_training_runs_dataset_source", "idx_model_training_runs_selected_model"],
      simple_relationship_summary: "`model_training_runs` does not depend on another table through foreign keys."
    },
    {
      table_name: "model_recommendations",
      layer: "analytics",
      purpose: "Stores the three shortlisted models, their recommendation rank, and the reason they were chosen.",
      primary_key_columns: ["id"],
      columns: ["id", "run_id", "model_name", "recommendation_rank", "recommendation_score", "rationale_text"],
      foreign_keys: [{ from_column: "run_id", reference_table: "model_training_runs", reference_column: "id" }],
      index_names: ["idx_model_recommendations_run_id", "idx_model_recommendations_rank"],
      simple_relationship_summary: "`model_recommendations` connects to other tables through: `run_id` -> `model_training_runs.id`."
    },
    {
      table_name: "model_candidate_metrics",
      layer: "analytics",
      purpose: "Stores evaluation details for every candidate model considered in a training run.",
      primary_key_columns: ["id"],
      columns: ["id", "run_id", "model_name", "validation_f1", "validation_average_precision", "selected"],
      foreign_keys: [{ from_column: "run_id", reference_table: "model_training_runs", reference_column: "id" }],
      index_names: ["idx_model_candidate_metrics_run_id", "idx_model_candidate_metrics_selected"],
      simple_relationship_summary: "`model_candidate_metrics` connects to other tables through: `run_id` -> `model_training_runs.id`."
    }
  ],
  normalization_summary: [
    "The schema separates operational transactions from predictions and alerts, which avoids storing model output inside the base transaction row.",
    "The schema separates training-run summaries from candidate-level metrics, which preserves one-to-many model comparison history without duplication.",
    "The schema also stores recommendation rows separately so model selection reasoning is preserved before final winner evaluation.",
    "The profiling layer separates raw uploaded-file metadata, dataset-level summaries, and feature-level explanations into different tables.",
    "The operational flow is close to 3NF because user identity is stored in `users` while transaction facts live in `transactions`.",
    "The schema is not fully normalized because `location` and `merchant` are still plain text in `transactions`; these could later move into lookup tables.",
    "`kaggle_transactions` is intentionally denormalized because it preserves imported training rows in the same wide shape as the source dataset."
  ],
  simple_overview:
    "This database is split into operational tables for live fraud workflow, raw-data tables for imported files, and analytics tables for profiling and model history.",
  technical_overview:
    "The schema currently contains 11 application tables. Operational tables: fraud_alerts, predictions, transactions, users. Analytics and audit tables: dataset_profiles, feature_profiles, model_candidate_metrics, model_recommendations, model_training_runs.",
  mermaid_er_diagram: `erDiagram
    RAW_DATASET_UPLOADS ||--o{ DATASET_PROFILES : references
    DATASET_PROFILES ||--o{ FEATURE_PROFILES : references
    RAW_DATASET_UPLOADS ||--o{ FEATURE_PROFILES : references
    TRANSACTIONS ||--o{ FRAUD_ALERTS : references
    MODEL_TRAINING_RUNS ||--o{ MODEL_CANDIDATE_METRICS : references
    MODEL_TRAINING_RUNS ||--o{ MODEL_RECOMMENDATIONS : references
    TRANSACTIONS ||--o{ PREDICTIONS : references
    USERS ||--o{ TRANSACTIONS : references
    DATASET_PROFILES {
        int id PK
        int upload_id
        int row_count
        int column_count
        int duplicate_row_count
        int missing_cell_count
        int numeric_column_count
        int categorical_column_count
        string additional_fields
    }
    FEATURE_PROFILES {
        int id PK
        int upload_id
        int dataset_profile_id
        string column_name
        string inferred_role
        string inferred_dtype
        string pandas_dtype
        int non_null_count
        string additional_fields
    }
    FRAUD_ALERTS {
        int id PK
        int transaction_id
        string alert_time
        string status
    }
    KAGGLE_TRANSACTIONS {
        int id PK
        int time_seconds
        float amount
        float v1
        float v2
        float v3
        float v4
        float v5
        string additional_fields
    }
    MODEL_CANDIDATE_METRICS {
        int id PK
        int run_id
        string model_name
        float cv_f1_mean
        float cv_f1_std
        float train_precision
        float train_recall
        float train_f1
        string additional_fields
    }
    MODEL_RECOMMENDATIONS {
        int id PK
        int run_id
        string model_name
        int recommendation_rank
        float recommendation_score
        string rationale_text
        int selected_for_training
        int final_winner
        string additional_fields
    }
    MODEL_TRAINING_RUNS {
        int id PK
        string dataset_source
        string selection_metric
        string selected_model_name
        float selected_threshold
        int sample_count
        int train_count
        int validation_count
        string additional_fields
    }
    PREDICTIONS {
        int id PK
        int transaction_id
        int prediction
        float probability
    }
    RAW_DATASET_UPLOADS {
        int id PK
        string filename
        string source_path
        int file_size_bytes
        int row_count
        int column_count
        string target_column
        string status
        string additional_fields
    }
    TRANSACTIONS {
        int id PK
        int user_id
        float amount
        int time
        string location
        string merchant
    }
    USERS {
        int id PK
        string name
        string email
        string card_number
    }`
};

export const DEMO_PRESENTATION_DATA = {
  latest_profile: {
    upload: {
      id: 6,
      filename: "sample_profile_dataset.csv",
      source_path:
        "F:\\Fast Material\\Semester 4\\projects\\DBS-AI\\ai-fraud-detection-system\\data\\samples\\sample_profile_dataset.csv",
      file_size_bytes: 521,
      row_count: 8,
      column_count: 10,
      target_column: "Class",
      status: "profiled"
    },
    dataset_profile: {
      id: 6,
      upload_id: 6,
      row_count: 8,
      column_count: 10,
      duplicate_row_count: 1,
      missing_cell_count: 1,
      numeric_column_count: 4,
      categorical_column_count: 6,
      target_column: "Class",
      warnings_json: JSON.stringify([
        "The dataset contains 1 duplicate rows.",
        "The dataset contains 1 missing cells.",
        "The dataset is small, so model evaluation may be unstable."
      ])
    },
    feature_profiles: [
      {
        column_name: "transaction_id",
        inferred_role: "identifier",
        pandas_dtype: "int64",
        sample_values_json: JSON.stringify(["1001", "1002", "1003"])
      },
      {
        column_name: "amount",
        inferred_role: "amount",
        pandas_dtype: "float64",
        sample_values_json: JSON.stringify(["1250.5", "85.0", "420.75"])
      },
      {
        column_name: "time",
        inferred_role: "time",
        pandas_dtype: "int64",
        sample_values_json: JSON.stringify(["23", "14", "3"])
      },
      {
        column_name: "location",
        inferred_role: "categorical_feature",
        pandas_dtype: "object",
        sample_values_json: JSON.stringify(["Lahore", "Karachi", "Islamabad"])
      },
      {
        column_name: "merchant",
        inferred_role: "categorical_feature",
        pandas_dtype: "object",
        sample_values_json: JSON.stringify(["electronics_store", "restaurant", "travel"])
      },
      {
        column_name: "Class",
        inferred_role: "target",
        pandas_dtype: "int64",
        sample_values_json: JSON.stringify(["0", "1"])
      }
    ]
  },
  diagrams: [
    {
      id: "erd",
      title: "ER Diagram",
      description: "Database relationships taken from the live schema explanation layer.",
      mermaid: DEMO_SCHEMA_DATA.mermaid_er_diagram,
      course_focus: "DBS",
      talking_points: [
        "Use this when the DBS instructor asks about primary keys, foreign keys, and normalization.",
        "It reflects the live schema explanation output instead of a separate hand-drawn database story."
      ]
    }
  ],
  viva_notes: [
    {
      question: "Why did you normalize this database design?",
      answer:
        "I separated users, transactions, predictions, alerts, profiling summaries, and model history so repeated information would not be stored in one large table."
    },
    {
      question: "Why are location and merchant still text columns in transactions?",
      answer:
        "For the current project scope I kept them simple to reduce implementation overhead, but they can be moved into lookup tables later for stronger normalization."
    },
    {
      question: "How do the database and AI parts connect?",
      answer:
        "The database stores profiled datasets, transactions, predictions, alerts, training runs, shortlist recommendations, and candidate metrics, while the AI layer reads training data and writes audited results back into SQLite."
    }
  ]
};

const OFFLINE_TEST_SAMPLES = [
  {
    input: { amount: 8200.5, time: 23, location: "Online", merchant: "electronics_store" },
    actual_label: 1
  },
  {
    input: { amount: 145.25, time: 13, location: "Lahore", merchant: "grocery_store" },
    actual_label: 0
  },
  {
    input: { amount: 3100.0, time: 2, location: "Karachi", merchant: "travel" },
    actual_label: 1
  }
];

function clampProbability(value) {
  return Math.max(0.02, Math.min(0.98, value));
}

function buildRiskSignals(input) {
  const signals = [];
  if (Number(input.amount) >= 5000) {
    signals.push("High transaction amount");
  }
  if (Number(input.time) <= 4 || Number(input.time) >= 22) {
    signals.push("Unusual transaction hour");
  }
  if (String(input.location) === "Online") {
    signals.push("Remote transaction channel");
  }
  if (["electronics_store", "travel", "online_retail"].includes(String(input.merchant))) {
    signals.push("Higher-risk merchant category");
  }
  if (signals.length === 0) {
    signals.push("Pattern looks close to normal behavior");
  }
  return signals;
}

function buildOfflineProbability(input) {
  let score = 0.08;
  if (Number(input.amount) >= 5000) {
    score += 0.42;
  } else if (Number(input.amount) >= 2000) {
    score += 0.18;
  }
  if (Number(input.time) <= 4 || Number(input.time) >= 22) {
    score += 0.18;
  }
  if (String(input.location) === "Online") {
    score += 0.12;
  }
  if (["electronics_store", "travel", "online_retail"].includes(String(input.merchant))) {
    score += 0.14;
  }
  return clampProbability(score);
}

export function buildOfflineManualPrediction(input) {
  const probability = buildOfflineProbability(input);
  const threshold = DEMO_MODEL_DATA.metadata.selected_threshold;
  const prediction = probability >= threshold ? 1 : 0;
  const riskSignals = buildRiskSignals(input);
  const confidenceGap = Math.abs(probability - threshold);
  const confidenceBand = confidenceGap >= 0.25 ? "High" : confidenceGap >= 0.1 ? "Medium" : "Low";

  return {
    prediction,
    prediction_label: prediction === 1 ? "Potential Fraud" : "Normal Transaction",
    probability,
    threshold,
    model_name: DEMO_MODEL_DATA.metadata.selected_model_display_name,
    confidence_band: confidenceBand,
    risk_signals: riskSignals,
    message:
      prediction === 1
        ? "The offline demo model marks this transaction as suspicious because several risk signals appear together."
        : "The offline demo model keeps this transaction in the normal class because the risk signals stay limited."
  };
}

export function buildOfflineTestSample(index) {
  const safeIndex = index % OFFLINE_TEST_SAMPLES.length;
  const sample = OFFLINE_TEST_SAMPLES[safeIndex];
  const predictionPayload = buildOfflineManualPrediction(sample.input);
  const correct = predictionPayload.prediction === sample.actual_label;

  return {
    correct,
    sample_index: safeIndex,
    total_test_samples: OFFLINE_TEST_SAMPLES.length,
    prediction_label: predictionPayload.prediction_label,
    actual_label_text: sample.actual_label === 1 ? "Fraud" : "Normal",
    probability: predictionPayload.probability,
    review_text: correct
      ? "The offline demo prediction matches the stored label for this held-out sample."
      : "The offline demo prediction does not match the stored label for this held-out sample.",
    input: sample.input,
    next_index: (safeIndex + 1) % OFFLINE_TEST_SAMPLES.length
  };
}

export const DEMO_SDA_CONTENT = {
  introduction:
    "This project is a local fraud-detection platform that connects dataset understanding, database storage, machine-learning evaluation, and presentation-oriented explanations in one system.",
  problemStatement:
    "Traditional semester projects often show only a model or only a database. This project solves that gap by combining data storage, fraud prediction, alert generation, and a presentable interface that explains how the full system works.",
  methodology: [
    "Profile or choose the source dataset and understand the important fields.",
    "Store structured records in SQLite and keep profiling plus audit history separately.",
    "Recommend and compare multiple machine-learning models for fraud detection.",
    "Save predictions and create alerts so the workflow is observable end to end.",
    "Present the result through a simple tab-based interface for AI, DBS, and SDA."
  ],
  architecture: [
    "Frontend layer: React + Vite presentation UI with separate academic tabs.",
    "API layer: FastAPI endpoints for schema explanation, model information, and prediction actions.",
    "Service layer: profiling, workflow, presentation, and schema explanation modules.",
    "Persistence layer: SQLite stores transactions, predictions, alerts, profiles, and model audit data.",
    "ML layer: scikit-learn pipelines train and evaluate candidate fraud models."
  ],
  toolsAndTechniques: [
    { name: "Python", detail: "Core backend language for API, workflow, and machine-learning logic." },
    { name: "FastAPI", detail: "Creates clean REST endpoints for frontend and presentation actions." },
    { name: "SQLite", detail: "Stores normalized operational data and model audit history locally." },
    { name: "scikit-learn", detail: "Builds preprocessing pipelines, model comparison, and evaluation metrics." },
    { name: "React + Vite", detail: "Delivers a fast presentation interface with reusable components." },
    { name: "Mermaid", detail: "Renders academic diagrams directly inside the UI for viva explanation." }
  ],
  guiDesign: [
    "Three subject-wise tabs keep the presentation focused: AI, DBS, and SDA.",
    "Large hero summary gives quick context before technical details.",
    "Cards, metrics, tables, and diagrams separate complex content into explainable blocks.",
    "Manual prediction form validates structured inputs instead of free-form text.",
    "Offline presentation mode keeps the UI usable even if the local backend is unavailable."
  ],
  validations: [
    "Transaction amount accepts only numeric values.",
    "Transaction time is limited to 0 through 23.",
    "Location and merchant inputs are restricted to known options.",
    "Backend errors are shown through status banners instead of silent failure.",
    "Operational database schema uses primary keys, foreign keys, unique constraints, and checks."
  ],
  functionalRequirements: [
    "The system shall profile source data and show sample records.",
    "The system shall store users, transactions, predictions, and fraud alerts in SQLite.",
    "The system shall recommend candidate fraud models and show the selected winner.",
    "The system shall allow manual transaction prediction through the UI.",
    "The system shall show diagrams and explanations for AI, DBS, and SDA presentation."
  ],
  nonFunctionalRequirements: [
    "The UI should remain simple, explainable, and presentation-friendly.",
    "The application should run locally on one machine without external infrastructure.",
    "The system should preserve auditability through stored model and dataset metadata.",
    "The design should remain modular so each layer can be explained independently.",
    "The interface should provide clear feedback when the backend is unavailable."
  ],
  outsourceLibraries: [
    { name: "react", purpose: "Component-based frontend rendering." },
    { name: "react-dom", purpose: "Browser DOM integration for the React UI." },
    { name: "lucide-react", purpose: "Consistent icon set across the dashboard." },
    { name: "mermaid", purpose: "Diagram rendering inside the SDA and DBS tabs." },
    { name: "fastapi", purpose: "High-level API framework for backend routes." },
    { name: "uvicorn", purpose: "ASGI server to run the FastAPI application." },
    { name: "pandas", purpose: "Dataset profiling and tabular data processing." },
    { name: "scikit-learn", purpose: "Model training, preprocessing, and evaluation." }
  ],
  testCases: [
    {
      id: "TC-01",
      scenario: "Manual prediction with valid input",
      input: "Amount=12450.75, Time=23, Location=Lahore, Merchant=electronics_store",
      expected: "Prediction result and probability are shown.",
      result: "Pass"
    },
    {
      id: "TC-02",
      scenario: "Time validation",
      input: "Time outside 0 to 23",
      expected: "User should not be able to submit invalid time.",
      result: "Pass"
    },
    {
      id: "TC-03",
      scenario: "Backend unavailable",
      input: "Frontend opened without API server",
      expected: "UI should stay usable in offline presentation mode.",
      result: "Pass"
    },
    {
      id: "TC-04",
      scenario: "DBS ER diagram rendering",
      input: "Open DBS tab",
      expected: "ER diagram and normalization story should render correctly.",
      result: "Pass"
    },
    {
      id: "TC-05",
      scenario: "SDA diagram rendering",
      input: "Open SDA tab",
      expected: "Use case, activity, sequence, collaboration, component, and state diagrams should render.",
      result: "Pass"
    }
  ],
  diagrams: [
    {
      id: "use_case",
      title: "Use Case Diagram",
      description: "Shows what the student or instructor can do in the system during the demo.",
      mermaid: `flowchart LR
    User([Student / Instructor])
    Select[Choose dataset or demo snapshot]
    ExplainDB[Explain database design]
    Train[Train or review model]
    Predict[Run fraud prediction]
    Review[Review results and diagrams]
    User --> Select
    User --> ExplainDB
    User --> Train
    User --> Predict
    User --> Review`,
      talkingPoint: "Use this first to explain the complete user-facing scope of the project."
    },
    {
      id: "activity",
      title: "Activity Diagram",
      description: "Explains the step-by-step project workflow from dataset understanding to final fraud alert.",
      mermaid: `flowchart TD
    A[Start presentation] --> B[Choose dataset or offline snapshot]
    B --> C[Profile data and inspect features]
    C --> D[Explain schema and normalization]
    D --> E[Recommend candidate models]
    E --> F[Train and evaluate winner]
    F --> G[Predict transaction]
    G --> H{Fraud?}
    H -->|Yes| I[Create alert]
    H -->|No| J[Store normal prediction]
    I --> K[Show diagrams and final explanation]
    J --> K`,
      talkingPoint: "This is the best diagram for methodology and workflow explanation."
    },
    {
      id: "network",
      title: "Network / Deployment Diagram",
      description: "Shows how the system runs locally across browser, frontend server, backend API, and storage.",
      mermaid: `flowchart TD
    Laptop[Local laptop]
    Browser[Browser :5173]
    Frontend[React/Vite frontend]
    Backend[FastAPI backend :8000]
    DB[(SQLite database)]
    Model[(Model artifacts)]
    Laptop --> Browser
    Browser --> Frontend
    Frontend --> Backend
    Backend --> DB
    Backend --> Model`,
      talkingPoint: "Use this to explain the local deployment and why the project is easy to demonstrate."
    },
    {
      id: "sequence",
      title: "Sequence Diagram",
      description: "Details the runtime message flow between UI, API, services, database, and model files.",
      mermaid: `sequenceDiagram
    participant U as User
    participant UI as React UI
    participant API as FastAPI
    participant S as Workflow Service
    participant DB as SQLite
    participant M as Model
    U->>UI: Enter transaction / open tab
    UI->>API: Request prediction or schema data
    API->>S: Invoke business logic
    S->>DB: Read or store records
    S->>M: Load model / metadata
    M-->>S: Prediction output
    S-->>API: Structured response
    API-->>UI: Render result`,
      talkingPoint: "This explains in detail how the system works internally during runtime."
    },
    {
      id: "interaction",
      title: "Interaction Diagram",
      description: "Focuses on how the user interacts with the three presentation tabs and system actions.",
      mermaid: `flowchart LR
    User([Presenter]) --> AI[AI tab]
    User --> DBS[DBS tab]
    User --> SDA[SDA tab]
    AI --> Train[Model analysis]
    AI --> Manual[Manual prediction]
    DBS --> ERD[Schema + ER diagram]
    SDA --> Arch[Architecture + diagrams]
    Manual --> Result[Prediction result]
    ERD --> Result
    Arch --> Result`,
      talkingPoint: "Use this when you want to explain tab-wise interaction rather than backend internals."
    },
    {
      id: "collaboration",
      title: "Collaboration Diagram",
      description: "Shows how modules collaborate to complete one fraud-detection request.",
      mermaid: `flowchart LR
    UI[1. UI] --> API[2. API layer]
    API --> Schema[3. Schema service]
    API --> Workflow[3. Workflow service]
    API --> Present[3. Presentation service]
    Workflow --> DB[4. SQLite]
    Workflow --> Model[4. Model artifact]
    Schema --> DB
    Present --> DB
    Model --> API[5. Prediction / metrics]
    DB --> API[5. Stored records]
    API --> UI[6. Final response]`,
      talkingPoint: "This is the collaboration view of how modules help each other during one request."
    },
    {
      id: "component",
      title: "Component Diagram",
      description: "Explains the software modules and how the project is divided architecturally.",
      mermaid: `flowchart LR
    subgraph Frontend
      App[App.jsx]
      Mermaid[MermaidDiagram]
      Demo[Offline demo data]
    end
    subgraph Backend
      Api[api/app.py]
      Workflow[services/workflow.py]
      Schema[services/schema_explainer.py]
      Presentation[services/presentation_support.py]
    end
    subgraph Storage
      Sql[(SQLite)]
      Files[(Model + metadata)]
    end
    App --> Api
    App --> Mermaid
    App --> Demo
    Api --> Workflow
    Api --> Schema
    Api --> Presentation
    Workflow --> Sql
    Workflow --> Files
    Schema --> Sql
    Presentation --> Sql`,
      talkingPoint: "Use this for the application architecture section."
    },
    {
      id: "state",
      title: "State Diagram",
      description: "Shows how a transaction or alert changes state as the workflow proceeds.",
      mermaid: `stateDiagram-v2
    [*] --> Captured
    Captured --> Stored
    Stored --> Predicted
    Predicted --> Normal : prediction = 0
    Predicted --> AlertOpen : prediction = 1
    AlertOpen --> Investigating
    Investigating --> Resolved
    Investigating --> Dismissed
    Normal --> [*]
    Resolved --> [*]
    Dismissed --> [*]`,
      talkingPoint: "This is useful when explaining alert lifecycle and working project behavior."
    }
  ]
};
