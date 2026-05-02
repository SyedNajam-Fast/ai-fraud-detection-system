import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Database,
  Gauge,
  GraduationCap,
  Play,
  RefreshCw,
  Rows3,
  WandSparkles
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000/api";

const sectionTabs = [
  {
    id: "ai",
    label: "AI",
    subtitle: "Dataset, model choice, training, and live prediction",
    icon: Brain
  },
  {
    id: "db",
    label: "DB",
    subtitle: "Database section reserved for your next prompt",
    icon: Database
  },
  {
    id: "sda",
    label: "SDA",
    subtitle: "SDA section reserved for your next prompt",
    icon: GraduationCap
  }
];

const aiSteps = [
  { number: "01", title: "Understand the dataset", icon: Rows3 },
  { number: "02", title: "Justify the model choice", icon: Brain },
  { number: "03", title: "Train and evaluate", icon: Activity },
  { number: "04", title: "Try manual input", icon: Play },
  { number: "05", title: "Test on unseen data", icon: Gauge }
];

const defaultManualInput = {
  amount: 12450.75,
  time: 23,
  location: "Lahore",
  merchant: "electronics_store"
};
const API_RETRY_DELAY_MS = 5000;
const MAX_API_RETRIES = 12;

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatScore(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toFixed(digits);
}

function formatCount(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toLocaleString();
}

function titleCase(value) {
  if (!value) {
    return "N/A";
  }
  return String(value)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatSource(value) {
  if (!value) {
    return "Not available";
  }
  const [kind, name] = String(value).split(":");
  if (!name) {
    return titleCase(value);
  }
  return `${titleCase(kind)} / ${titleCase(name)}`;
}

function statusText(flag) {
  if (flag === null || flag === undefined) {
    return "N/A";
  }
  return flag ? "Flagged" : "Passed";
}

function buildSelectedModelStory(metadata, shortlist) {
  if (!metadata?.selected_model_name) {
    return "The shortlist is ready. Train the models to show which option performs best on validation and test data.";
  }

  const winningShortlistItem =
    shortlist.find((item) => item.model_name === metadata.selected_model_name) || shortlist[0] || null;

  if (winningShortlistItem?.rationale) {
    return winningShortlistItem.rationale;
  }

  return "The selected model gave the strongest validation performance among the shortlisted options.";
}

function SectionTab({ icon: Icon, label, subtitle, active, onClick }) {
  return (
    <button className={`section-tab ${active ? "active" : ""}`} onClick={onClick}>
      <div className="section-tab-icon">
        <Icon size={18} />
      </div>
      <div>
        <strong>{label}</strong>
        <span>{subtitle}</span>
      </div>
    </button>
  );
}

function HeroMetric({ label, value }) {
  return (
    <article className="hero-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function StorySection({ step, title, icon: Icon, intro, children }) {
  return (
    <section className="story-section">
      <div className="story-heading">
        <div className="story-step-tag">
          <span>{step}</span>
          <Icon size={18} />
        </div>
        <div>
          <h2>{title}</h2>
          <p>{intro}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function MetricCard({ label, value, tone = "neutral" }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ConfusionMatrix({ matrix }) {
  const safeMatrix =
    Array.isArray(matrix) &&
    matrix.length === 2 &&
    Array.isArray(matrix[0]) &&
    Array.isArray(matrix[1]) &&
    matrix[0].length === 2 &&
    matrix[1].length === 2
      ? matrix
      : null;

  return (
    <div className="matrix-card">
      <div className="matrix-copy">
        <strong>Evaluation Matrix</strong>
        <p>
          This is the held-out test confusion matrix. It shows how many normal and fraud cases were
          classified correctly and incorrectly.
        </p>
      </div>
      {safeMatrix ? (
        <div className="matrix-grid">
          <div className="matrix-axis blank" />
          <div className="matrix-axis">Predicted Normal</div>
          <div className="matrix-axis">Predicted Fraud</div>
          <div className="matrix-axis">Actual Normal</div>
          <div className="matrix-cell success">{safeMatrix[0][0]}</div>
          <div className="matrix-cell warning">{safeMatrix[0][1]}</div>
          <div className="matrix-axis">Actual Fraud</div>
          <div className="matrix-cell warning">{safeMatrix[1][0]}</div>
          <div className="matrix-cell success">{safeMatrix[1][1]}</div>
        </div>
      ) : (
        <p className="empty-copy">Train the model to populate the confusion matrix.</p>
      )}
    </div>
  );
}

function PlaceholderSection({ icon: Icon, title, description, cards }) {
  return (
    <main className="placeholder-shell">
      <section className="placeholder-hero">
        <div className="placeholder-icon">
          <Icon size={24} />
        </div>
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </section>

      <section className="placeholder-grid">
        {cards.map((card) => (
          <article key={card.title} className="placeholder-card">
            <strong>{card.title}</strong>
            <p>{card.text}</p>
          </article>
        ))}
      </section>
    </main>
  );
}

function App() {
  const [activeSection, setActiveSection] = useState("ai");
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [manualInput, setManualInput] = useState(defaultManualInput);
  const [manualResult, setManualResult] = useState(null);
  const [testSample, setTestSample] = useState(null);
  const [testSampleIndex, setTestSampleIndex] = useState(0);
  const [error, setError] = useState("");
  const [loadingKey, setLoadingKey] = useState("");
  const [connectionNote, setConnectionNote] = useState("");
  const retryTimeoutRef = useRef(null);
  const retryAttemptRef = useRef(0);

  useEffect(() => {
    void loadAiView();
    return () => {
      clearRetryTimer();
    };
  }, []);

  async function apiFetch(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }
    return payload;
  }

  function clearRetryTimer() {
    if (retryTimeoutRef.current) {
      window.clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }

  function scheduleRetry() {
    if (retryAttemptRef.current >= MAX_API_RETRIES || retryTimeoutRef.current) {
      return;
    }

    retryAttemptRef.current += 1;
    setConnectionNote(
      `Backend is starting on ${API_BASE}. Retrying automatically in ${API_RETRY_DELAY_MS / 1000} seconds (${retryAttemptRef.current}/${MAX_API_RETRIES}).`
    );
    retryTimeoutRef.current = window.setTimeout(() => {
      retryTimeoutRef.current = null;
      void loadAiView(true);
    }, API_RETRY_DELAY_MS);
  }

  async function loadAiView(isRetry = false) {
    const requestSpecs = [
      { key: "datasetPreview", label: "dataset preview", path: "/ai/dataset-preview" },
      { key: "recommendation", label: "model recommendation", path: "/recommendations/current" },
      { key: "modelData", label: "saved model state", path: "/model/latest" }
    ];

    try {
      setLoadingKey(isRetry ? "retrying-api-connection" : "loading-ai-story");
      setError("");

      const results = await Promise.allSettled(requestSpecs.map((item) => apiFetch(item.path)));
      const failedLabels = [];
      let successCount = 0;

      results.forEach((result, index) => {
        const spec = requestSpecs[index];
        if (result.status === "fulfilled") {
          successCount += 1;
          if (spec.key === "datasetPreview") {
            setDatasetPreview(result.value);
            const defaults = result.value?.manual_input_options?.defaults;
            if (defaults) {
              setManualInput(defaults);
            }
          }
          if (spec.key === "recommendation") {
            setRecommendation(result.value);
          }
          if (spec.key === "modelData") {
            setModelData(result.value);
          }
          return;
        }

        failedLabels.push(spec.label);
      });

      if (successCount > 0) {
        clearRetryTimer();
        retryAttemptRef.current = 0;
        setConnectionNote("");
      }

      if (failedLabels.length === requestSpecs.length) {
        setError(`Backend is not ready yet at ${API_BASE}.`);
        scheduleRetry();
        return;
      }

      if (failedLabels.length > 0) {
        setError(`Some AI sections are still loading: ${failedLabels.join(", ")}.`);
      } else {
        setError("");
      }
    } catch (requestError) {
      setError(requestError.message || `Backend is not ready yet at ${API_BASE}.`);
      scheduleRetry();
    } finally {
      setLoadingKey("");
    }
  }

  async function handleTrain() {
    try {
      setLoadingKey("training-model");
      setError("");
      const payload = await apiFetch("/train", { method: "POST" });
      setTrainingResult(payload.training);

      const [recommendationPayload, modelPayload] = await Promise.all([
        apiFetch("/recommendations/current"),
        apiFetch("/model/latest")
      ]);
      setRecommendation(recommendationPayload);
      setModelData(modelPayload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function handleManualPredict() {
    try {
      setLoadingKey("manual-prediction");
      setError("");
      const payload = await apiFetch("/predict/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: Number(manualInput.amount),
          time: Number(manualInput.time),
          location: manualInput.location,
          merchant: manualInput.merchant
        })
      });
      setManualResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function handlePredictTestSample() {
    try {
      setLoadingKey("held-out-test-sample");
      setError("");
      const payload = await apiFetch(`/predict/test-sample?index=${testSampleIndex}`);
      setTestSample(payload);
      setTestSampleIndex(payload.next_index || 0);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  const metadata = modelData?.metadata || {};
  const shortlist =
    recommendation?.recommendation_summary?.shortlisted_models ||
    metadata.shortlisted_models ||
    [];
  const fullModelPool =
    recommendation?.recommendation_summary?.all_models ||
    metadata.full_model_pool ||
    [];
  const datasetSignals =
    recommendation?.recommendation_summary?.dataset_characteristics ||
    metadata.dataset_characteristics ||
    {};
  const metricsSource = trainingResult || metadata || {};
  const validationMetrics = metricsSource.validation_metrics || {};
  const testMetrics = metricsSource.test_metrics || {};
  const selectedModelName = metricsSource.selected_model_name || metadata.selected_model_name || "";
  const selectedModel =
    metricsSource.selected_model_display_name ||
    metadata.selected_model_display_name ||
    (selectedModelName ? titleCase(selectedModelName) : "Not trained yet");
  const hasTrainedModel = Boolean(metadata.selected_model_name || trainingResult?.selected_model_name);
  const selectedModelStory = buildSelectedModelStory(metadata, shortlist);
  const locationOptions = datasetPreview?.manual_input_options?.location_options || [manualInput.location];
  const merchantOptions = datasetPreview?.manual_input_options?.merchant_options || [manualInput.merchant];

  return (
    <div className="presentation-shell">
      <div className="ambient-shape ambient-one" />
      <div className="ambient-shape ambient-two" />

      <header className="hero-panel">
        <div className="hero-copy">
          <div className="hero-eyebrow">Evaluation Day Presentation UI</div>
          <h1>AI Fraud Detection System</h1>
          <p>
            A simplified, explainable interface built for presentation: show the dataset, justify the
            model, train it, give manual inputs, and finally test it on unseen data in real time.
          </p>
        </div>

        <div className="section-tab-row" aria-label="Main presentation sections">
          {sectionTabs.map((tab) => (
            <SectionTab
              key={tab.id}
              icon={tab.icon}
              label={tab.label}
              subtitle={tab.subtitle}
              active={activeSection === tab.id}
              onClick={() => setActiveSection(tab.id)}
            />
          ))}
        </div>

        <div className="hero-action-row">
          <button className="ghost-button" onClick={() => void loadAiView()}>
            <RefreshCw size={16} />
            Reload AI Data
          </button>
          <span className="hero-action-note">
            If the backend is still starting, this page now retries automatically.
          </span>
        </div>

        <div className="hero-metric-row">
          <HeroMetric label="Dataset rows" value={formatCount(datasetPreview?.sample_count)} />
          <HeroMetric label="Fraud rate" value={formatPercent(datasetPreview?.fraud_rate)} />
          <HeroMetric label="Selected model" value={selectedModel} />
          <HeroMetric label="Test accuracy" value={formatPercent(testMetrics?.accuracy)} />
        </div>
      </header>

      {error ? (
        <div className="status-banner error">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      ) : null}

      {connectionNote ? (
        <div className="status-banner info">
          <RefreshCw size={16} />
          <span>{connectionNote}</span>
        </div>
      ) : null}

      {loadingKey ? (
        <div className="status-banner info">
          <RefreshCw size={16} />
          <span>{loadingKey}</span>
        </div>
      ) : null}

      {activeSection === "ai" ? (
        <main className="ai-stage">
          <section className="flow-strip">
            {aiSteps.map((step) => {
              const Icon = step.icon;
              return (
                <article key={step.number} className="flow-step">
                  <span>{step.number}</span>
                  <strong>{step.title}</strong>
                  <Icon size={18} />
                </article>
              );
            })}
          </section>

          <StorySection
            step="Step 1"
            title="Understand the Dataset"
            icon={Rows3}
            intro="Start the demo by showing what the model sees: the feature columns, a few sample rows, and the class balance."
          >
            <div className="story-grid two-column">
              <article className="surface-card">
                <div className="card-heading">
                  <Rows3 size={18} />
                  <h3>Dataset Snapshot</h3>
                </div>
                <div className="metric-strip">
                  <MetricCard label="Source" value={formatSource(datasetPreview?.dataset_source)} tone="accent" />
                  <MetricCard label="Rows" value={formatCount(datasetPreview?.sample_count)} />
                  <MetricCard label="Normal class" value={formatPercent(datasetPreview?.normal_rate)} />
                  <MetricCard label="Fraud class" value={formatPercent(datasetPreview?.fraud_rate)} tone="warm" />
                </div>
                <div className="explain-box">
                  <strong>Simple explanation</strong>
                  <p>
                    Before training, we inspect the structure of the data and the fraud ratio so we know
                    whether the problem is balanced or imbalanced.
                  </p>
                </div>
              </article>

              <article className="surface-card">
                <div className="card-heading">
                  <WandSparkles size={18} />
                  <h3>Feature Meaning</h3>
                </div>
                <div className="feature-grid">
                  {(datasetPreview?.feature_cards || []).map((feature) => (
                    <article key={feature.name} className="feature-card">
                      <span className={`feature-kind ${feature.kind}`}>{feature.kind}</span>
                      <strong>{feature.name}</strong>
                      <p>{feature.description}</p>
                    </article>
                  ))}
                </div>
              </article>
            </div>

            <article className="surface-card">
              <div className="card-heading">
                <Rows3 size={18} />
                <h3>Sample Rows</h3>
              </div>
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Amount</th>
                      <th>Time</th>
                      <th>Location</th>
                      <th>Merchant</th>
                      <th>Fraud</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(datasetPreview?.sample_rows || []).map((row, index) => (
                      <tr key={`${row.location}-${row.merchant}-${index}`}>
                        <td>{row.amount}</td>
                        <td>{row.time}</td>
                        <td>{row.location}</td>
                        <td>{row.merchant}</td>
                        <td>{row.fraud}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </StorySection>

          <StorySection
            step="Step 2"
            title="Why This Model Was Chosen"
            icon={Brain}
            intro="The system first recommends a shortlist, then selects the best-performing model after training and validation."
          >
            <div className="story-grid two-column">
              <article className="surface-card">
                <div className="card-heading">
                  <Brain size={18} />
                  <h3>Dataset Signals Used for Recommendation</h3>
                </div>
                <div className="metric-strip">
                  <MetricCard label="Samples" value={formatCount(datasetSignals.sample_count)} />
                  <MetricCard label="Numeric features" value={formatCount(datasetSignals.numeric_feature_count)} />
                  <MetricCard label="Categorical features" value={formatCount(datasetSignals.categorical_feature_count)} />
                  <MetricCard label="Fraud rate" value={formatPercent(datasetSignals.fraud_rate)} tone="warm" />
                  <MetricCard
                    label="Imbalance detected"
                    value={datasetSignals.class_imbalance_detected ? "Yes" : "No"}
                    tone="accent"
                  />
                  <MetricCard label="Encoded features" value={formatCount(datasetSignals.encoded_feature_estimate)} />
                </div>
              </article>

              <article className="surface-card spotlight-card">
                <div className="card-heading">
                  <CheckCircle2 size={18} />
                  <h3>Selected Model</h3>
                </div>
                <div className="spotlight-copy">
                  <span className="spotlight-label">Winner</span>
                  <strong>{selectedModel}</strong>
                  <p>{selectedModelStory}</p>
                </div>
                <div className="mini-metrics">
                  <div>
                    <span>Validation F1</span>
                    <strong>{formatScore(validationMetrics.f1)}</strong>
                  </div>
                  <div>
                    <span>Test F1</span>
                    <strong>{formatScore(testMetrics.f1)}</strong>
                  </div>
                  <div>
                    <span>Overfit check</span>
                    <strong>{statusText(metricsSource.overfit_flag)}</strong>
                  </div>
                </div>
              </article>
            </div>

            <div className="shortlist-grid">
              {shortlist.map((item) => (
                <article key={item.model_name} className="shortlist-card">
                  <div className="shortlist-rank">Rank {item.shortlist_rank || "-"}</div>
                  <strong>{item.display_name || titleCase(item.model_name)}</strong>
                  <p>{item.rationale}</p>
                </article>
              ))}
            </div>

            <details className="details-card">
              <summary>Show full model scoring table</summary>
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Score</th>
                      <th>Status</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fullModelPool.map((item) => (
                      <tr key={item.model_name}>
                        <td>{item.display_name || titleCase(item.model_name)}</td>
                        <td>{formatScore(item.score, 1)}</td>
                        <td>{item.shortlisted ? `Shortlisted (${item.shortlist_rank})` : "Not shortlisted"}</td>
                        <td>{item.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </StorySection>

          <StorySection
            step="Step 3"
            title="Train and Evaluate the Model"
            icon={Activity}
            intro="This step shows the actual performance after training, including the decision threshold, metrics, and confusion matrix."
          >
            <div className="story-grid two-column">
              <article className="surface-card">
                <div className="card-heading">
                  <Activity size={18} />
                  <h3>Training Action</h3>
                </div>
                <p className="support-copy">
                  Use this during the demo if you want to retrain in front of the examiner and immediately
                  update the saved model metrics.
                </p>
                <div className="action-row">
                  <button className="primary-button" onClick={() => void handleTrain()}>
                    <Play size={16} />
                    {hasTrainedModel ? "Retrain Model" : "Train Model"}
                  </button>
                  <button className="secondary-button" onClick={() => void loadAiView()}>
                    <RefreshCw size={16} />
                    Refresh View
                  </button>
                </div>
              </article>

              <article className="surface-card">
                <div className="card-heading">
                  <Gauge size={18} />
                  <h3>Evaluation Metrics</h3>
                </div>
                <div className="metric-strip">
                  <MetricCard label="Validation F1" value={formatScore(validationMetrics.f1)} />
                  <MetricCard label="Test Accuracy" value={formatPercent(testMetrics.accuracy)} tone="accent" />
                  <MetricCard label="Test F1" value={formatScore(testMetrics.f1)} />
                  <MetricCard label="PR AUC" value={formatScore(testMetrics.average_precision)} />
                  <MetricCard label="ROC AUC" value={formatScore(testMetrics.roc_auc)} />
                  <MetricCard label="Threshold" value={formatScore(metricsSource.selected_threshold)} tone="warm" />
                </div>
                <div className="explain-box">
                  <strong>Simple explanation</strong>
                  <p>
                    We judge the model using test accuracy, F1 score, PR AUC, ROC AUC, and the confusion
                    matrix because fraud data is usually imbalanced.
                  </p>
                </div>
              </article>
            </div>

            <ConfusionMatrix matrix={testMetrics.confusion_matrix} />
          </StorySection>

          <StorySection
            step="Step 4"
            title="Give Manual Input and See the Prediction"
            icon={Play}
            intro="After the model is trained, you can enter a transaction manually and explain the prediction in simple terms."
          >
            <div className="story-grid two-column">
              <article className="surface-card">
                <div className="card-heading">
                  <Play size={18} />
                  <h3>Prediction Input</h3>
                </div>
                <form
                  className="prediction-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    void handleManualPredict();
                  }}
                >
                  <label className="field">
                    <span>Transaction amount</span>
                    <input
                      type="number"
                      step="0.01"
                      value={manualInput.amount}
                      onChange={(event) => setManualInput((current) => ({ ...current, amount: event.target.value }))}
                    />
                  </label>

                  <label className="field">
                    <span>Transaction hour (0 to 23)</span>
                    <input
                      type="number"
                      min="0"
                      max="23"
                      value={manualInput.time}
                      onChange={(event) => setManualInput((current) => ({ ...current, time: event.target.value }))}
                    />
                  </label>

                  <label className="field">
                    <span>Location</span>
                    <select
                      value={manualInput.location}
                      onChange={(event) => setManualInput((current) => ({ ...current, location: event.target.value }))}
                    >
                      {locationOptions.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="field">
                    <span>Merchant type</span>
                    <select
                      value={manualInput.merchant}
                      onChange={(event) => setManualInput((current) => ({ ...current, merchant: event.target.value }))}
                    >
                      {merchantOptions.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </label>

                  <button className="primary-button" type="submit" disabled={!hasTrainedModel}>
                    <Play size={16} />
                    Predict Transaction
                  </button>

                  {!hasTrainedModel ? (
                    <p className="helper-copy">Train the model first to enable live prediction.</p>
                  ) : null}
                </form>
              </article>

              <article className="surface-card result-card">
                <div className="card-heading">
                  <WandSparkles size={18} />
                  <h3>Prediction Result</h3>
                </div>
                {manualResult ? (
                  <>
                    <div className="result-banner">
                      <span className={`result-pill ${manualResult.prediction === 1 ? "danger" : "safe"}`}>
                        {manualResult.prediction_label}
                      </span>
                      <strong>{formatPercent(manualResult.probability)}</strong>
                    </div>
                    <p className="support-copy">{manualResult.message}</p>
                    <div className="mini-metrics">
                      <div>
                        <span>Model</span>
                        <strong>{manualResult.model_name}</strong>
                      </div>
                      <div>
                        <span>Threshold</span>
                        <strong>{formatScore(manualResult.threshold)}</strong>
                      </div>
                      <div>
                        <span>Confidence</span>
                        <strong>{manualResult.confidence_band}</strong>
                      </div>
                    </div>
                    <div className="signal-list">
                      {manualResult.risk_signals.map((signal) => (
                        <span key={signal} className="signal-pill">
                          {signal}
                        </span>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="empty-copy">
                    Enter the transaction values and click predict to show the decision live during the demo.
                  </p>
                )}
              </article>
            </div>
          </StorySection>

          <StorySection
            step="Step 5"
            title="Show Prediction on Unseen Test Data"
            icon={Gauge}
            intro="Finish by testing the saved model on held-out samples so you can show real-time inference on data the model did not see during training."
          >
            <div className="story-grid two-column">
              <article className="surface-card">
                <div className="card-heading">
                  <Gauge size={18} />
                  <h3>Held-Out Test Demo</h3>
                </div>
                <p className="support-copy">
                  This button picks the next row from the 20% test split, predicts it live, and compares the
                  result with the actual class label.
                </p>
                <div className="action-row">
                  <button className="primary-button" onClick={() => void handlePredictTestSample()} disabled={!hasTrainedModel}>
                    <Play size={16} />
                    Predict Next Test Sample
                  </button>
                </div>
                {!hasTrainedModel ? (
                  <p className="helper-copy">Train the model first to enable test-sample prediction.</p>
                ) : null}
              </article>

              <article className="surface-card result-card">
                <div className="card-heading">
                  <CheckCircle2 size={18} />
                  <h3>Test Sample Result</h3>
                </div>
                {testSample ? (
                  <>
                    <div className="result-banner">
                      <span className={`result-pill ${testSample.correct ? "safe" : "warning"}`}>
                        {testSample.correct ? "Correct" : "Mismatch"}
                      </span>
                      <strong>
                        Sample {testSample.sample_index + 1} / {testSample.total_test_samples}
                      </strong>
                    </div>
                    <p className="support-copy">{testSample.review_text}</p>
                    <div className="mini-metrics">
                      <div>
                        <span>Predicted</span>
                        <strong>{testSample.prediction_label}</strong>
                      </div>
                      <div>
                        <span>Actual</span>
                        <strong>{testSample.actual_label_text}</strong>
                      </div>
                      <div>
                        <span>Probability</span>
                        <strong>{formatPercent(testSample.probability)}</strong>
                      </div>
                    </div>
                    <div className="sample-kv-grid">
                      <div>
                        <span>Amount</span>
                        <strong>{testSample.input.amount}</strong>
                      </div>
                      <div>
                        <span>Time</span>
                        <strong>{testSample.input.time}</strong>
                      </div>
                      <div>
                        <span>Location</span>
                        <strong>{testSample.input.location}</strong>
                      </div>
                      <div>
                        <span>Merchant</span>
                        <strong>{testSample.input.merchant}</strong>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="empty-copy">
                    Click the button to predict one unseen test row and compare the output with the real label.
                  </p>
                )}
              </article>
            </div>
          </StorySection>
        </main>
      ) : null}

      {activeSection === "db" ? (
        <PlaceholderSection
          icon={Database}
          title="DB Section"
          description="The UI is already simplified into a 3-section presentation layout. This DB area is intentionally clean and ready for the database-specific flow you want to add next."
          cards={[
            {
              title: "Suggested story",
              text: "Show the tables, explain why they are separated, then connect them to predictions and alerts."
            },
            {
              title: "Ready space",
              text: "This section can be turned into schema overview, ER diagram, and normalization explanation in the next prompt."
            },
            {
              title: "Current status",
              text: "Kept intentionally minimal so your evaluation flow stays focused on AI first."
            }
          ]}
        />
      ) : null}

      {activeSection === "sda" ? (
        <PlaceholderSection
          icon={GraduationCap}
          title="SDA Section"
          description="This section is reserved for the SDA flow you will provide next, so the layout stays consistent and presentable."
          cards={[
            {
              title: "Suggested story",
              text: "Explain the system modules, the backend workflow, and how the frontend talks to the API."
            },
            {
              title: "Ready space",
              text: "This area can become an architecture walkthrough, API flow, and viva support section."
            },
            {
              title: "Current status",
              text: "Left clean on purpose so we can design it around your exact SDA prompt."
            }
          ]}
        />
      ) : null}
    </div>
  );
}

export default App;
