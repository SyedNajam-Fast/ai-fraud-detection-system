import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Copy,
  Database,
  Download,
  FileUp,
  Gauge,
  GraduationCap,
  LayoutDashboard,
  Play,
  RefreshCw,
  Rows3,
  WandSparkles
} from "lucide-react";
import MermaidDiagram from "./components/MermaidDiagram";

const API_BASE = "http://127.0.0.1:8000/api";

const tabs = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "data", label: "Data Understanding", icon: Rows3 },
  { id: "schema", label: "Database Design", icon: Database },
  { id: "ai", label: "Model Recommendation", icon: Brain },
  { id: "workflow", label: "Training & Workflow", icon: Activity },
  { id: "presentation", label: "Diagrams & Viva", icon: GraduationCap }
];

const quickStats = [
  ["users", "Users"],
  ["transactions", "Transactions"],
  ["predictions", "Predictions"],
  ["fraud_alerts", "Fraud Alerts"],
  ["raw_dataset_uploads", "Profiled Datasets"],
  ["model_training_runs", "Training Runs"]
];

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function modeText(mode, simple, technical = simple) {
  return mode === "simple" ? simple : technical;
}

function formatReadiness(status) {
  if (status === "ready") {
    return "Ready";
  }
  if (status === "attention") {
    return "Needs Attention";
  }
  return formatValue(status);
}

function downloadTextFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [mode, setMode] = useState("simple");
  const [dashboard, setDashboard] = useState(null);
  const [datasetOptions, setDatasetOptions] = useState([]);
  const [latestProfile, setLatestProfile] = useState(null);
  const [schemaData, setSchemaData] = useState(null);
  const [presentationData, setPresentationData] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [workflowResult, setWorkflowResult] = useState(null);
  const [selectedDiagramId, setSelectedDiagramId] = useState("");
  const [presentationExportStatus, setPresentationExportStatus] = useState("");
  const [error, setError] = useState("");
  const [loadingKey, setLoadingKey] = useState("");
  const [selectedPath, setSelectedPath] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);

  useEffect(() => {
    void loadInitial();
  }, []);

  async function apiFetch(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }
    return payload;
  }

  async function loadInitial() {
    try {
      setLoadingKey("initial");
      const [dashboardPayload, datasetsPayload, modelPayload] = await Promise.all([
        apiFetch("/dashboard"),
        apiFetch("/datasets/options"),
        apiFetch("/model/latest")
      ]);
      setDashboard(dashboardPayload);
      setDatasetOptions(datasetsPayload.datasets || []);
      setLatestProfile(dashboardPayload.latest_profile || null);
      setModelData(modelPayload);

      const defaultDataset = (datasetsPayload.datasets || []).find((item) => item.available);
      if (defaultDataset) {
        setSelectedPath(defaultDataset.path);
        setTargetColumn(defaultDataset.defaultTargetColumn || "");
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function loadSchema() {
    try {
      setLoadingKey("schema");
      const payload = await apiFetch("/schema");
      setSchemaData(payload.schema);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function loadPresentation() {
    try {
      setLoadingKey("presentation");
      const payload = await apiFetch("/presentation");
      setPresentationData(payload.presentation);
      const firstDiagram = payload.presentation?.diagrams?.[0];
      if (firstDiagram) {
        setSelectedDiagramId((currentValue) => currentValue || firstDiagram.id);
      }
      return payload.presentation;
    } catch (requestError) {
      setError(requestError.message);
      return null;
    } finally {
      setLoadingKey("");
    }
  }

  async function loadRecommendation() {
    try {
      setLoadingKey("recommendation");
      const payload = await apiFetch("/recommendations/current");
      setRecommendation(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function refreshModel() {
    const modelPayload = await apiFetch("/model/latest");
    setModelData(modelPayload);
  }

  async function refreshPresentation() {
    const payload = await apiFetch("/presentation");
    setPresentationData(payload.presentation);
    const firstDiagram = payload.presentation?.diagrams?.[0];
    if (firstDiagram) {
      setSelectedDiagramId((currentValue) => currentValue || firstDiagram.id);
    }
    return payload.presentation;
  }

  async function handleCopyMarkdownReport() {
    try {
      const freshPresentation =
        presentationData?.markdown_report ? presentationData : await loadPresentation();
      const reportText = freshPresentation?.markdown_report || "";
      if (!reportText) {
        throw new Error("Presentation report is not available yet.");
      }
      await navigator.clipboard.writeText(reportText);
      setPresentationExportStatus("Markdown report copied.");
    } catch (copyError) {
      setError(copyError.message);
    }
  }

  async function handleDownloadPresentationExport(exportFormat) {
    try {
      setLoadingKey(`presentation-${exportFormat}`);
      setError("");
      const payload = await apiFetch(`/presentation/export?format=${exportFormat}`);
      const exported = payload.export;
      downloadTextFile(exported.filename, exported.content, exported.content_type);
      setPresentationExportStatus(`${exported.filename} downloaded.`);
    } catch (downloadError) {
      setError(downloadError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function handleCopyMermaidSource(chart) {
    try {
      await navigator.clipboard.writeText(chart);
      setPresentationExportStatus("Mermaid source copied.");
    } catch (copyError) {
      setError(copyError.message);
    }
  }

  async function handleProfilePath() {
    try {
      setLoadingKey("profile-path");
      setError("");
      const payload = await apiFetch("/profile/path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          csv_path: selectedPath,
          target_column: targetColumn || null
        })
      });
      setLatestProfile(payload.profile);
      setDashboard(payload.dashboard);
      await refreshPresentation();
      setActiveTab("data");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function handleUploadProfile() {
    if (!selectedFile) {
      setError("Choose a CSV file before uploading.");
      return;
    }

    try {
      setLoadingKey("profile-upload");
      setError("");
      const formData = new FormData();
      formData.append("file", selectedFile);
      if (targetColumn) {
        formData.append("target_column", targetColumn);
      }
      const response = await fetch(`${API_BASE}/profile/upload`, {
        method: "POST",
        body: formData
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || "Upload failed.");
      }
      setLatestProfile(payload.profile);
      setDashboard(payload.dashboard);
      await refreshPresentation();
      setActiveTab("data");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function handleTrain() {
    try {
      setLoadingKey("train");
      setError("");
      const payload = await apiFetch("/train", { method: "POST" });
      setTrainingResult(payload.training);
      setDashboard(payload.dashboard);
      await refreshModel();
      await refreshPresentation();
      setActiveTab("workflow");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  async function handleWorkflow(forceTrain = false) {
    try {
      setLoadingKey(forceTrain ? "workflow-force" : "workflow");
      setError("");
      const payload = await apiFetch("/workflow/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force_train: forceTrain })
      });
      setWorkflowResult(payload.workflow);
      setDashboard(payload.dashboard);
      await refreshModel();
      await refreshPresentation();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingKey("");
    }
  }

  const counts = dashboard?.counts || {};
  const latestMetadata = modelData?.metadata || {};
  const latestRecommendations = modelData?.recommendations || [];
  const diagrams = presentationData?.diagrams || [];
  const activeDiagram =
    diagrams.find((diagram) => diagram.id === selectedDiagramId) || diagrams[0] || null;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="eyebrow">Semester Project Control Room</div>
          <h1>Fraud Detection, Data Understanding, Database Design, and Viva Dashboard</h1>
          <p>
            One local interface for profiling datasets, explaining the schema, recommending models, training the
            shortlist, running the fraud workflow, and presenting diagrams plus viva notes.
          </p>
        </div>
        <div className="topbar-actions">
          <div className="mode-toggle" role="tablist" aria-label="Explanation mode">
            <button className={mode === "simple" ? "active" : ""} onClick={() => setMode("simple")}>
              Simple Mode
            </button>
            <button className={mode === "technical" ? "active" : ""} onClick={() => setMode("technical")}>
              Technical Mode
            </button>
          </div>
          <button className="refresh-button" onClick={() => void loadInitial()}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>
      </header>

      {error ? <div className="status-banner error">{error}</div> : null}
      {loadingKey ? <div className="status-banner info">Running: {loadingKey}</div> : null}
      {presentationExportStatus ? <div className="status-banner success">{presentationExportStatus}</div> : null}

      <section className="stats-grid">
        {quickStats.map(([key, label]) => (
          <article key={key} className="stat-card">
            <span>{label}</span>
            <strong>{counts[key] ?? 0}</strong>
          </article>
        ))}
      </section>

      <nav className="tab-strip" aria-label="Main sections">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              className={activeTab === tab.id ? "active" : ""}
              onClick={() => {
                setActiveTab(tab.id);
                if (tab.id === "schema" && !schemaData) {
                  void loadSchema();
                }
                if (tab.id === "ai" && !recommendation) {
                  void loadRecommendation();
                }
                if (tab.id === "presentation" && !presentationData) {
                  void loadPresentation();
                }
              }}
            >
              <Icon size={16} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>

      <main className="content-band">
        {activeTab === "dashboard" ? (
          <section className="panel-grid two-column">
            <article className="panel">
              <div className="panel-header">
                <Gauge size={18} />
                <h2>Latest Model State</h2>
              </div>
              <p className="panel-copy">
                {modeText(
                  mode,
                  "This area shows the currently saved model and the shortlist used to choose it.",
                  "This area mirrors the latest saved model metadata and shortlist persistence."
                )}
              </p>
              <dl className="detail-list">
                <div><dt>Selected model</dt><dd>{latestMetadata.selected_model_name || "N/A"}</dd></div>
                <div><dt>Threshold</dt><dd>{formatValue(latestMetadata.selected_threshold)}</dd></div>
                <div><dt>Dataset source</dt><dd>{latestMetadata.dataset_source || "N/A"}</dd></div>
                <div><dt>Selection metric</dt><dd>{latestMetadata.selection_metric || "N/A"}</dd></div>
              </dl>
              <div className="pill-row">
                {(latestMetadata.shortlisted_models || []).map((item) => (
                  <span key={item.model_name} className="pill accent">{item.model_name}</span>
                ))}
              </div>
            </article>

            <article className="panel">
              <div className="panel-header">
                <FileUp size={18} />
                <h2>Latest Profile Snapshot</h2>
              </div>
              <p className="panel-copy">
                {modeText(
                  mode,
                  "This summarizes the last dataset the system inspected.",
                  "This summarizes the latest raw dataset upload and profile rows."
                )}
              </p>
              {latestProfile ? (
                <dl className="detail-list">
                  <div><dt>Filename</dt><dd>{latestProfile.upload?.filename}</dd></div>
                  <div><dt>Rows</dt><dd>{formatValue(latestProfile.dataset_profile?.row_count)}</dd></div>
                  <div><dt>Columns</dt><dd>{formatValue(latestProfile.dataset_profile?.column_count)}</dd></div>
                  <div><dt>Target column</dt><dd>{latestProfile.dataset_profile?.target_column || "N/A"}</dd></div>
                </dl>
              ) : (
                <p className="empty-state">No dataset has been profiled yet.</p>
              )}
            </article>

            <article className="panel wide">
              <div className="panel-header">
                <WandSparkles size={18} />
                <h2>Quick Actions</h2>
              </div>
              <div className="action-row">
                <button className="primary-button" onClick={() => void handleProfilePath()}>
                  <FileUp size={16} />
                  Profile Selected Dataset
                </button>
                <button className="primary-button muted" onClick={() => void loadRecommendation()}>
                  <Brain size={16} />
                  Recommend Models
                </button>
                <button className="primary-button muted" onClick={() => void handleTrain()}>
                  <Brain size={16} />
                  Train Shortlist
                </button>
                <button className="primary-button muted" onClick={() => void handleWorkflow(false)}>
                  <Play size={16} />
                  Run Workflow
                </button>
              </div>
            </article>
          </section>
        ) : null}

        {activeTab === "data" ? (
          <section className="panel-grid two-column">
            <article className="panel">
              <div className="panel-header">
                <FileUp size={18} />
                <h2>Profile a Dataset</h2>
              </div>
              <label className="field">
                <span>Local dataset path</span>
                <select value={selectedPath} onChange={(event) => setSelectedPath(event.target.value)}>
                  <option value="">Choose a known dataset path</option>
                  {datasetOptions.map((item) => (
                    <option key={item.path} value={item.path} disabled={!item.available}>
                      {item.label} {item.available ? "" : "(missing locally)"}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Target column override</span>
                <input value={targetColumn} onChange={(event) => setTargetColumn(event.target.value)} placeholder="Class" />
              </label>
              <div className="action-row">
                <button className="primary-button" onClick={() => void handleProfilePath()}>
                  <Rows3 size={16} />
                  Profile from Path
                </button>
              </div>
              <label className="field">
                <span>Upload CSV file</span>
                <input type="file" accept=".csv" onChange={(event) => setSelectedFile(event.target.files?.[0] || null)} />
              </label>
              <div className="action-row">
                <button className="primary-button muted" onClick={() => void handleUploadProfile()}>
                  <FileUp size={16} />
                  Upload and Profile
                </button>
              </div>
            </article>

            <article className="panel">
              <div className="panel-header">
                <Rows3 size={18} />
                <h2>Latest Profile Summary</h2>
              </div>
              {latestProfile ? (
                <>
                  <dl className="detail-list">
                    <div><dt>File</dt><dd>{latestProfile.upload?.filename}</dd></div>
                    <div><dt>Rows</dt><dd>{latestProfile.dataset_profile?.row_count}</dd></div>
                    <div><dt>Columns</dt><dd>{latestProfile.dataset_profile?.column_count}</dd></div>
                    <div><dt>Duplicates</dt><dd>{latestProfile.dataset_profile?.duplicate_row_count}</dd></div>
                    <div><dt>Missing cells</dt><dd>{latestProfile.dataset_profile?.missing_cell_count}</dd></div>
                    <div><dt>Target column</dt><dd>{latestProfile.dataset_profile?.target_column || "N/A"}</dd></div>
                    <div><dt>Imbalance ratio</dt><dd>{formatPercent(latestProfile.dataset_profile?.class_imbalance_ratio)}</dd></div>
                  </dl>
                  <div className="warning-list">
                    {JSON.parse(latestProfile.dataset_profile?.warnings_json || "[]").map((warning) => (
                      <span key={warning} className="warning-pill">{warning}</span>
                    ))}
                  </div>
                </>
              ) : (
                <p className="empty-state">Profile a dataset to populate this summary.</p>
              )}
            </article>

            <article className="panel wide">
              <div className="panel-header">
                <Rows3 size={18} />
                <h2>Column Explanations</h2>
              </div>
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Role</th>
                      <th>Type</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(latestProfile?.feature_profiles || []).map((feature) => (
                      <tr key={feature.id}>
                        <td>{feature.column_name}</td>
                        <td>{feature.inferred_role}</td>
                        <td>{feature.inferred_dtype}</td>
                        <td>{mode === "simple" ? feature.simple_description : feature.technical_description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </section>
        ) : null}

        {activeTab === "schema" ? (
          <section className="panel-grid two-column">
            <article className="panel">
              <div className="panel-header">
                <Database size={18} />
                <h2>Schema Overview</h2>
              </div>
              {schemaData ? (
                <>
                  <p className="panel-copy">{mode === "simple" ? schemaData.simple_overview : schemaData.technical_overview}</p>
                  <div className="bullet-list">
                    {schemaData.layer_summaries.map((item) => <div key={item}>{item}</div>)}
                  </div>
                  <div className="bullet-list">
                    {schemaData.normalization_summary.map((item) => <div key={item}>{item}</div>)}
                  </div>
                </>
              ) : (
                <button className="primary-button" onClick={() => void loadSchema()}>
                  <Database size={16} />
                  Load Schema Explanation
                </button>
              )}
            </article>

            <article className="panel">
              <div className="panel-header">
                <Database size={18} />
                <h2>ER Diagram</h2>
              </div>
              {schemaData ? (
                <MermaidDiagram chart={schemaData.mermaid_er_diagram} title="ER Diagram" />
              ) : (
                <p className="empty-state">Schema data not loaded yet.</p>
              )}
            </article>

            <article className="panel wide">
              <div className="panel-header">
                <Database size={18} />
                <h2>Table-by-Table Explanation</h2>
              </div>
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Table</th>
                      <th>Layer</th>
                      <th>Primary Key</th>
                      <th>Purpose</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(schemaData?.tables || []).map((table) => (
                      <tr key={table.table_name}>
                        <td>{table.table_name}</td>
                        <td>{table.layer}</td>
                        <td>{table.primary_key_columns.join(", ")}</td>
                        <td>{mode === "simple" ? table.purpose : table.normalization_note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </section>
        ) : null}

        {activeTab === "ai" ? (
          <section className="panel-grid two-column">
            <article className="panel">
              <div className="panel-header">
                <Brain size={18} />
                <h2>Current Recommendation</h2>
              </div>
              {recommendation ? (
                <>
                  <dl className="detail-list">
                    {Object.entries(recommendation.recommendation_summary.dataset_characteristics || {}).map(([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>{typeof value === "number" && key.includes("ratio") ? formatPercent(value) : formatValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                  <div className="model-list">
                    {recommendation.recommendation_summary.shortlisted_models.map((item) => (
                      <article key={item.model_name} className="model-card">
                        <header>
                          <strong>{item.model_name}</strong>
                          <span>Rank {item.shortlist_rank}</span>
                        </header>
                        <p>{item.rationale}</p>
                      </article>
                    ))}
                  </div>
                </>
              ) : (
                <button className="primary-button" onClick={() => void loadRecommendation()}>
                  <Brain size={16} />
                  Load Recommendation
                </button>
              )}
            </article>

            <article className="panel">
              <div className="panel-header">
                <Gauge size={18} />
                <h2>Latest Trained Model</h2>
              </div>
              <dl className="detail-list">
                <div><dt>Selected model</dt><dd>{latestMetadata.selected_model_name || "N/A"}</dd></div>
                <div><dt>Validation F1</dt><dd>{formatValue(latestMetadata.validation_metrics?.f1)}</dd></div>
                <div><dt>Validation PR AUC</dt><dd>{formatValue(latestMetadata.validation_metrics?.average_precision)}</dd></div>
                <div><dt>Validation ROC AUC</dt><dd>{formatValue(latestMetadata.validation_metrics?.roc_auc)}</dd></div>
                <div><dt>Test F1</dt><dd>{formatValue(latestMetadata.test_metrics?.f1)}</dd></div>
              </dl>
              <div className="model-list">
                {latestRecommendations.map((item) => (
                  <article key={item.id} className={`model-card ${item.final_winner ? "winner" : ""}`}>
                    <header>
                      <strong>{item.model_name}</strong>
                      <span>Rank {item.recommendation_rank}</span>
                    </header>
                    <p>{mode === "simple" ? item.rationale_text : `Score ${item.recommendation_score}: ${item.rationale_text}`}</p>
                  </article>
                ))}
              </div>
            </article>

            <article className="panel wide">
              <div className="panel-header">
                <Brain size={18} />
                <h2>Full Model Pool</h2>
              </div>
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
                    {(recommendation?.recommendation_summary.all_models || []).map((item) => (
                      <tr key={item.model_name}>
                        <td>{item.model_name}</td>
                        <td>{formatValue(item.score)}</td>
                        <td>{item.shortlisted ? `Shortlisted (${item.shortlist_rank})` : "Not shortlisted"}</td>
                        <td>{item.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </section>
        ) : null}

        {activeTab === "workflow" ? (
          <section className="panel-grid two-column">
            <article className="panel">
              <div className="panel-header">
                <Play size={18} />
                <h2>Execute Training and Workflow</h2>
              </div>
              <p className="panel-copy">
                {modeText(
                  mode,
                  "Use these actions during your demo to show recommendation, training, and live fraud prediction from the same interface.",
                  "These actions invoke the backend training and orchestration services through FastAPI."
                )}
              </p>
              <div className="action-row">
                <button className="primary-button" onClick={() => void handleTrain()}>
                  <Brain size={16} />
                  Train Shortlist
                </button>
                <button className="primary-button muted" onClick={() => void handleWorkflow(false)}>
                  <Play size={16} />
                  Run Workflow
                </button>
                <button className="primary-button muted" onClick={() => void handleWorkflow(true)}>
                  <RefreshCw size={16} />
                  Retrain and Run
                </button>
              </div>
            </article>

            <article className="panel">
              <div className="panel-header">
                <Activity size={18} />
                <h2>Latest Workflow Result</h2>
              </div>
              {workflowResult ? (
                <dl className="detail-list">
                  <div><dt>Transaction ID</dt><dd>{workflowResult.transaction_id}</dd></div>
                  <div><dt>Prediction</dt><dd>{workflowResult.prediction}</dd></div>
                  <div><dt>Probability</dt><dd>{formatValue(workflowResult.probability)}</dd></div>
                  <div><dt>Alert ID</dt><dd>{workflowResult.alert_id ?? "No alert"}</dd></div>
                </dl>
              ) : (
                <p className="empty-state">Run the workflow to see the live transaction result here.</p>
              )}
            </article>

            <article className="panel wide">
              <div className="panel-header">
                <Brain size={18} />
                <h2>Latest Training Result</h2>
              </div>
              {trainingResult ? (
                <div className="table-shell">
                  <table>
                    <thead>
                      <tr><th>Item</th><th>Value</th></tr>
                    </thead>
                    <tbody>
                      <tr><td>Selected model</td><td>{trainingResult.selected_model_name}</td></tr>
                      <tr><td>Threshold</td><td>{formatValue(trainingResult.selected_threshold)}</td></tr>
                      <tr><td>Validation F1</td><td>{formatValue(trainingResult.validation_metrics?.f1)}</td></tr>
                      <tr><td>Validation PR AUC</td><td>{formatValue(trainingResult.validation_metrics?.average_precision)}</td></tr>
                      <tr><td>Validation ROC AUC</td><td>{formatValue(trainingResult.validation_metrics?.roc_auc)}</td></tr>
                      <tr><td>Test F1</td><td>{formatValue(trainingResult.test_metrics?.f1)}</td></tr>
                      <tr><td>Overfit check</td><td>{trainingResult.overfit_flag ? "Flagged" : "Passed"}</td></tr>
                      <tr><td>Underfit check</td><td>{trainingResult.underfit_flag ? "Flagged" : "Passed"}</td></tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="empty-state">Train the shortlist to inspect the latest model metrics here.</p>
              )}
            </article>
          </section>
        ) : null}

        {activeTab === "presentation" ? (
          <section className="panel-grid two-column">
            <article className="panel">
              <div className="panel-header">
                <GraduationCap size={18} />
                <h2>Demo Readiness</h2>
              </div>
              {presentationData ? (
                <>
                  <div className={`readiness-overview ${presentationData.demo_readiness.overall_status}`}>
                    {presentationData.demo_readiness.overall_status === "ready" ? (
                      <CheckCircle2 size={18} />
                    ) : (
                      <AlertTriangle size={18} />
                    )}
                    <div>
                      <strong>{formatReadiness(presentationData.demo_readiness.overall_status)}</strong>
                      <p>{presentationData.demo_readiness.summary}</p>
                    </div>
                  </div>
                  <div className="readiness-list">
                    {presentationData.demo_readiness.checks.map((item) => (
                      <article key={item.id} className={`readiness-card ${item.status}`}>
                        <header>
                          <strong>{item.label}</strong>
                          <span>{formatReadiness(item.status)}</span>
                        </header>
                        <p>{item.detail}</p>
                      </article>
                    ))}
                  </div>
                  <div className="bullet-list">
                    {presentationData.presentation_tips.map((item) => <div key={item}>{item}</div>)}
                  </div>
                </>
              ) : (
                <button className="primary-button" onClick={() => void loadPresentation()}>
                  <GraduationCap size={16} />
                  Load Presentation View
                </button>
              )}
            </article>

            <article className="panel">
              <div className="panel-header">
                <GraduationCap size={18} />
                <h2>Report and Export</h2>
              </div>
              {presentationData ? (
                <>
                  <div className="action-row">
                    <button className="primary-button" onClick={() => void handleCopyMarkdownReport()}>
                      <Copy size={16} />
                      Copy Markdown
                    </button>
                    <button
                      className="primary-button muted"
                      onClick={() => void handleDownloadPresentationExport("markdown")}
                    >
                      <Download size={16} />
                      Download Markdown
                    </button>
                    <button
                      className="primary-button muted"
                      onClick={() => void handleDownloadPresentationExport("json")}
                    >
                      <Download size={16} />
                      Download JSON
                    </button>
                  </div>
                  <div className="summary-stack">
                    {presentationData.report_sections.map((section) => (
                      <article key={section.title} className="summary-card">
                        <strong>{section.title}</strong>
                        <p>{mode === "simple" ? section.simple_text : section.technical_text}</p>
                      </article>
                    ))}
                  </div>
                </>
              ) : (
                <p className="empty-state">Load the presentation view to generate the report pack.</p>
              )}
            </article>

            <article className="panel wide">
              <div className="panel-header">
                <GraduationCap size={18} />
                <h2>Diagram Explorer</h2>
              </div>
              {presentationData ? (
                <div className="diagram-workbench">
                  <aside className="diagram-list">
                    {diagrams.map((diagram) => (
                      <button
                        key={diagram.id}
                        className={diagram.id === activeDiagram?.id ? "active" : ""}
                        onClick={() => setSelectedDiagramId(diagram.id)}
                      >
                        <strong>{diagram.title}</strong>
                        <span>{diagram.course_focus}</span>
                      </button>
                    ))}
                  </aside>
                  <div className="diagram-stage">
                    {activeDiagram ? (
                      <>
                        <article className="diagram-card active-diagram-card">
                          <header>
                            <div>
                              <strong>{activeDiagram.title}</strong>
                              <span>{activeDiagram.description}</span>
                            </div>
                            <span className="course-badge">{activeDiagram.course_focus}</span>
                          </header>
                          <div className="bullet-list compact-list">
                            {activeDiagram.talking_points.map((item) => <div key={item}>{item}</div>)}
                          </div>
                          <div className="action-row">
                            <button
                              className="primary-button"
                              onClick={() => void handleCopyMermaidSource(activeDiagram.mermaid)}
                            >
                              <Copy size={16} />
                              Copy Mermaid
                            </button>
                            <button
                              className="primary-button muted"
                              onClick={() =>
                                downloadTextFile(
                                  `${activeDiagram.id}.mmd`,
                                  activeDiagram.mermaid,
                                  "text/plain;charset=utf-8"
                                )
                              }
                            >
                              <Download size={16} />
                              Download Source
                            </button>
                          </div>
                        </article>
                        <MermaidDiagram chart={activeDiagram.mermaid} title={activeDiagram.title} />
                      </>
                    ) : (
                      <p className="empty-state">No diagrams are available yet.</p>
                    )}
                  </div>
                </div>
              ) : (
                <p className="empty-state">Load the presentation view to inspect Mermaid diagrams.</p>
              )}
            </article>

            <article className="panel wide">
              <div className="panel-header">
                <GraduationCap size={18} />
                <h2>Viva Notes</h2>
              </div>
              <div className="summary-stack">
                {(presentationData?.viva_notes || []).map((note) => (
                  <article key={note.question} className="summary-card">
                    <strong>{note.question}</strong>
                    <p>{note.answer}</p>
                  </article>
                ))}
              </div>
            </article>
          </section>
        ) : null}
      </main>
    </div>
  );
}

export default App;
