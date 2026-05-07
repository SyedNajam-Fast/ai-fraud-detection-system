import { useEffect, useState } from "react";
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
import MermaidDiagram from "./components/MermaidDiagram";
import {
  DEMO_AI_DATASET_PREVIEW,
  DEMO_MODEL_DATA,
  DEMO_PRESENTATION_DATA,
  DEMO_RECOMMENDATION,
  DEMO_SDA_CONTENT,
  DEMO_SCHEMA_DATA,
  buildOfflineManualPrediction,
  buildOfflineTestSample
} from "./demoData";

const API_BASE = "http://127.0.0.1:8000/api";
const BACKEND_START_COMMAND = ".\\start_backend.bat";

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
    subtitle: "Source data, table breakdown, normalization, and ER diagram",
    icon: Database
  },
  {
    id: "sda",
    label: "SDA",
    subtitle: "Architecture and software design flow for the final tab",
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

const dbSteps = [
  { number: "01", title: "Show source data", icon: Rows3 },
  { number: "02", title: "Break it into tables", icon: Database },
  { number: "03", title: "Explain normalization", icon: CheckCircle2 },
  { number: "04", title: "Present the ER diagram", icon: Activity }
];

const sdaSteps = [
  { number: "01", title: "Introduction", icon: GraduationCap },
  { number: "02", title: "Workflow", icon: Activity },
  { number: "03", title: "Architecture", icon: Database },
  { number: "04", title: "Testing", icon: CheckCircle2 }
];

const DB_LAYER_ORDER = ["raw_training", "raw_profiling", "operational", "analytics"];
const DB_CORE_TABLE_NAMES = [
  "kaggle_transactions",
  "raw_dataset_uploads",
  "dataset_profiles",
  "users",
  "transactions",
  "predictions",
  "fraud_alerts"
];
const DB_LAYER_COPY = {
  raw_training: {
    label: "Raw training layer",
    description: "This keeps the original imported fraud rows in a wide form before they are mapped into the smaller project schema."
  },
  raw_profiling: {
    label: "Raw profiling layer",
    description: "This records which dataset file was uploaded or selected so the source can be tracked separately."
  },
  operational: {
    label: "Operational layer",
    description: "These are the live workflow tables used during transaction processing, prediction storage, and alert generation."
  },
  analytics: {
    label: "Analytics and audit layer",
    description: "These tables store profiling summaries plus training history so the system stays explainable."
  }
};

const defaultManualInput = {
  amount: 12450.75,
  time: 23,
  location: "Lahore",
  merchant: "electronics_store"
};

function offlinePresentationMessage() {
  return `Offline presentation mode is active. The UI is using local demo data because the backend is unavailable. Start ${BACKEND_START_COMMAND} for live API data.`;
}

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

function formatBytes(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }

  const numericValue = Number(value);
  if (numericValue < 1024) {
    return `${numericValue} B`;
  }
  if (numericValue < 1024 * 1024) {
    return `${(numericValue / 1024).toFixed(1)} KB`;
  }
  return `${(numericValue / (1024 * 1024)).toFixed(1)} MB`;
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

function safeParseJson(value, fallback = []) {
  if (!value) {
    return fallback;
  }

  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function stripTicks(value) {
  return String(value || "").replace(/`/g, "");
}

function getNormalizationVerdict(normalizationSummary) {
  return (normalizationSummary || []).some((note) =>
    String(note).toLowerCase().includes("not fully normalized")
  )
    ? "Mostly 3NF"
    : "Normalized";
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

function SdaListCard({ title, icon: Icon, items }) {
  return (
    <article className="sda-list-card">
      <div className="card-heading">
        <Icon size={18} />
        <h3>{title}</h3>
      </div>
      <div className="sda-list-stack">
        {items.map((item) => (
          <article key={typeof item === "string" ? item : item.name} className="sda-list-item">
            {typeof item !== "string" ? (
              <>
                <strong>{item.name}</strong>
                <p>{item.detail || item.purpose}</p>
              </>
            ) : (
              <p>{item}</p>
            )}
          </article>
        ))}
      </div>
    </article>
  );
}

function SdaDiagramCard({ diagram }) {
  return (
    <article className="sda-diagram-card">
      <div className="card-heading">
        <Activity size={18} />
        <h3>{diagram.title}</h3>
      </div>
      <p className="sda-support-copy">{diagram.description}</p>
      <MermaidDiagram chart={diagram.mermaid} title={diagram.title} />
      <div className="explain-box">
        <strong>How to explain it</strong>
        <p>{diagram.talkingPoint}</p>
      </div>
    </article>
  );
}

function DbLayerCard({ layer, tables }) {
  const copy = DB_LAYER_COPY[layer] || {
    label: titleCase(layer),
    description: "This group is part of the current database schema."
  };

  return (
    <article className="db-layer-card">
      <span className={`db-layer-pill ${layer}`}>{copy.label}</span>
      <p>{copy.description}</p>
      <div className="db-token-row">
        {tables.map((table) => (
          <span key={table.table_name} className="db-token">
            {titleCase(table.table_name)}
          </span>
        ))}
      </div>
    </article>
  );
}

function DbTableCard({ table }) {
  return (
    <article className="db-table-card">
      <div className="db-table-card-header">
        <span className={`db-layer-pill ${table.layer}`}>{titleCase(table.layer)}</span>
        <strong>{titleCase(table.table_name)}</strong>
      </div>
      <p>{table.purpose}</p>
      <div className="db-table-meta">
        <div>
          <span>Primary key</span>
          <strong>{table.primary_key_columns?.join(", ") || "N/A"}</strong>
        </div>
        <div>
          <span>Foreign keys</span>
          <strong>{formatCount(table.foreign_keys?.length || 0)}</strong>
        </div>
        <div>
          <span>Indexes</span>
          <strong>{formatCount(table.index_names?.length || 0)}</strong>
        </div>
      </div>
      <p className="db-relationship-copy">{stripTicks(table.simple_relationship_summary)}</p>
    </article>
  );
}

function DbPresentationSection({ schemaData, presentationData }) {
  const schemaTables = schemaData?.tables || [];
  const latestProfile = presentationData?.latest_profile || null;
  const profileUpload = latestProfile?.upload || null;
  const profileMetrics = latestProfile?.dataset_profile || null;
  const featureProfiles = latestProfile?.feature_profiles || [];
  const profileWarnings = safeParseJson(profileMetrics?.warnings_json, []);
  const normalizationSummary = schemaData?.normalization_summary || [];
  const normalizationVerdict = getNormalizationVerdict(normalizationSummary);
  const erDiagram = (presentationData?.diagrams || []).find((diagram) => diagram.id === "erd") || null;
  const dbVivaNotes = (presentationData?.viva_notes || []).filter((note) => {
    const question = String(note.question || "").toLowerCase();
    return question.includes("normalize") || question.includes("location") || question.includes("database");
  });
  const layerGroups = DB_LAYER_ORDER.map((layer) => ({
    layer,
    tables: schemaTables.filter((table) => table.layer === layer)
  })).filter((group) => group.tables.length > 0);
  const showcaseTables = DB_CORE_TABLE_NAMES.map((tableName) =>
    schemaTables.find((table) => table.table_name === tableName)
  ).filter(Boolean);
  const relationshipTables = showcaseTables.filter((table) => (table.foreign_keys || []).length > 0);

  return (
    <main className="db-stage">
      <section className="flow-strip db-flow-strip">
        {dbSteps.map((step) => {
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
        title="Start with the Source Data"
        icon={Rows3}
        intro="Begin the DBS explanation by showing where the data came from, how large it is, and what the original columns look like."
      >
        <div className="story-grid two-column">
          <article className="surface-card">
            <div className="card-heading">
              <Rows3 size={18} />
              <h3>Latest Profiled Dataset</h3>
            </div>
            <div className="metric-strip">
              <MetricCard label="File" value={profileUpload?.filename || "Not profiled yet"} tone="accent" />
              <MetricCard label="Rows" value={formatCount(profileMetrics?.row_count ?? profileUpload?.row_count)} />
              <MetricCard
                label="Columns"
                value={formatCount(profileMetrics?.column_count ?? profileUpload?.column_count)}
              />
              <MetricCard label="Target" value={profileMetrics?.target_column || profileUpload?.target_column || "N/A"} />
              <MetricCard label="Duplicates" value={formatCount(profileMetrics?.duplicate_row_count)} tone="warm" />
              <MetricCard label="Missing cells" value={formatCount(profileMetrics?.missing_cell_count)} tone="warm" />
            </div>
            <div className="db-source-copy">
              <strong>Source path</strong>
              <p>{profileUpload?.source_path || "Profile a dataset first to show the file path."}</p>
            </div>
            <div className="db-source-copy">
              <strong>File size</strong>
              <p>{formatBytes(profileUpload?.file_size_bytes)}</p>
            </div>
            {profileWarnings.length > 0 ? (
              <div className="db-note-stack">
                {profileWarnings.map((warning) => (
                  <article key={warning} className="db-note-card">
                    <AlertTriangle size={16} />
                    <span>{warning}</span>
                  </article>
                ))}
              </div>
            ) : null}
          </article>

          <article className="surface-card">
            <div className="card-heading">
              <Database size={18} />
              <h3>How the Raw Data Looks</h3>
            </div>
            {featureProfiles.length > 0 ? (
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Role</th>
                      <th>Type</th>
                      <th>Example values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featureProfiles.slice(0, 6).map((feature) => {
                      const sampleValues = safeParseJson(feature.sample_values_json, []);
                      return (
                        <tr key={feature.column_name}>
                          <td>{feature.column_name}</td>
                          <td>{titleCase(feature.inferred_role)}</td>
                          <td>{feature.pandas_dtype}</td>
                          <td>
                            <div className="sample-value-row">
                              {sampleValues.map((sampleValue) => (
                                <span key={`${feature.column_name}-${sampleValue}`} className="sample-value-chip">
                                  {sampleValue}
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty-copy">Profile a dataset first to show the original columns and sample values.</p>
            )}
          </article>
        </div>
      </StorySection>

      <StorySection
        step="Step 2"
        title="Break the Data into Related Tables"
        icon={Database}
        intro="This is the core DBS part: instead of one large mixed table, the system stores raw, operational, and audit data separately."
      >
        <div className="db-layer-grid">
          {layerGroups.map((group) => (
            <DbLayerCard key={group.layer} layer={group.layer} tables={group.tables} />
          ))}
        </div>

        <article className="surface-card">
          <div className="card-heading">
            <Database size={18} />
            <h3>Core Tables for the Viva</h3>
          </div>
          <div className="db-table-grid">
            {showcaseTables.map((table) => (
              <DbTableCard key={table.table_name} table={table} />
            ))}
          </div>
        </article>

        <details className="details-card">
          <summary>Show all schema tables</summary>
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>Table</th>
                  <th>Layer</th>
                  <th>Columns</th>
                  <th>Foreign keys</th>
                </tr>
              </thead>
              <tbody>
                {schemaTables.map((table) => (
                  <tr key={table.table_name}>
                    <td>{titleCase(table.table_name)}</td>
                    <td>{titleCase(table.layer)}</td>
                    <td>{formatCount(table.columns?.length || 0)}</td>
                    <td>{formatCount(table.foreign_keys?.length || 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </StorySection>

      <StorySection
        step="Step 3"
        title="Explain the Normalization Clearly"
        icon={CheckCircle2}
        intro="Keep the answer honest and simple: the operational part is mostly normalized, but a few fields stay denormalized for project scope."
      >
        <div className="story-grid two-column">
          <article className="surface-card">
            <div className="card-heading">
              <CheckCircle2 size={18} />
              <h3>Normalization Verdict</h3>
            </div>
            <div className="metric-strip">
              <MetricCard label="Verdict" value={normalizationVerdict} tone="accent" />
              <MetricCard label="Operational design" value="Separated tables" />
              <MetricCard label="Still simple" value="Location, merchant" tone="warm" />
              <MetricCard label="Raw table style" value="Wide by design" />
            </div>
            <div className="db-note-stack">
              {normalizationSummary.map((note) => (
                <article key={note} className="db-note-card">
                  <CheckCircle2 size={16} />
                  <span>{stripTicks(note)}</span>
                </article>
              ))}
            </div>
          </article>

          <article className="surface-card">
            <div className="card-heading">
              <WandSparkles size={18} />
              <h3>Short Viva Answers</h3>
            </div>
            <div className="db-viva-grid">
              {dbVivaNotes.map((note) => (
                <article key={note.question} className="db-viva-card">
                  <strong>{note.question}</strong>
                  <p>{note.answer}</p>
                </article>
              ))}
            </div>
          </article>
        </div>
      </StorySection>

      <StorySection
        step="Step 4"
        title="Finish with the ER Diagram"
        icon={Activity}
        intro="End the DBS flow with one complete relationship view so you can point to primary keys, foreign keys, and parent-child links."
      >
        <div className="story-grid two-column db-diagram-grid">
          <article className="surface-card">
            <div className="card-heading">
              <Database size={18} />
              <h3>Live ER Diagram</h3>
            </div>
            {erDiagram ? (
              <MermaidDiagram chart={erDiagram.mermaid} title={erDiagram.title} />
            ) : (
              <p className="empty-copy">The ER diagram will appear here when the backend schema data is available.</p>
            )}
          </article>

          <article className="surface-card">
            <div className="card-heading">
              <CheckCircle2 size={18} />
              <h3>Main Relationships to Say Aloud</h3>
            </div>
            <div className="db-note-stack">
              {relationshipTables.map((table) => (
                <article key={table.table_name} className="db-note-card">
                  <Database size={16} />
                  <span>{stripTicks(table.simple_relationship_summary)}</span>
                </article>
              ))}
            </div>
            {erDiagram?.talking_points?.length ? (
              <div className="explain-box">
                <strong>Presentation tip</strong>
                <p>{erDiagram.talking_points[0]}</p>
              </div>
            ) : null}
          </article>
        </div>
      </StorySection>
    </main>
  );
}

function SdaPresentationSection({ sdaContent }) {
  const {
    introduction,
    problemStatement,
    methodology,
    architecture,
    toolsAndTechniques,
    guiDesign,
    validations,
    functionalRequirements,
    nonFunctionalRequirements,
    outsourceLibraries,
    testCases,
    diagrams
  } = sdaContent;

  const methodologyDiagrams = diagrams.filter((diagram) => ["use_case", "activity"].includes(diagram.id));
  const architectureDiagrams = diagrams.filter((diagram) =>
    ["network", "sequence", "interaction", "collaboration", "component"].includes(diagram.id)
  );
  const verificationDiagrams = diagrams.filter((diagram) => diagram.id === "state");

  return (
    <main className="sda-stage">
      <section className="flow-strip sda-flow-strip">
        {sdaSteps.map((step) => {
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
        step="Part 1"
        title="Introduction and Problem Statement"
        icon={GraduationCap}
        intro="Start the SDA section by explaining what the project is, what problem it solves, and why the interface is designed for presentation."
      >
        <div className="story-grid two-column">
          <article className="surface-card">
            <div className="card-heading">
              <GraduationCap size={18} />
              <h3>Introduction</h3>
            </div>
            <p className="support-copy">{introduction}</p>
            <div className="explain-box">
              <strong>Working project summary</strong>
              <p>
                The system profiles fraud-related data, stores structured records, evaluates models,
                predicts suspicious transactions, and presents the full workflow in one explainable UI.
              </p>
            </div>
          </article>

          <article className="surface-card">
            <div className="card-heading">
              <AlertTriangle size={18} />
              <h3>Problem Statement</h3>
            </div>
            <p className="support-copy">{problemStatement}</p>
            <div className="metric-strip">
              <MetricCard label="Subjects covered" value="AI, DBS, SDA" tone="accent" />
              <MetricCard label="Project mode" value="Local demo system" />
              <MetricCard label="Presentation support" value="Diagrams + UI" tone="warm" />
            </div>
          </article>
        </div>
      </StorySection>

      <StorySection
        step="Part 2"
        title="Methodology and Workflow"
        icon={Activity}
        intro="This part explains the process the project follows, from understanding data to generating final fraud outputs."
      >
        <div className="story-grid two-column">
          <SdaListCard title="Methodology / Workflow" icon={Activity} items={methodology} />
          <SdaListCard title="GUI Design Principles" icon={WandSparkles} items={guiDesign} />
        </div>
        <div className="sda-diagram-grid">
          {methodologyDiagrams.map((diagram) => (
            <SdaDiagramCard key={diagram.id} diagram={diagram} />
          ))}
        </div>
      </StorySection>

      <StorySection
        step="Part 3"
        title="Application Architecture and Tools"
        icon={Database}
        intro="Use these diagrams to explain how the frontend, backend, services, database, and model artifacts collaborate."
      >
        <div className="story-grid two-column">
          <SdaListCard title="Application Architecture" icon={Database} items={architecture} />
          <SdaListCard title="Tools and Techniques" icon={Brain} items={toolsAndTechniques} />
        </div>
        <div className="sda-diagram-grid">
          {architectureDiagrams.map((diagram) => (
            <SdaDiagramCard key={diagram.id} diagram={diagram} />
          ))}
        </div>
      </StorySection>

      <StorySection
        step="Part 4"
        title="Requirements, Validation, Testing, and Libraries"
        icon={CheckCircle2}
        intro="Finish the SDA explanation with concrete software-engineering details: requirements, validations, test cases, external libraries, and system state transitions."
      >
        <div className="story-grid two-column">
          <SdaListCard title="Functional Requirements" icon={CheckCircle2} items={functionalRequirements} />
          <SdaListCard title="Non-Functional Requirements" icon={Gauge} items={nonFunctionalRequirements} />
        </div>

        <div className="story-grid two-column">
          <SdaListCard title="Input Validations and GUI Checks" icon={Rows3} items={validations} />
          <SdaListCard title="Outsource Libraries" icon={Brain} items={outsourceLibraries} />
        </div>

        <article className="surface-card">
          <div className="card-heading">
            <CheckCircle2 size={18} />
            <h3>Testing with Test Cases</h3>
          </div>
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Scenario</th>
                  <th>Input</th>
                  <th>Expected Result</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {testCases.map((testCase) => (
                  <tr key={testCase.id}>
                    <td>{testCase.id}</td>
                    <td>{testCase.scenario}</td>
                    <td>{testCase.input}</td>
                    <td>{testCase.expected}</td>
                    <td>{testCase.result}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <div className="sda-diagram-grid">
          {verificationDiagrams.map((diagram) => (
            <SdaDiagramCard key={diagram.id} diagram={diagram} />
          ))}
        </div>
      </StorySection>
    </main>
  );
}

function App() {
  const [activeSection, setActiveSection] = useState("db");
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [schemaData, setSchemaData] = useState(null);
  const [presentationData, setPresentationData] = useState(null);
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
  const [backendMode, setBackendMode] = useState("unknown");

  useEffect(() => {
    if (activeSection === "db" && (!schemaData || !presentationData)) {
      void loadDbView();
    }
    if (activeSection === "ai" && (!datasetPreview || !recommendation || !modelData)) {
      void loadAiView();
    }
  }, [activeSection, datasetPreview, modelData, presentationData, recommendation, schemaData]);

  async function apiFetch(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }
    return payload;
  }

  async function checkBackendHealth() {
    try {
      await apiFetch("/health");
      return true;
    } catch {
      return false;
    }
  }

  function loadOfflineAiSnapshot() {
    setDatasetPreview(DEMO_AI_DATASET_PREVIEW);
    setRecommendation(DEMO_RECOMMENDATION);
    setModelData(DEMO_MODEL_DATA);
    setManualInput(DEMO_AI_DATASET_PREVIEW.manual_input_options.defaults);
    setBackendMode("offline");
    setError("");
    setConnectionNote(offlinePresentationMessage());
  }

  function loadOfflineDbSnapshot() {
    setSchemaData(DEMO_SCHEMA_DATA);
    setPresentationData(DEMO_PRESENTATION_DATA);
    setBackendMode("offline");
    setError("");
    setConnectionNote(offlinePresentationMessage());
  }

  async function loadAiView() {
    const requestSpecs = [
      { key: "datasetPreview", label: "dataset preview", path: "/ai/dataset-preview" },
      { key: "recommendation", label: "model recommendation", path: "/recommendations/current" },
      { key: "modelData", label: "saved model state", path: "/model/latest" }
    ];

    try {
      setLoadingKey("loading-ai-story");
      setError("");

      const backendReady = await checkBackendHealth();
      if (!backendReady) {
        loadOfflineAiSnapshot();
        return;
      }

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
        setBackendMode("live");
        setConnectionNote("");
      }

      if (failedLabels.length === requestSpecs.length) {
        loadOfflineAiSnapshot();
        return;
      }

      if (failedLabels.length > 0) {
        setError(`Some AI sections are still loading: ${failedLabels.join(", ")}.`);
      } else {
        setError("");
      }
    } catch (requestError) {
      if (requestError instanceof Error) {
        setError(requestError.message);
      }
      loadOfflineAiSnapshot();
    } finally {
      setLoadingKey("");
    }
  }

  async function loadDbView(forceRefresh = false) {
    if (!forceRefresh && schemaData && presentationData) {
      return;
    }

    try {
      setLoadingKey("loading-dbs-story");
      setError("");

      const backendReady = await checkBackendHealth();
      if (!backendReady) {
        loadOfflineDbSnapshot();
        return;
      }

      const [schemaPayload, presentationPayload] = await Promise.all([
        apiFetch("/schema"),
        apiFetch("/presentation")
      ]);

      setSchemaData(schemaPayload.schema);
      setPresentationData(presentationPayload.presentation);
      setBackendMode("live");
      setConnectionNote("");
    } catch (requestError) {
      if (requestError instanceof Error) {
        setError(requestError.message);
      }
      loadOfflineDbSnapshot();
    } finally {
      setLoadingKey("");
    }
  }

  async function handleTrain() {
    try {
      if (backendMode === "offline") {
        setConnectionNote("Offline presentation mode is using a saved snapshot. Live retraining needs the backend.");
        return;
      }
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
      if (backendMode === "offline") {
        setManualResult(buildOfflineManualPrediction(manualInput));
        setConnectionNote(offlinePresentationMessage());
        return;
      }
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
      if (backendMode === "offline") {
        const payload = buildOfflineTestSample(testSampleIndex);
        setTestSample(payload);
        setTestSampleIndex(payload.next_index || 0);
        setConnectionNote(offlinePresentationMessage());
        return;
      }
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
  const isOfflineMode = backendMode === "offline";
  const sdaContent = DEMO_SDA_CONTENT;
  const dbNormalizationVerdict = getNormalizationVerdict(schemaData?.normalization_summary || []);
  const dbTableCount = schemaData?.tables?.length || 0;
  const dbOperationalTableCount =
    schemaData?.tables?.filter((table) => table.layer === "operational").length || 0;
  const dbProfileRows =
    presentationData?.latest_profile?.dataset_profile?.row_count ||
    presentationData?.latest_profile?.upload?.row_count ||
    null;

  function handleSdaRefresh() {
    setConnectionNote("SDA section uses built-in presentation content and Mermaid diagrams.");
    setError("");
  }

  const currentHero =
    activeSection === "db"
      ? {
          eyebrow: "DBS Presentation UI",
          title: "Source Data, Table Design, and ER Diagram",
          description:
            "A clean database presentation flow: start from the source data, show how the tables are separated, explain normalization, and finish with the ER diagram.",
          actionLabel: "Reload DB Data",
          actionNote:
            isOfflineMode
              ? offlinePresentationMessage()
              : `This tab reads the live schema explanation from the backend. Start it with ${BACKEND_START_COMMAND}.`,
          action: () => void loadDbView(true),
          metrics: [
            { label: "Profiled rows", value: formatCount(dbProfileRows) },
            { label: "Schema tables", value: formatCount(dbTableCount) },
            { label: "Operational tables", value: formatCount(dbOperationalTableCount) },
            { label: "Normalization", value: dbNormalizationVerdict }
          ]
        }
      : activeSection === "sda"
        ? {
            eyebrow: "SDA Presentation UI",
            title: "Software Design and Architecture",
            description:
              "This section explains the working project through architecture content, requirements, testing notes, and a full diagram gallery designed for the SDA viva.",
            actionLabel: "Refresh SDA View",
            actionNote:
              "This tab is fully presentation-ready offline and includes use case, activity, network, sequence, interaction, collaboration, component, and state diagrams.",
            action: handleSdaRefresh,
            metrics: [
              { label: "Diagrams", value: formatCount(sdaContent.diagrams.length) },
              { label: "Functional reqs", value: formatCount(sdaContent.functionalRequirements.length) },
              { label: "Test cases", value: formatCount(sdaContent.testCases.length) },
              { label: "Libraries", value: formatCount(sdaContent.outsourceLibraries.length) }
            ]
          }
        : {
            eyebrow: "Evaluation Day Presentation UI",
            title: "AI Fraud Detection System",
            description:
              "A simplified, explainable interface built for presentation: show the dataset, justify the model, train it, give manual inputs, and finally test it on unseen data in real time.",
            actionLabel: "Reload AI Data",
            actionNote: isOfflineMode
              ? offlinePresentationMessage()
              : `If the backend is not running yet, start it with ${BACKEND_START_COMMAND}.`,
            action: () => void loadAiView(),
            metrics: [
              { label: "Dataset rows", value: formatCount(datasetPreview?.sample_count) },
              { label: "Fraud rate", value: formatPercent(datasetPreview?.fraud_rate) },
              { label: "Selected model", value: selectedModel },
              { label: "Test accuracy", value: formatPercent(testMetrics?.accuracy) }
            ]
          };

  return (
    <div className="presentation-shell">
      <div className="ambient-shape ambient-one" />
      <div className="ambient-shape ambient-two" />

      <header className="hero-panel">
        <div className="hero-copy">
          <div className="hero-eyebrow">{currentHero.eyebrow}</div>
          <h1>{currentHero.title}</h1>
          <p>{currentHero.description}</p>
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
          <button className="ghost-button" onClick={currentHero.action}>
            <RefreshCw size={16} />
            {currentHero.actionLabel}
          </button>
          <span className="hero-action-note">{currentHero.actionNote}</span>
        </div>

        <div className="hero-metric-row">
          {currentHero.metrics.map((metric) => (
            <HeroMetric key={metric.label} label={metric.label} value={metric.value} />
          ))}
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
                  <button className="primary-button" onClick={() => void handleTrain()} disabled={isOfflineMode}>
                    <Play size={16} />
                    {isOfflineMode ? "Training Disabled Offline" : hasTrainedModel ? "Retrain Model" : "Train Model"}
                  </button>
                  <button className="secondary-button" onClick={() => void loadAiView()}>
                    <RefreshCw size={16} />
                    Refresh View
                  </button>
                </div>
                {isOfflineMode ? (
                  <p className="helper-copy">
                    Offline presentation mode uses a saved snapshot. Live retraining needs the backend.
                  </p>
                ) : null}
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
        <DbPresentationSection schemaData={schemaData} presentationData={presentationData} />
      ) : null}

      {activeSection === "sda" ? (
        <SdaPresentationSection sdaContent={sdaContent} />
      ) : null}
    </div>
  );
}

export default App;
