import { startTransition, useEffect, useState } from "react";
import createDOMPurify from "dompurify";

let diagramCounter = 0;
let mermaidModulePromise = null;
let mermaidInitialized = false;
const diagramCache = new Map();

function ensureDOMPurify() {
  if (typeof window === "undefined") {
    return null;
  }

  const maybePurifier = createDOMPurify;
  if (typeof maybePurifier?.sanitize === "function") {
    window.DOMPurify = maybePurifier;
    return maybePurifier;
  }

  if (typeof maybePurifier === "function") {
    const purifier = maybePurifier(window);
    window.DOMPurify = purifier;
    return purifier;
  }

  return null;
}

async function loadMermaid() {
  if (!mermaidModulePromise) {
    mermaidModulePromise = import("mermaid").then((module) => {
      const mermaid = module.default;
      ensureDOMPurify();
      if (!mermaidInitialized) {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "neutral",
          flowchart: {
            useMaxWidth: true,
            htmlLabels: false
          }
        });
        mermaidInitialized = true;
      }
      return mermaid;
    });
  }

  return mermaidModulePromise;
}

function MermaidDiagram({ chart, title }) {
  const [svg, setSvg] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const elementId = `mermaid-diagram-${diagramCounter++}`;

    async function renderDiagram() {
      if (!chart) {
        startTransition(() => {
          setSvg("");
          setError("");
        });
        return;
      }

      if (diagramCache.has(chart)) {
        startTransition(() => {
          setSvg(diagramCache.get(chart) || "");
          setError("");
        });
        return;
      }

      try {
        const mermaid = await loadMermaid();
        const normalizedChart = chart.trim();
        await mermaid.parse(normalizedChart);
        const { svg: renderedSvg } = await mermaid.render(elementId, normalizedChart);
        if (!cancelled) {
          if (renderedSvg.includes("Syntax error in text")) {
            throw new Error("Mermaid returned an error diagram instead of a valid render.");
          }
          diagramCache.set(chart, renderedSvg);
          startTransition(() => {
            setSvg(renderedSvg);
            setError("");
          });
        }
      } catch (renderError) {
        if (!cancelled) {
          startTransition(() => {
            setError(renderError instanceof Error ? renderError.message : "Diagram rendering failed.");
            setSvg("");
          });
        }
      }
    }

    void renderDiagram();

    return () => {
      cancelled = true;
    };
  }, [chart]);

  if (error) {
    return (
      <div className="diagram-fallback">
        <strong>{title}</strong>
        <p>{error}</p>
        <details className="diagram-source-details">
          <summary>Show Mermaid source</summary>
          <pre className="code-panel">{chart}</pre>
        </details>
      </div>
    );
  }

  if (!svg) {
    return <div className="diagram-loading">Rendering diagram...</div>;
  }

  return <div className="mermaid-stage" dangerouslySetInnerHTML={{ __html: svg }} />;
}

export default MermaidDiagram;
