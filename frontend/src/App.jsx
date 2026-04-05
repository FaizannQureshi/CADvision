import { useEffect, useMemo, useState } from "react";
import "./App.css";

// Dev: talk to local API. Prod (Docker/nginx): use same-origin unless VITE_API_URL is set.
const API_URL =
  import.meta.env.VITE_API_URL !== undefined && import.meta.env.VITE_API_URL !== ""
    ? import.meta.env.VITE_API_URL
    : import.meta.env.DEV
      ? "http://localhost:8000"
      : "";

const toDataUrl = (base64) => (base64 ? `data:image/png;base64,${base64}` : null);

function App() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [modalImage, setModalImage] = useState(null);
  const [modalZoom, setModalZoom] = useState(1);

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  useEffect(() => {
    if (!modalImage) {
      document.body.style.overflow = "";
      return;
    }

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setModalImage(null);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [modalImage]);

  const handleFileChange = (event, setter) => {
    const selected = event.target.files?.[0];
    setter(selected || null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file1 || !file2) {
      setError("Choose both drawings: your baseline file first, then the revised version.");
      return;
    }

    setError("");
    setIsSubmitting(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file1", file1, file1.name);
      formData.append("file2", file2, file2.name);

      const response = await fetch(`${API_URL}/compare`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        let detail = "Comparison failed.";
        try {
          const payload = JSON.parse(text);
          detail = payload.detail || detail;
        } catch {
          detail = text || detail;
        }
        throw new Error(detail);
      }

      const payload = await response.json();
      setResult(payload);
    } catch (err) {
      setError(err.message || "Something went wrong. Check your connection and try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const combinedSrc = useMemo(
    () => toDataUrl(result?.images?.highlighted_1),
    [result]
  );

  const openModal = (src, alt) => {
    if (!src) return;
    setModalImage({ src, alt });
    setModalZoom(1);
  };

  const closeModal = () => setModalImage(null);

  const adjustZoom = (delta) => {
    setModalZoom((prev) => clamp(parseFloat((prev + delta).toFixed(2)), 1, 4));
  };

  const handleZoomInput = (event) => {
    const value = parseFloat(event.target.value);
    if (!Number.isNaN(value)) {
      setModalZoom(clamp(value, 1, 4));
    }
  };

  const handleWheelZoom = (event) => {
    event.preventDefault();
    event.stopPropagation();
    const direction = event.deltaY < 0 ? 0.1 : -0.1;
    adjustZoom(direction);
  };

  return (
    <div className="page">
      <header className="top-strip">
        <div className="top-strip__inner">
          <div className="site-brand">
            <span className="brand-mark" aria-hidden="true">
              CV
            </span>
            <div className="brand-lockup-text">
              <span className="brand-title">CADVision</span>
              <span className="brand-tagline">AI drawing comparison</span>
            </div>
          </div>
          <div className="top-strip__end">
            <span className="top-caption">CADVISION</span>
          </div>
        </div>
      </header>

      <div className="page-content">
        <section className="hero animate-section animate-section--1">
          <div className="hero-copy">
            <p className="hero-eyebrow">Drawing review, simplified</p>
            <h1>See what changed between your CAD revisions</h1>
            <p>
              Upload your baseline and updated exports — we align the sheets, highlight adds and removals,
              and summarize the differences so you can review faster.
            </p>
          </div>
          <div className="hero-visual">
            <div className="hero-visual-frame">
              <span className="hero-visual-label">How it works</span>
              <ol className="hero-steps">
                <li>
                  <span className="hero-step-num" aria-hidden="true">1</span>
                  <span>
                    <strong>Upload two files</strong> — PNG, JPG, or PDF (first page for PDFs).
                  </span>
                </li>
                <li>
                  <span className="hero-step-num" aria-hidden="true">2</span>
                  <span>
                    <strong>We compare &amp; highlight</strong> — greens for additions, reds for removals.
                  </span>
                </li>
                <li>
                  <span className="hero-step-num" aria-hidden="true">3</span>
                  <span>
                    <strong>Read the summary</strong> — optional AI notes for your review or handoff.
                  </span>
                </li>
              </ol>
            </div>
          </div>
        </section>

        <main className="workspace animate-section animate-section--2">
          <section className="workspace-card">
            <div className="workspace-header">
              <h2>Upload your drawings</h2>
              <p className="workspace-lead">
                Put <strong>Drawing A</strong> first (baseline or older revision), then <strong>Drawing B</strong>{" "}
                (updated). Same sheet or view works best for a clear diff.
              </p>
            </div>

            <form className="upload-form" onSubmit={handleSubmit}>
              <label className="upload-field">
                <span className="upload-field-title">Drawing A — baseline</span>
                <span className="upload-field-hint">Older revision, reference, or “before”</span>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.pdf"
                  onChange={(event) => handleFileChange(event, setFile1)}
                  disabled={isSubmitting}
                  aria-describedby="hint-a"
                />
                <span id="hint-a" className="file-picked" aria-live="polite">
                  {file1 ? file1.name : "No file chosen yet"}
                </span>
              </label>
              <label className="upload-field">
                <span className="upload-field-title">Drawing B — revised</span>
                <span className="upload-field-hint">Newer revision or “after”</span>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.pdf"
                  onChange={(event) => handleFileChange(event, setFile2)}
                  disabled={isSubmitting}
                  aria-describedby="hint-b"
                />
                <span id="hint-b" className="file-picked" aria-live="polite">
                  {file2 ? file2.name : "No file chosen yet"}
                </span>
              </label>

              <button
                className={`compare-button${isSubmitting ? " compare-button--loading" : ""}`}
                type="submit"
                disabled={isSubmitting}
                aria-busy={isSubmitting}
              >
                {isSubmitting ? "Analyzing your drawings…" : "Compare drawings"}
              </button>
            </form>

            {error && (
              <div className="status error" role="alert">
                {error}
              </div>
            )}
          </section>

          {result && (
            <section className="results animate-section animate-section--3">
              <div className="results-header">
                {(() => {
                  const aiText = result?.ai_summary || "";
                  const noChanges = /no change(s)?|no structural changes detected/i.test(aiText);
                  return (
                    <h2>{noChanges ? "No changes detected" : "Here’s your comparison"}</h2>
                  );
                })()}
                <p>
                  {(() => {
                    const aiText = result?.ai_summary || "";
                    const noChanges = /no change(s)?|no structural changes detected/i.test(aiText);
                    return noChanges
                      ? "These two files look the same to our pipeline — try another pair if you expected edits."
                      : "Use the map below to spot changes. Green usually means something new; red means removed. Scroll down for a written summary.";
                  })()}
                </p>

                {!(/no change(s)?|no structural changes detected/i.test(result?.ai_summary || "")) && (
                  <div className="legend">
                    <span className="legend-item">
                      <span className="legend-swatch legend-add" />
                      Added in Drawing B (green)
                    </span>
                    <span className="legend-item">
                      <span className="legend-swatch legend-del" />
                      Removed vs Drawing A (red)
                    </span>
                  </div>
                )}
              </div>

              {combinedSrc && (
                <figure className="image-card combined-image">
                  <img src={combinedSrc} alt="Combined comparison" />
                  <figcaption>Combined view with highlights</figcaption>
                  <div className="figure-actions">
                    <button
                      type="button"
                      className="view-button"
                      onClick={() => openModal(combinedSrc, "Combined comparison")}
                    >
                      View Full Size
                    </button>
                    <a
                      href={combinedSrc}
                      download="cad_comparison.png"
                      className="download-link"
                    >
                      Download
                    </a>
                  </div>
                </figure>
              )}

              <div className="summary-card">
                <div className="summary-card-header">
                  <div>
                    <p className="summary-eyebrow">AI summary</p>
                    <h3>What changed (plain language)</h3>
                  </div>
                </div>
                {result?.ai_summary ? (
                  <div 
                    className="summary-content markdown-content"
                    dangerouslySetInnerHTML={{ __html: result.ai_summary }}
                  />
                ) : (
                  <p className="summary-placeholder">
                    When the comparison finishes, a short written summary will show up here.
                  </p>
                )}
              </div>
            </section>
          )}
        </main>
      </div>

      <footer className="site-footer">
        <span>© {new Date().getFullYear()} CADVision</span>
        <span className="site-footer-sep" aria-hidden="true">
          ·
        </span>
        <span>Built for quick drawing reviews</span>
      </footer>

      {modalImage && (
        <div className="modal-backdrop" role="presentation" onClick={closeModal}>
          <div
            className="modal-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={modalImage.alt}
            onClick={(event) => event.stopPropagation()}
          >
            <button type="button" className="modal-close" onClick={closeModal}>
              Close
            </button>
            <div className="modal-toolbar">
              <button type="button" onClick={() => adjustZoom(-0.1)}>
                −
              </button>
              <input
                type="range"
                min="1"
                max="4"
                step="0.1"
                value={modalZoom}
                onChange={handleZoomInput}
                aria-label="Zoom level"
              />
              <span>{Math.round(modalZoom * 100)}%</span>
              <button type="button" onClick={() => adjustZoom(0.1)}>
                +
              </button>
              <button type="button" onClick={() => setModalZoom(1)}>
                Reset
              </button>
            </div>
            <div className="modal-image-container" onWheel={handleWheelZoom}>
              <img
                src={modalImage.src}
                alt={modalImage.alt}
                style={{ transform: `scale(${modalZoom})` }}
              />
            </div>
            <p className="modal-caption">{modalImage.alt}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;