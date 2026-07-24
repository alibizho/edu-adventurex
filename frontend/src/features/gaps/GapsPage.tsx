import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { apiMessage } from "../learning-data/apiClient";
import { backendLearningDataSource } from "../learning-data/backendLearningDataSource";
import { classSessionId, type AnalysisStatus, type FusionSegment } from "../learning-data/backend.types";
import { useBackendLearningPaths } from "../learning-data/useBackendLearningPaths";

function severity(segment: FusionSegment) {
  if (segment.quadrant === "blind_spot") return "HIGH";
  if (segment.quadrant === "aware_gap") return "MEDIUM";
  return "LOW";
}

export function GapsPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { entries, isLoading: pathsLoading, error: pathsError } = useBackendLearningPaths();
  const requestedPath = searchParams.get("path");
  const requestedClass = searchParams.get("class");
  const selection = useMemo(() => {
    const candidates = requestedPath ? entries.filter(({ path }) => path.path_id === requestedPath) : entries;
    for (const entry of candidates) {
      const unit = requestedClass
        ? entry.path.classes.find(({ class_id }) => class_id === requestedClass)
        : entry.path.classes.find(({ class_id }) => entry.memory.class_progress[class_id]?.status === "complete");
      if (unit) return { ...entry, unit };
    }
    return null;
  }, [entries, requestedClass, requestedPath]);
  const [analysis, setAnalysis] = useState<AnalysisStatus | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  useEffect(() => {
    if (!selection) return;
    let active = true;
    setAnalysis(null);
    setAnalysisError(null);
    backendLearningDataSource.getAnalysis(classSessionId(selection.path.path_id, selection.unit.class_id))
      .then((result) => { if (active) setAnalysis(result); })
      .catch((error) => { if (active) setAnalysisError(apiMessage(error)); });
    return () => { active = false; };
  }, [selection]);

  const gaps = useMemo(() => (analysis?.fusion?.per_segment ?? []).filter(({ quadrant }) => quadrant !== "mastery" && quadrant !== "productive_struggle"), [analysis]);
  useEffect(() => setSelectedId(gaps[0]?.segment_id ?? null), [analysis?.updated_at]);
  const selected = gaps.find(({ segment_id }) => segment_id === selectedId) ?? gaps[0] ?? null;

  async function retryAnalysis() {
    if (!selection) return;
    const id = classSessionId(selection.path.path_id, selection.unit.class_id);
    setAnalysisError(null);
    try {
      const result = await backendLearningDataSource.startAnalysis(id);
      setAnalysis(result);
    } catch (error) {
      setAnalysisError(apiMessage(error));
    }
  }

  const pending = analysis?.status === "pending" || analysis?.status === "running";
  return (
    <div className="screen gaps-screen halftone-screen">
      <AppHeader />
      <main className="gaps-main">
        <h1 className="dashboard-title">KNOWLEDGE GAPS</h1>
        <div className="gaps-summary"><span>CURRENT MODULE: {selection?.unit.title.toUpperCase() ?? "NONE"}</span><span>GAPS FOUND: {gaps.length}</span></div>
        {pathsLoading ? (
          <section className="gaps-empty retro-panel" role="status">LOADING BACKEND RESULTS...</section>
        ) : pathsError || !selection ? (
          <section className="gaps-empty retro-panel"><strong>NO COMPLETED BACKEND SESSION</strong><p>{pathsError ?? "COMPLETE A DYNAMIC CLASS TO GENERATE GAP DATA."}</p><button className="solid-action" onClick={() => navigate(ROUTES.material)}>START LEARNING</button></section>
        ) : pending ? (
          <section className="gaps-empty retro-panel" role="status"><strong>ANALYZING TEACHING TRANSFER...</strong><p>THE SUMMARY WILL UPDATE WHEN THE BACKGROUND RUN FINISHES.</p></section>
        ) : analysisError || analysis?.status === "failed" ? (
          <section className="gaps-empty retro-panel" role="alert"><strong>ANALYSIS UNAVAILABLE</strong><p>{analysisError ?? analysis?.error}</p><button className="solid-action" onClick={retryAnalysis}>RETRY ANALYSIS</button></section>
        ) : !selected ? (
          <section className="gaps-empty retro-panel"><strong>NO KNOWLEDGE GAPS DETECTED</strong><p>THE CURRENT FUSION RESULT CONTAINS NO BLIND SPOTS OR AWARE GAPS.</p><button className="outline-action" onClick={() => navigate(ROUTES.progress)}>BACK TO PROGRESS</button></section>
        ) : (
          <section className="gaps-workspace retro-panel">
            <aside className="gaps-list" aria-label="Detected knowledge gaps"><h2>DETECTED GAPS</h2>{gaps.map((gap) => (
              <button key={gap.segment_id} type="button" className={`gap-selector gap-selector--${severity(gap).toLowerCase()} ${gap.segment_id === selected.segment_id ? "is-selected" : ""}`} onClick={() => setSelectedId(gap.segment_id)}>
                <span>SEGMENT {gap.segment_id + 1}</span><strong>{severity(gap)}</strong>
              </button>
            ))}</aside>
            <article className="gap-detail"><h2>{selected.quadrant.replace("_", " ").toUpperCase()}</h2>
              <section><h3>WHY IT MATTERS</h3><p>{selected.quadrant === "blind_spot" ? "THE EXPLANATION SOUNDED CONFIDENT BUT DID NOT TRANSFER CORRECTLY." : "THE EXPLANATION SHOWED UNCERTAINTY AND NEEDS A CLEARER CAUSAL LINK."}</p></section>
              <section><h3>EVIDENCE</h3><blockquote>“{selected.text || "NO TRANSCRIPT TEXT RETURNED."}”</blockquote></section>
              <section><h3>RECOMMENDED REVIEW</h3><p>REOPEN {selection.unit.title.toUpperCase()} AND EXPLAIN THIS SEGMENT WITH A CONCRETE EXAMPLE.</p></section>
              <div className="dashboard-actions"><button className="outline-action" onClick={() => navigate(ROUTES.progress)}>BACK TO PROGRESS</button><button className="solid-action" onClick={() => navigate(`${ROUTES.study}?path=${encodeURIComponent(selection.path.path_id)}&class=${encodeURIComponent(selection.unit.class_id)}`)}>REVIEW AGAIN</button></div>
            </article>
          </section>
        )}
      </main>
      <StatusBar label={pending ? "ANALYSIS_RUNNING" : selected ? "GAP_ANALYSIS_READY" : "NO_GAP_DATA"} full meta="SOURCE: FUSION API" />
    </div>
  );
}
