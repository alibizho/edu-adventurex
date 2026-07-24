import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { apiMessage } from "../learning-data/apiClient";
import { backendLearningDataSource } from "../learning-data/backendLearningDataSource";
import { classSessionId, type AnalysisStatus, type GrowthPath, type PathMemory, type SessionSnapshot } from "../learning-data/backend.types";
import { StudentSidebar } from "../study/components/StudentSidebar";
import type { StudyToolId } from "../study/study.types";

const TOOLS = [{ id: "progress", label: "Progress" }, { id: "tutorial", label: "Tutorial" }, { id: "reset", label: "Reset" }] as const;
const clamp = (value: number) => Math.round(Math.max(0, Math.min(100, value)));

function formatDuration(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  return [hours, minutes, rest].map((value) => String(value).padStart(2, "0")).join(":");
}

export function BackendSummaryPage({ pathId, classId }: { pathId: string; classId: string }) {
  const navigate = useNavigate();
  const [path, setPath] = useState<GrowthPath | null>(null);
  const [memory, setMemory] = useState<PathMemory | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const sessionId = classSessionId(pathId, classId);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [loadedPath, loadedMemory, loadedSnapshot] = await Promise.all([
        backendLearningDataSource.getPath(pathId),
        backendLearningDataSource.getMemory(pathId),
        backendLearningDataSource.getSession(sessionId),
      ]);
      setPath(loadedPath);
      setMemory(loadedMemory);
      setSnapshot(loadedSnapshot);
      try {
        setAnalysis(await backendLearningDataSource.getAnalysis(sessionId));
      } catch {
        setAnalysis(null);
      }
    } catch (caught) {
      setError(apiMessage(caught));
    } finally {
      setIsLoading(false);
    }
  }, [pathId, sessionId]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (analysis?.status !== "pending" && analysis?.status !== "running") return;
    const timer = window.setTimeout(async () => {
      try {
        const next = await backendLearningDataSource.getAnalysis(sessionId);
        setAnalysis(next);
        if (next.status === "complete") setSnapshot(await backendLearningDataSource.getSession(sessionId));
      } catch (caught) {
        setError(apiMessage(caught));
      }
    }, 2500);
    return () => window.clearTimeout(timer);
  }, [analysis?.status, analysis?.updated_at, sessionId]);

  const unit = path?.classes.find(({ class_id }) => class_id === classId) ?? null;
  const progress = memory?.class_progress[classId];
  const fusion = analysis?.fusion ?? snapshot?.fusion;
  const run = analysis?.run ?? snapshot?.run;
  const gapCount = Object.entries(fusion?.quadrant_counts ?? {}).reduce((total, [key, value]) => key === "blind_spot" || key === "aware_gap" ? total + value : total, 0);
  const duration = progress?.started_at && progress.completed_at ? Math.max(0, Math.round(progress.completed_at - progress.started_at)) : 0;
  const metrics = useMemo(() => {
    const segments = fusion?.per_segment ?? [];
    const clarity = segments.length ? 100 - segments.reduce((sum, item) => sum + item.disturbance * 100, 0) / segments.length : 0;
    return [
      { id: "transfer", label: "TRANSFER", score: clamp(((run?.delta_overall ?? 0) + 1) * 50) },
      { id: "survival", label: "QUESTION QUALITY", score: clamp((run?.survival_rate ?? 0) * 100) },
      { id: "clarity", label: "TEACHING CLARITY", score: clamp(clarity) },
      { id: "calibration", label: "SELF CALIBRATION", score: clamp(((fusion?.calibration_rho ?? 0) + 1) * 50) },
    ];
  }, [fusion, run]);
  const pending = analysis?.status === "pending" || analysis?.status === "running";
  const rank = !run ? "ANALYSIS PENDING" : run.delta_overall >= 0.4 ? "ELITE TEACHER" : run.delta_overall >= 0.15 ? "SKILLED TEACHER" : "DEVELOPING TEACHER";

  async function retryAnalysis() {
    setError(null);
    try { setAnalysis(await backendLearningDataSource.startAnalysis(sessionId)); }
    catch (caught) { setError(apiMessage(caught)); }
  }

  function handleTool(tool: StudyToolId) {
    if (tool === "progress") navigate(`${ROUTES.progress}?path=${encodeURIComponent(pathId)}`);
    else if (tool === "tutorial") setMessage("THE SUMMARY UPDATES AUTOMATICALLY WHEN BACKGROUND ANALYSIS FINISHES.");
    else void load();
  }

  return (
    <div className="screen summary-screen">
      <AppHeader />
      <main className="summary-layout">
        <StudentSidebar student={{ name: "AI STUDENT", readiness: progress?.readiness ?? 0 }} tools={TOOLS} message={message} onToolAction={handleTool} avatarVariant="student" />
        <section className="summary-canvas">
          {isLoading ? <div className="summary-empty retro-panel" role="status">LOADING BACKEND SUMMARY...</div> : error && !path ? (
            <div className="summary-empty retro-panel" role="alert"><h1>SUMMARY UNAVAILABLE</h1><p>{error}</p><button className="solid-action" onClick={() => void load()}>RETRY</button></div>
          ) : !path || !unit || progress?.status !== "complete" ? (
            <div className="summary-empty retro-panel"><h1>NO COMPLETED SESSION</h1><p>COMPLETE THIS CLASS BEFORE OPENING ITS SUMMARY.</p><button className="solid-action" onClick={() => navigate(`${ROUTES.study}?path=${encodeURIComponent(pathId)}&class=${encodeURIComponent(classId)}`)}>BACK TO CLASS</button></div>
          ) : (
            <div className="summary-dashboard">
              <h1>{path.confirmed_topic.toUpperCase()}</h1>
              <div className="summary-top-grid">
                <section className="summary-panel growth-panel"><h2><i />GROWTH TIMELINE</h2><div className="growth-track">{path.classes.map((item) => {
                  const status = memory?.class_progress[item.class_id]?.status === "complete" ? "completed" : item.class_id === classId ? "next" : "locked";
                  return <div className={`growth-stop growth-stop--${status}`} key={item.class_id}>{status === "completed" && <span>COMPLETED</span>}<b>{status === "completed" ? "✓" : status === "next" ? "■" : "○"}</b><strong>{item.title}</strong></div>;
                })}</div></section>
                <section className="master-panel"><h2>★ MASTER STATUS</h2><p>{pending ? "TRANSFER ANALYSIS IS RUNNING IN THE BACKGROUND." : `YOU COMPLETED ${unit.title.toUpperCase()} WITH ${progress.readiness}% READINESS.`}</p><strong>RANK: {rank}</strong>{(analysis?.status === "failed" || (!analysis && !run)) && <button className="outline-action summary-retry" onClick={retryAnalysis}>RETRY ANALYSIS</button>}</section>
              </div>
              <div className="summary-bottom-grid">
                <section className="summary-panel mastery-panel"><h2>CONCEPTS MASTERED</h2><div className="mastery-grid">{metrics.map((metric) => <div className="mastery-item" key={metric.id}><span className="mastery-icon mastery-icon--circle"><i /></span><div><strong>{metric.label}</strong><span role="progressbar" aria-valuenow={metric.score} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${metric.score}%` }} /></span></div></div>)}</div></section>
                <section className="summary-panel statistics-panel"><h2>TEACHING STATISTICS</h2><dl><div><dt>TEACHING TURNS</dt><dd>{progress.turn_count}</dd></div><div><dt>QUESTIONS RECORDED</dt><dd>{snapshot?.questions.length ?? 0}</dd></div><div><dt>GAPS DISCOVERED</dt><dd>{gapCount}</dd></div></dl></section>
              </div>
              {error && <p className="summary-api-error" role="alert">{error}</p>}
            </div>
          )}
        </section>
      </main>
      <StatusBar label={pending ? "ANALYZING_SESSION" : run ? "SESSION_ANALYZED" : "SESSION_COMPLETE"} full meta={`${formatDuration(duration)} SESSION TIME`} />
    </div>
  );
}
