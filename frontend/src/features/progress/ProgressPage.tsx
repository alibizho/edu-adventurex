import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { StudentSidebar } from "../study/components/StudentSidebar";
import type { StudyToolConfig, StudyToolId } from "../study/study.types";
import { useBackendLearningPaths } from "../learning-data/useBackendLearningPaths";

const TOOLS: readonly StudyToolConfig[] = [
  { id: "progress", label: "Progress" },
  { id: "tutorial", label: "Tutorial" },
  { id: "reset", label: "Reset" },
];

function formatTime(value: number | null | undefined) {
  return value ? new Date(value * 1000).toLocaleString() : "--";
}

export function ProgressPage() {
  const navigate = useNavigate();
  const { entries, isLoading, error, reload } = useBackendLearningPaths();
  const [message, setMessage] = useState<string | null>(null);
  const rows = useMemo(() => entries.flatMap(({ path, memory }) => path.classes.map((unit) => ({
    path,
    unit,
    progress: memory.class_progress[unit.class_id],
  }))), [entries]);
  const completed = rows.filter(({ progress }) => progress?.status === "complete").length;
  const average = rows.length
    ? Math.round(rows.reduce((total, row) => total + (row.progress?.readiness ?? 0), 0) / rows.length)
    : 0;
  const next = rows.find(({ progress }) => progress?.status === "in_progress")
    ?? rows.find(({ progress }) => progress?.status !== "complete")
    ?? rows[0];
  const gapTarget = [...rows].reverse().find(({ progress }) => progress?.status === "complete");

  function handleTool(tool: StudyToolId) {
    if (tool === "tutorial") {
      setMessage("PROGRESS IS LOADED FROM THE FASTAPI + POSTGRES LEARNING STORE.");
    } else if (tool === "reset") {
      void reload();
      setMessage("PROGRESS RELOADED.");
    }
  }

  return (
    <div className="screen progress-screen">
      <AppHeader />
      <main className="progress-layout">
        <StudentSidebar student={{ name: "AI STUDENT", readiness: average }} tools={TOOLS} message={message} onToolAction={handleTool} />
        <section className="progress-canvas halftone-screen">
          <div className="progress-main">
            <h1 className="dashboard-title">LEARNING PROGRESS</h1>
            <section className="progress-panel retro-panel">
              <div className="progress-summary">
                <span>TOPICS <strong>{entries.length}</strong></span>
                <span>COMPLETED <strong>{completed}</strong></span>
                <span>AVERAGE READINESS <strong>{average}%</strong></span>
              </div>
              {isLoading ? (
                <div className="dashboard-empty" role="status"><strong>LOADING BACKEND PROGRESS...</strong></div>
              ) : error ? (
                <div className="dashboard-empty" role="alert"><strong>BACKEND UNAVAILABLE</strong><p>{error}</p><button className="outline-action" onClick={() => void reload()}>RETRY</button></div>
              ) : rows.length === 0 ? (
                <div className="dashboard-empty"><strong>NO LEARNING PATHS YET</strong><p>SUBMIT MATERIALS TO BUILD YOUR FIRST PATH.</p></div>
              ) : (
                <div className="progress-table-wrap">
                  <table className="progress-table">
                    <thead><tr><th>TOPIC</th><th>CLASS</th><th>READINESS</th><th>STATUS</th><th>LAST SESSION</th></tr></thead>
                    <tbody>{rows.map(({ path, unit, progress }) => {
                      const readiness = progress?.readiness ?? 0;
                      const status = progress?.status ?? "not_started";
                      return (
                        <tr key={`${path.path_id}:${unit.class_id}`}>
                          <td data-label="TOPIC">{path.confirmed_topic}</td>
                          <td data-label="CLASS">{unit.title}</td>
                          <td data-label="READINESS"><div className="readiness-cell"><span>{readiness}%</span><span className="table-progress"><i style={{ width: `${readiness}%` }} /></span></div></td>
                          <td data-label="STATUS"><span className={`progress-status progress-status--${status}`}>{status.replace("_", " ").toUpperCase()}</span></td>
                          <td data-label="LAST SESSION">{formatTime(progress?.completed_at ?? progress?.started_at)}</td>
                        </tr>
                      );
                    })}</tbody>
                  </table>
                </div>
              )}
              <div className="dashboard-actions">
                <button type="button" className="outline-action" disabled={!gapTarget} onClick={() => gapTarget && navigate(`${ROUTES.gaps}?path=${encodeURIComponent(gapTarget.path.path_id)}&class=${encodeURIComponent(gapTarget.unit.class_id)}`)}>VIEW GAPS</button>
                <button type="button" className="solid-action" onClick={() => next ? navigate(`${ROUTES.study}?path=${encodeURIComponent(next.path.path_id)}&class=${encodeURIComponent(next.unit.class_id)}`) : navigate(ROUTES.material)}>{rows.length ? "CONTINUE LEARNING" : "START LEARNING"}</button>
              </div>
            </section>
          </div>
        </section>
      </main>
      <StatusBar label={error ? "BACKEND_ERROR" : "PROGRESS_OVERVIEW"} full meta="SOURCE: POSTGRES" />
    </div>
  );
}
