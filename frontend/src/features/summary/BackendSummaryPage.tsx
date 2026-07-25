import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { apiMessage } from "../learning-data/apiClient";
import { backendLearningDataSource } from "../learning-data/backendLearningDataSource";
import { classChecklist, classSessionId, type AnalysisStatus, type GrowthPath, type PathMemory, type SessionSnapshot } from "../learning-data/backend.types";
import { ObjectiveChecklist } from "../study/components/ObjectiveChecklist";
import { StudentSidebar } from "../study/components/StudentSidebar";
import type { StudyToolId } from "../study/study.types";

const TOOLS = [{ id: "map", label: "Map" }, { id: "tutorial", label: "Tutorial" }, { id: "reset", label: "Reset" }] as const;

/**
 * What each number on the statistics panel actually counts, in one line.
 *
 * `gaps` is the one to be careful with. It is not the classroom student's opinion — it counts
 * TRANSCRIPT SEGMENTS from the background transfer run, where readers given your transcript scored
 * no better than a control group that never heard it (fusion.py: blind_spot + aware_gap, both
 * `delta <= 0`). So it is a comparison against a baseline, not a count of wrong answers, and it is
 * per utterance rather than per concept — several may belong to one idea.
 */
const STAT_HINTS = {
  turns: "EVERY STRETCH OF YOUR TEACHING THE CLASS HEARD AND ANALYZED.",
  questions: "TIMES A STUDENT STOPPED YOU BECAUSE SOMETHING SOUNDED UNCLEAR.",
  gaps: "STRETCHES WHERE READERS GIVEN YOUR TEACHING SCORED NO BETTER THAN ONES WHO NEVER HEARD IT.",
} as const;

function Stat({ label, hint, value }: { label: string; hint: string; value: number | string }) {
  return (
    <div>
      {/* The hint lives inside the <dt> rather than a sibling: dl > div may only hold dt/dd. */}
      <dt>{label}<span>{hint}</span></dt>
      <dd>{value}</dd>
    </div>
  );
}

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
  // Gaps come out of the transfer analysis, which runs after the class ends. Until it lands there
  // is no count — showing 0 would assert "no gaps found" when nothing has been looked for yet.
  const gapCount = fusion
    ? Object.entries(fusion.quadrant_counts ?? {}).reduce((total, [key, value]) => key === "blind_spot" || key === "aware_gap" ? total + value : total, 0)
    : null;
  const duration = progress?.started_at && progress.completed_at ? Math.max(0, Math.round(progress.completed_at - progress.started_at)) : 0;
  const pending = analysis?.status === "pending" || analysis?.status === "running";
  const rank = !run ? "ANALYSIS PENDING" : run.delta_overall >= 0.4 ? "ELITE TEACHER" : run.delta_overall >= 0.15 ? "SKILLED TEACHER" : "DEVELOPING TEACHER";
  // "You explained every goal" is only true if you actually did the explaining. The backend marks
  // a class guided-explanation when a student had to hand over an answer even once, and dropping
  // that here turned being told into a clean pass.
  const guided = progress?.completion_mode === "guided-explanation";
  const masterStatus = !unit ? "" : [
    progress?.passed_on_mastery
      ? `YOU COVERED EVERY GOAL IN ${unit.title.toUpperCase()}.`
      : `YOU COVERED ${progress?.readiness ?? 0}% OF THE GOALS IN ${unit.title.toUpperCase()}.`,
    guided
      ? "SOME OF IT WAS EXPLAINED TO YOU RATHER THAN BY YOU — WORTH A SECOND PASS."
      : "EVERY GOAL YOU COVERED, YOU EXPLAINED YOURSELF.",
  ].join(" ");

  async function retryAnalysis() {
    setError(null);
    try { setAnalysis(await backendLearningDataSource.startAnalysis(sessionId)); }
    catch (caught) { setError(apiMessage(caught)); }
  }

  function openClass(target: string) {
    navigate(`${ROUTES.study}?path=${encodeURIComponent(pathId)}&class=${encodeURIComponent(target)}`);
  }

  function handleTool(tool: StudyToolId) {
    if (tool === "map") navigate(ROUTES.map);
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
                {/* Every class is reachable from here — the summary is where you decide what to
                    teach next, and making that a trip back through the concept map was friction
                    for no reason. State per class comes from the stored progress rather than
                    "is it the one being summarised": a class part-way through used to read the
                    same as one never opened. */}
                <section className="summary-panel growth-panel"><h2><i />GROWTH TIMELINE</h2><div className="growth-track">{path.classes.map((item) => {
                  const state = memory?.class_progress[item.class_id]?.status;
                  const stop = state === "complete" ? "completed" : state === "in_progress" ? "active" : "open";
                  const isCurrent = item.class_id === classId;
                  return (
                    <button
                      type="button"
                      key={item.class_id}
                      className={`growth-stop growth-stop--${stop}${isCurrent ? " is-current" : ""}`}
                      // The visible label is a tick and a title; spell out what pressing it does.
                      aria-label={`TEACH ${item.title.toUpperCase()}`}
                      aria-current={isCurrent ? "step" : undefined}
                      onClick={() => openClass(item.class_id)}
                    >
                      {stop !== "open" && <span>{stop === "completed" ? "COMPLETED" : "IN PROGRESS"}</span>}
                      <b>{stop === "completed" ? "✓" : stop === "active" ? "■" : "○"}</b>
                      <strong>{item.title}</strong>
                    </button>
                  );
                })}</div><p className="growth-hint">PICK ANY CLASS TO TEACH IT.</p></section>
                <section className="master-panel"><h2>★ MASTER STATUS</h2><p>{pending ? "TRANSFER ANALYSIS IS RUNNING IN THE BACKGROUND." : masterStatus}</p><strong>RANK: {rank}</strong>{(analysis?.status === "failed" || (!analysis && !run)) && <button className="outline-action summary-retry" onClick={retryAnalysis}>RETRY ANALYSIS</button>}</section>
              </div>
              <div className="summary-bottom-grid">
                {/* What they were asked to explain, and what they actually did explain — with the
                    sentence that earned each tick, so the score is auditable rather than asserted. */}
                <section className="summary-panel mastery-panel"><h2>CLASS GOALS</h2><ObjectiveChecklist objectives={classChecklist(unit)} covered={progress.covered_objectives} evidence={progress.objective_evidence} /></section>
                <section className="summary-panel statistics-panel">
                  <h2>TEACHING STATISTICS</h2>
                  <dl>
                    <Stat label="TEACHING TURNS" hint={STAT_HINTS.turns} value={progress.turn_count} />
                    <Stat label="QUESTIONS RECORDED" hint={STAT_HINTS.questions} value={snapshot?.questions.length ?? 0} />
                    <Stat
                      label="GAPS DISCOVERED"
                      hint={gapCount === null ? "FOUND BY THE TRANSFER ANALYSIS, WHICH IS STILL RUNNING." : STAT_HINTS.gaps}
                      value={gapCount ?? "—"}
                    />
                  </dl>
                </section>
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
