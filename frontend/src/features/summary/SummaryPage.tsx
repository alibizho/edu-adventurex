import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { useSession } from "../../app/SessionProvider";
import type { ConceptMasteryIcon, SessionSummary } from "../../app/session.types";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { CONCEPTS } from "../concepts/concepts.data";
import type { ConceptId } from "../concepts/concepts.types";
import { StudentSidebar } from "../study/components/StudentSidebar";
import { isConceptId } from "../study/study.data";
import type { StudyToolConfig, StudyToolId } from "../study/study.types";
import { BackendSummaryPage } from "./BackendSummaryPage";

const SUMMARY_TOOLS: readonly StudyToolConfig[] = [
  { id: "progress", label: "Progress" },
  { id: "tutorial", label: "Tutorial" },
  { id: "reset", label: "Reset" },
];

function formatDuration(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  return [hours, minutes, remainingSeconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function MasteryIcon({ icon }: { icon: ConceptMasteryIcon }) {
  return <span className={`mastery-icon mastery-icon--${icon}`} aria-hidden="true"><i /></span>;
}

function LegacySummaryPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { state, getSessionSummary, startReview } = useSession();
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [toolMessage, setToolMessage] = useState<string | null>(null);

  const conceptId = useMemo<ConceptId | null>(() => {
    const queryConcept = searchParams.get("concept");
    if (isConceptId(queryConcept)) return queryConcept;
    const recent = CONCEPTS
      .map(({ id }) => state.concepts[id])
      .filter(({ latestSummary }) => Boolean(latestSummary))
      .sort((a, b) => (b.latestSummary?.completedAt ?? "").localeCompare(a.latestSummary?.completedAt ?? ""))[0];
    return recent?.conceptId ?? null;
  }, [searchParams, state.concepts]);

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setLoadError(null);
    if (!conceptId) {
      setSummary(null);
      setIsLoading(false);
      return () => { active = false; };
    }

    getSessionSummary(conceptId)
      .then((result) => { if (active) setSummary(result); })
      .catch(() => { if (active) setLoadError("SUMMARY DATA COULD NOT BE LOADED."); })
      .finally(() => { if (active) setIsLoading(false); });
    return () => { active = false; };
  }, [conceptId, getSessionSummary]);

  function handleToolAction(toolId: StudyToolId) {
    if (toolId === "progress") {
      navigate(conceptId ? `${ROUTES.progress}?concept=${conceptId}` : ROUTES.progress);
      return;
    }
    if (toolId === "tutorial") {
      setToolMessage("TIP: REVIEW YOUR MASTERY SCORES, THEN CONTINUE TO THE KNOWLEDGE MAP.");
      return;
    }
    if (!conceptId) {
      setToolMessage("NO COMPLETED SESSION TO REVIEW.");
      return;
    }
    startReview(conceptId);
    navigate(`${ROUTES.study}?concept=${conceptId}`);
  }

  const statusLabel = isLoading ? "LOADING_SUMMARY" : loadError ? "SUMMARY_ERROR" : summary ? "SESSION_MASTERED" : "NO_SESSION_DATA";

  return (
    <div className="screen summary-screen">
      <AppHeader />
      <main className="summary-layout">
        <StudentSidebar
          student={{ name: "AI STUDENT", readiness: summary?.readiness ?? 0 }}
          tools={SUMMARY_TOOLS}
          message={toolMessage}
          onToolAction={handleToolAction}
          avatarVariant="student"
        />

        <section className="summary-canvas">
          {isLoading ? (
            <div className="summary-empty retro-panel" role="status">LOADING SESSION SUMMARY...</div>
          ) : !summary || loadError ? (
            <div className="summary-empty retro-panel">
              <h1>NO COMPLETED SESSION</h1>
              <p>{loadError ?? "FINISH A TEACHING CONVERSATION TO CREATE YOUR FIRST SUMMARY."}</p>
              <button type="button" className="solid-action" onClick={() => navigate(ROUTES.concepts)}>START LEARNING</button>
            </div>
          ) : (
            <div className="summary-dashboard">
              <h1>{summary.moduleTitle}</h1>

              <div className="summary-top-grid">
                <section className="summary-panel growth-panel" aria-labelledby="growth-title">
                  <h2 id="growth-title"><i />GROWTH TIMELINE</h2>
                  <div className="growth-track">
                    {summary.milestones.map((milestone) => (
                      <div key={milestone.id} className={`growth-stop growth-stop--${milestone.status}`}>
                        {milestone.status === "completed" && <span>COMPLETED</span>}
                        <b aria-hidden="true">{milestone.status === "completed" ? "✓" : milestone.status === "next" ? "■" : "○"}</b>
                        <strong>{milestone.label}</strong>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="master-panel" aria-labelledby="master-title">
                  <h2 id="master-title">★ MASTER STATUS</h2>
                  <p>“{summary.masterTitle}: {summary.masterQuote}”</p>
                  <strong>RANK: {summary.rank}</strong>
                </section>
              </div>

              <div className="summary-bottom-grid">
                <section className="summary-panel mastery-panel" aria-labelledby="mastery-title">
                  <h2 id="mastery-title">CONCEPTS MASTERED</h2>
                  <div className="mastery-grid">
                    {summary.mastery.map((metric) => (
                      <div className="mastery-item" key={metric.id}>
                        <MasteryIcon icon={metric.icon} />
                        <div>
                          <strong>{metric.label}</strong>
                          <span role="progressbar" aria-label={`${metric.label} mastery`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={metric.score}>
                            <i style={{ width: `${metric.score}%` }} />
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="summary-panel statistics-panel" aria-labelledby="statistics-title">
                  <h2 id="statistics-title">TEACHING STATISTICS</h2>
                  <dl>
                    <div><dt>STUDENTS TAUGHT</dt><dd>{summary.studentsTaught}</dd></div>
                    <div><dt>QUESTIONS ANSWERED</dt><dd>{summary.questionsAnswered}</dd></div>
                    <div><dt>GAPS DISCOVERED</dt><dd>{summary.gapsDiscovered}</dd></div>
                  </dl>
                </section>
              </div>
            </div>
          )}
        </section>
      </main>
      <StatusBar label={statusLabel} full meta={summary ? `${formatDuration(summary.durationSeconds)} SESSION TIME` : ""} />
    </div>
  );
}

export function SummaryPage() {
  const [searchParams] = useSearchParams();
  const pathId = searchParams.get("path");
  const classId = searchParams.get("class");
  if (pathId && classId) return <BackendSummaryPage pathId={pathId} classId={classId} />;
  return <LegacySummaryPage />;
}
