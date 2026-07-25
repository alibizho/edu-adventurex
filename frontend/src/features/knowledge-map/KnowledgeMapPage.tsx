import { useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { useBackendLearningPaths } from "../learning-data/useBackendLearningPaths";

// Nodes used to be absolutely positioned from a six-slot percent table. That never centred: the
// slots were hand-picked and right-biased, a seventh item landed exactly on top of the first, and
// left/top anchor a node's top-left corner rather than its middle. The container centres its
// children now, which is correct for any number of classes.

export function KnowledgeMapPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { entries, isLoading, error, reload } = useBackendLearningPaths();
  const selectedPathId = searchParams.get("path");
  const selected = useMemo(() => entries.find(({ path }) => path.path_id === selectedPathId) ?? null, [entries, selectedPathId]);

  return (
    <div className="screen knowledge-map-screen">
      <AppHeader />
      <main className={`knowledge-map-canvas${selected ? " is-domain-open" : ""}`}>
        <h1>{selected ? selected.path.confirmed_topic.toUpperCase() : "MY KNOWLEDGE MAP"}</h1>
        {isLoading ? (
          <div className="map-message retro-panel" role="status">LOADING KNOWLEDGE MAP...</div>
        ) : error ? (
          <div className="map-message retro-panel" role="alert">{error}<br /><button className="outline-action" onClick={() => void reload()}>RETRY</button></div>
        ) : selected ? (
          <section className="domain-concept-map" aria-label={`${selected.path.confirmed_topic} classes`}>
            <button type="button" className="map-back" onClick={() => setSearchParams({})}>← BACK TO OVERALL MAP</button>
            {selected.path.classes.map((unit) => {
              const progress = selected.memory.class_progress[unit.class_id];
              const status = progress?.status ?? "not_started";
              return (
                <button
                  type="button"
                  key={unit.class_id}
                  className={`map-node concept-map-node map-node--${status}`}
                  // Always the class itself, finished or not: the map is the way in, and what you
                  // want on arriving is the material. A completed class used to open its summary
                  // instead, so the one route to re-teaching something was to not have learnt it.
                  onClick={() => navigate(`${ROUTES.study}?path=${encodeURIComponent(selected.path.path_id)}&class=${encodeURIComponent(unit.class_id)}`)}
                >
                  <i /><strong>{unit.title}</strong><span>{progress?.readiness ?? 0}%</span>
                </button>
              );
            })}
          </section>
        ) : (
          <section className="domain-map backend-domain-map" aria-label="Learning topics">
            {entries.map(({ path, memory }) => {
              const progresses = path.classes.map(({ class_id }) => memory.class_progress[class_id]);
              const completed = progresses.filter((item) => item?.status === "complete").length;
              const readiness = progresses.length ? Math.round(progresses.reduce((sum, item) => sum + (item?.readiness ?? 0), 0) / progresses.length) : 0;
              const status = completed === path.classes.length && completed > 0 ? "complete" : readiness > 0 ? "in_progress" : "not_started";
              return (
                <button type="button" key={path.path_id} className={`map-node domain-node backend-domain-node map-node--${status}`} onClick={() => setSearchParams({ path: path.path_id })}>
                  <i /><strong>{path.confirmed_topic}</strong><span>{completed}/{path.classes.length} · {readiness}%</span>
                </button>
              );
            })}
            {entries.length === 0 && <div className="map-message retro-panel">NO LEARNING PATHS YET.</div>}
          </section>
        )}
      </main>
      <StatusBar label={selected ? "PATH_SELECTED" : "WAITING_FOR_INPUT"} />
    </div>
  );
}
