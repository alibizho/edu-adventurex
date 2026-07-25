import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { ConceptMap } from "./components/ConceptMap";
import { CONCEPTS } from "./concepts.data";
import type { ConceptNodeConfig } from "./concepts.types";
import { apiMessage } from "../learning-data/apiClient";
import { backendLearningDataSource } from "../learning-data/backendLearningDataSource";
import type { GrowthPath } from "../learning-data/backend.types";

const DYNAMIC_NODE_LAYOUT: ReadonlyArray<Pick<ConceptNodeConfig, "shape" | "x" | "y" | "icon">> = [
  { shape: "square", x: 22, y: 32, icon: "solid" },
  { shape: "square small", x: 48, y: 20 },
  { shape: "wide", x: 72, y: 29, icon: "outline" },
  { shape: "circle hero", x: 47, y: 57 },
  { shape: "wide", x: 76, y: 63 },
  { shape: "square", x: 20, y: 78, icon: "grid" },
  { shape: "circle small-circle", x: 38, y: 80 },
  { shape: "circle medium-circle", x: 66, y: 82 },
];

export function ConceptPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedPathId = searchParams.get("path") ?? localStorage.getItem("wut:active-path");
  const [path, setPath] = useState<GrowthPath | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(Boolean(requestedPathId));
  const [selectedConceptId, setSelectedConceptId] = useState<string | null>(null);

  useEffect(() => {
    if (!requestedPathId) return;
    let active = true;
    setIsLoading(true);
    setLoadError(null);
    backendLearningDataSource.getPath(requestedPathId)
      .then((result) => {
        if (!active) return;
        setPath(result);
        localStorage.setItem("wut:active-path", result.path_id);
      })
      .catch((error) => { if (active) setLoadError(apiMessage(error)); })
      .finally(() => { if (active) setIsLoading(false); });
    return () => { active = false; };
  }, [requestedPathId]);

  const concepts = useMemo(() => path
    ? path.classes.map((unit, index) => ({
        id: unit.class_id,
        label: unit.title.toUpperCase(),
        ...DYNAMIC_NODE_LAYOUT[index % DYNAMIC_NODE_LAYOUT.length],
      }))
    : CONCEPTS,
  [path]);
  const selectedConcept = concepts.find(({ id }) => id === selectedConceptId) ?? null;

  function handleSelect(conceptId: string) {
    setSelectedConceptId((currentId) => currentId === conceptId ? null : conceptId);
  }

  function handleConfirm() {
    if (!selectedConcept) return;

    if (path) {
      localStorage.setItem("wut:active-class", selectedConcept.id);
      navigate(`${ROUTES.study}?path=${encodeURIComponent(path.path_id)}&class=${encodeURIComponent(selectedConcept.id)}`);
      return;
    }
    navigate(`${ROUTES.study}?concept=${encodeURIComponent(selectedConcept.id)}`);
  }

  return (
    <div className="screen concept-screen">
      <AppHeader />
      <main className="concept-main halftone-screen">
        <h1 className="boxed-title concept-title">SELECT A CONCEPT TO START TEACHING</h1>

        {isLoading ? (
          <div className="concept-api-state retro-panel" role="status">BUILDING YOUR LEARNING PATH...</div>

        ) : loadError ? (
          <div className="concept-api-state retro-panel" role="alert">
            <strong>LEARNING PATH UNAVAILABLE</strong>

            <p>{loadError}</p>

            <button type="button" className="solid-action" onClick={() => navigate(ROUTES.material)}>BACK HOME</button>

          </div>

        ) : (
          <ConceptMap
            concepts={concepts}
            selectedConceptId={selectedConceptId}
            onSelect={handleSelect}
          />

        )}
      </main>

      <div className="concept-footer">
        <StatusBar label={loadError ? "BACKEND_ERROR" : selectedConcept ? "SELECTION READY" : "WAITING FOR INPUT"} />

        <div className="concept-build">OCEAN SIMULATION: V1.0.4&nbsp; | &nbsp;DITHERING MODE: ACTIVE</div>

        <button
          type="button"
          className="solid-action confirm-action"
          disabled={!selectedConcept || isLoading || Boolean(loadError)}
          onClick={handleConfirm}
        >
          CONFIRM SELECTION
        </button>

      </div>

    </div>

  );
}
