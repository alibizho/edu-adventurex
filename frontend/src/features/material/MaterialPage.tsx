import { useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { GeneratingTopics } from "./components/GeneratingTopics";
import { MaterialEntryPanel } from "./components/MaterialEntryPanel";
import { MATERIAL_PAGE_CONTENT, SURPRISE_TOPICS } from "./material.data";
import type { MaterialInputStatus } from "./material.types";
import type { GrowthPath } from "../learning-data/backend.types";
import { useMaterialInput } from "./useMaterialInput";

const STATUS_LABELS: Record<MaterialInputStatus, string> = {
  idle: "READY TO LEARN",
  ready: "MATERIAL READY",
  error: "INPUT ERROR",
  processing: "PREPARING MATERIALS",
};

export function MaterialPage() {
  const navigate = useNavigate();
  const surpriseIndex = useRef(0);
  const handlePrepared = useCallback((path: GrowthPath) => {
    localStorage.setItem("wut:active-path", path.path_id);
    navigate(`${ROUTES.concepts}?path=${encodeURIComponent(path.path_id)}`);
  }, [navigate]);
  const materialInput = useMaterialInput({ onPrepared: handlePrepared });

  function handleShortcut(id: (typeof MATERIAL_PAGE_CONTENT.shortcuts)[number]["id"], prompt?: string) {
    if (id === "surprise") {
      materialInput.updateText(SURPRISE_TOPICS[surpriseIndex.current % SURPRISE_TOPICS.length]);
      surpriseIndex.current += 1;
      return;
    }

    if (prompt) materialInput.updateText(prompt);
  }

  // A build takes most of a minute (a model call per class), so it takes over the screen rather
  // than leaving the form sitting there disabled with no sign of what is happening.
  if (materialInput.pipeline.length > 0) {
    return (
      <div className="screen material-screen">
        <AppHeader />
        <main className="material-main">
          <GeneratingTopics lines={materialInput.pipeline} />
        </main>
        <StatusBar label="BUILDING YOUR COURSE" />
      </div>
    );
  }

  return (
    <div className="screen material-screen">
      <AppHeader />
      <main className="material-main">
        <h1 className="boxed-title">{MATERIAL_PAGE_CONTENT.title}</h1>

        <MaterialEntryPanel
          content={MATERIAL_PAGE_CONTENT.entry}
          text={materialInput.text}
          files={materialInput.files}
          error={materialInput.error}
          isProcessing={materialInput.isProcessing}
          canSubmit={materialInput.canSubmit}
          onTextChange={materialInput.updateText}
          onFilesSelected={materialInput.addFiles}
          onFileRemove={materialInput.removeFile}
          onSubmit={materialInput.submit}
        />

        {materialInput.warnings.length > 0 && (
          <div className="material-api-notice" role="status">
            {materialInput.warnings.map((warning) => <p key={warning}>{warning}</p>)}
          </div>
        )}
        {materialInput.scopeSuggestions.length > 0 && (
          <section className="material-scope retro-panel" aria-labelledby="scope-title">
            <h2 id="scope-title">CHOOSE A FOCUSED LEARNING PATH</h2>
            <div>
              {materialInput.scopeSuggestions.map((suggestion) => (
                <button
                  type="button"
                  key={suggestion.topic}
                  disabled={materialInput.isProcessing}
                  onClick={() => materialInput.buildConfirmedTopic(suggestion.topic, suggestion.suggested_classes)}
                >
                  <strong>{suggestion.topic}</strong>
                  <span>{suggestion.rationale}</span>
                </button>
              ))}
            </div>
          </section>
        )}
        <div className="material-shortcuts" aria-label="Suggested learning topics">
          {MATERIAL_PAGE_CONTENT.shortcuts.map((shortcut, index) => (
            <button
              key={shortcut.id}
              type="button"
              className={`retro-tab ${index === 0 ? "is-active" : ""}`}
              disabled={materialInput.isProcessing}
              onClick={() => handleShortcut(shortcut.id, shortcut.prompt)}
            >
              {shortcut.label}
            </button>
          ))}
        </div>
      </main>
      <StatusBar label={STATUS_LABELS[materialInput.status]} />
    </div>
  );
}
