import { Archive, Eye } from "lucide-react";
import type { StudyDocument } from "../study.types";

type LearningDocumentProps = {
  document: StudyDocument;
  isReady: boolean;
  onReady: () => void;
};

export function LearningDocument({ document, isReady, onReady }: LearningDocumentProps) {
  return (
    <article className="document-panel retro-panel">
      <header className="document-meta">
        <span>DOC_ID: {document.id}</span>

        <span>PAGE {document.page} OF {document.pageCount}</span>

      </header>

      <h2>{document.title}</h2>

      <p>{document.introduction}</p>

      <figure className="conceptual-figure">
        <div className="figure-placeholder">
          <Eye size={46} strokeWidth={2.5} />

          <i />
          <Archive size={46} strokeWidth={2.5} />

        </div>

        <figcaption>{document.figureCaption}</figcaption>

      </figure>

      <p>{document.detail}</p>

      <button
        type="button"
        className="solid-action ready-action"
        disabled={isReady}
        onClick={onReady}
      >
        {isReady ? document.completedLabel : document.readyLabel}
      </button>

    </article>

  );
}
