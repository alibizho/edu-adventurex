import type { CSSProperties, KeyboardEvent } from "react";
import { Square } from "lucide-react";
import type { ConceptNodeConfig } from "../concepts.types";

type SelectableConceptNode = Omit<ConceptNodeConfig, "id"> & { id: string };

type ConceptMapProps = {
  concepts: readonly SelectableConceptNode[];
  selectedConceptId: string | null;
  onSelect: (conceptId: string) => void;
};

export function ConceptMap({ concepts, selectedConceptId, onSelect }: ConceptMapProps) {
  function handleNodeKeyDown(event: KeyboardEvent<HTMLButtonElement>, conceptId: string) {
    if (event.key !== "Enter" && event.key !== " ") return;

    event.preventDefault();
    onSelect(conceptId);
  }

  // The node the rest of the course hangs off. Generated paths put it in the middle of the stage;
  // if a path is too short to have one, the first concept stands in so the map is never a
  // scattering of unconnected boxes.
  const hub = concepts.find((concept) => concept.shape.includes("hero")) ?? concepts[0] ?? null;

  return (
    <div className="concept-stage" aria-label="Concept selection map">
      {/* Spokes from the hub, drawn under the nodes: every box is one topic of the subject in the
          middle, and until this was here the map read as eight unrelated things. Centre to centre
          on purpose — the nodes are opaque, so each line emerges from one edge and disappears
          under the other, without having to solve for where a circle's boundary is. */}
      {hub && concepts.length > 1 && (
        <svg className="concept-links" aria-hidden="true" focusable="false">
          {concepts.filter((concept) => concept.id !== hub.id).map((concept) => (
            <line
              key={concept.id}
              className={selectedConceptId === concept.id ? "is-active" : undefined}
              x1={`${hub.x}%`}
              y1={`${hub.y}%`}
              x2={`${concept.x}%`}
              y2={`${concept.y}%`}
            />
          ))}
        </svg>
      )}

      {concepts.map((concept) => {
        const isSelected = selectedConceptId === concept.id;

        return (
          <button
            key={concept.id}
            type="button"
            className={`concept-node ${concept.shape} ${isSelected ? "is-selected" : ""}`}
            style={{ "--x": `${concept.x}%`, "--y": `${concept.y}%` } as CSSProperties}
            aria-label={concept.label.replaceAll("\n", " ")}
            aria-pressed={isSelected}
            data-concept-id={concept.id}
            onClick={() => onSelect(concept.id)}
            onKeyDown={(event) => handleNodeKeyDown(event, concept.id)}
          >
            {concept.icon === "solid" && <i className="node-solid" />}
            {concept.icon === "outline" && <Square size={26} strokeWidth={4} aria-hidden="true" />}
            {concept.icon === "grid" && <i className="node-grid"><b /><b /><b /><b /></i>}
            {/* One text node, with the line breaks left to `white-space: pre-line`. Splitting the
                label into a span per line put block boxes inside the label, and line-clamping
                cannot count lines through those — which is how a generated class title ended up
                spilling out through the top and bottom of its box. */}
            <span className="node-label" title={concept.label.replaceAll("\n", " ")}>{concept.label}</span>
          </button>
        );
      })}
    </div>
  );
}
