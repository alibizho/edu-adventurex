import type { ClassObjective } from "../../learning-data/backend.types";

type ObjectiveChecklistProps = {
  objectives: readonly ClassObjective[];
  covered: readonly string[];
  evidence: Record<string, string>;
};

export function ObjectiveChecklist({ objectives, covered, evidence }: ObjectiveChecklistProps) {
  return (
    <ul className="objective-list">
      {objectives.map((objective) => {
        const done = covered.includes(objective.id);
        return (
          <li key={objective.id} className={done ? "is-covered" : ""}>
            <span className="objective-box" aria-hidden="true">{done ? "✓" : ""}</span>

            <span className="objective-text">
              {objective.text}
              {done && evidence[objective.id] && (
                <em className="objective-evidence">“{evidence[objective.id]}”</em>

              )}
            </span>

          </li>

        );
      })}
    </ul>

  );
}
