import type { ClassObjective } from "../../learning-data/backend.types";

type ObjectiveChecklistProps = {
  objectives: readonly ClassObjective[];
  covered: readonly string[];
  evidence: Record<string, string>;
};

/**
 * What the class is actually for: the goals, and which the learner has explained well enough to
 * convince someone who didn't already know.
 *
 * The quote under a ticked goal is the point — it keeps the checkmark accountable by showing the
 * sentence that earned it, rather than asking anyone to trust a score. Shown live in the classroom
 * and again on the summary, which is why it lives here instead of in either of them.
 */
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
