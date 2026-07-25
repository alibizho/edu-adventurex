import { ArrowUpRight, CircleHelp, RotateCcw } from "lucide-react";
import { PixelRobot } from "../../../components/visuals/PixelRobot";
import type { StudentStatus, StudyToolConfig, StudyToolId } from "../study.types";

type StudentSidebarProps = {
  student: StudentStatus;
  tools: readonly StudyToolConfig[];
  message: string | null;
  onToolAction: (toolId: StudyToolId) => void;
  avatarVariant?: "robot" | "student";
};

function StudyToolIcon({ toolId }: { toolId: StudyToolId }) {
  if (toolId === "map") return <ArrowUpRight size={26} strokeWidth={3} aria-hidden="true" />;
  if (toolId === "tutorial") return <CircleHelp size={27} strokeWidth={3} aria-hidden="true" />;
  return <RotateCcw size={27} strokeWidth={3} aria-hidden="true" />;
}

export function StudentSidebar({
  student,
  tools,
  message,
  onToolAction,
  avatarVariant = "robot",
}: StudentSidebarProps) {
  const readiness = Math.min(100, Math.max(0, student.readiness));

  return (
    <aside className="student-sidebar">
      <div className="student-card">
        <div className={`student-avatar student-avatar--${avatarVariant}`}>
          {avatarVariant === "student" ? (
            <img src="/images/wut-student-fullbody.png" alt="" aria-hidden="true" />
          ) : (
            <PixelRobot />
          )}
        </div>

        <div>
          <strong>{student.name}</strong>

          <span>READINESS: {student.readiness}%</span>

        </div>

      </div>

      <div
        className="readiness-track"
        role="progressbar"
        aria-label={`${student.name} readiness`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={readiness}
      >
        <div className="readiness-fill" style={{ width: `${readiness}%` }} />

      </div>

      <nav className="side-nav" aria-label="Study tools">
        <div className="side-nav-actions">
          {tools.map((tool) => (
            <button key={tool.id} type="button" onClick={() => onToolAction(tool.id)}>
              <StudyToolIcon toolId={tool.id} />

              {tool.label}
            </button>

          ))}
        </div>

        <p className="study-tool-message" role="status" aria-live="polite">{message ?? ""}</p>

      </nav>

    </aside>

  );
}
