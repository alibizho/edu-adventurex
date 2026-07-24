import { useEffect, useState } from "react";
import { PixelRobot } from "../../../components/visuals/PixelRobot";
import type { StudyModule } from "../study.types";

type ClassroomStudentId = "left" | "right";

type TeachingLobbyProps = {
  studyModule: StudyModule;
  onReturnToReading: () => void;
  onStartConversation: () => void;
};

type PixelIconProps = {
  className?: string;
};

const ZOOM_DURATION = 520;

const STUDENTS: readonly {
  id: ClassroomStudentId;
  label: string;
}[] = [
  { id: "left", label: "Start a conversation with the left student" },
  { id: "right", label: "Start a conversation with the right student" },
];

function PixelDocumentIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M13 4h28l11 11v45H13V4zm7 7v42h25V20H36v-9H20zm23 2 4 4h-4v-4z" fill="currentColor" />
      <path d="M25 25h15v5H25zm0 10h15v5H25zm0 10h11v5H25z" fill="var(--paper)" />
    </svg>
  );
}

function PixelReturnIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M28 6 4 25l24 19V32h16v13H27v11h29V24H28V6z" fill="currentColor" />
    </svg>
  );
}

function PixelHandIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M8 28h13V16h8v8h7v5h7v5h7v17h-7v7H21v-6h-7v-7H8V28zm13 8h-6v5h6v5h8v5h14V36h-7v7h-7V24h-2v19h-6V36z" fill="currentColor" />
    </svg>
  );
}

function PixelQuestionIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M7 4h39v6h6v27H31l-9 10v-10H7V4zm7 7v19h13v6l6-6h12V11H14z" fill="currentColor" />
      <path d="M26 14h9v4h5v8h-5v5h-7v-8h5v-3h-7v-6zm2 20h7v6h-7zM22 46h12v5h6v9H16v-9h6z" fill="currentColor" />
    </svg>
  );
}

function PixelSearchIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M10 7h29v5h7v7h5v21h-5v6h-7v5H18v-5h-7v-7H6V18h4V7zm8 7v5h-5v19h5v6h19v-5h7V20h-7v-6H18zm-2 34h8v7h-7v5H7v-8h9v-4z" fill="currentColor" />
    </svg>
  );
}

function PixelBookIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M4 9h23l5 5 5-5h23v42H39l-7 5-7-5H4V9zm7 7v28h16v-4H14v-5h13v-4H14v-5h13V16H11zm26 0v10h13v5H37v4h13v5H37v4h16V16H37z" fill="currentColor" />
    </svg>
  );
}

function PixelLightbulbIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M22 4h20v5h7v7h5v20h-5v7h-6v9H21v-9h-6v-7h-5V16h5V9h7V4zm2 8h-5v7h-4v14h5v6h7v8h10v-8h7v-6h5V19h-5v-7H24zm2 38h12v6H26zm3 8h6v4h-6z" fill="currentColor" />
    </svg>
  );
}

function PixelArrowIcon({ direction, className }: PixelIconProps & { direction: "left" | "right" }) {
  return (
    <svg className={className} viewBox="0 0 32 32" aria-hidden="true" shapeRendering="crispEdges">
      <path d={direction === "left" ? "M28 12H13V6L3 16l10 10v-6h15v-8z" : "M4 12h15V6l10 10-10 10v-6H4v-8z"} fill="currentColor" />
    </svg>
  );
}

function PixelDownIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 32 32" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M5 8h22L16 26 5 8z" fill="currentColor" />
    </svg>
  );
}

function SessionProgress() {
  return (
    <div className="lobby-progress" role="progressbar" aria-label="Session progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={40}>
      <div className="lobby-progress-blocks" aria-hidden="true">
        {Array.from({ length: 10 }, (_, index) => <i key={index} className={index < 4 ? "is-filled" : ""} />)}
      </div>
      <strong>40%</strong>
    </div>
  );
}

function TutorialContent({ disabled, onCollapse }: { disabled: boolean; onCollapse: () => void }) {
  return (
    <div className="tutorial-content" id="teaching-tutorial-content">
      <button
        type="button"
        className="tutorial-disclosure tutorial-disclosure--collapse"
        aria-label="Collapse tutorial"
        aria-expanded="true"
        aria-controls="teaching-tutorial-content"
        disabled={disabled}
        onClick={onCollapse}
      >
        <PixelDownIcon />
      </button>

      <section className="tutorial-select">
        <div className="tutorial-heading-row">
          <div className="tutorial-bot"><PixelRobot /></div>
          <div>
            <h3>SELECT A STUDENT</h3>
            <p>CLICK THE ? ABOVE<br />THEIR HEAD TO START<br />A CONVERSATION</p>
          </div>
        </div>
        <div className="tutorial-ready"><PixelHandIcon /><span>WUT IS READY<br />TO LEARN!</span></div>
      </section>

      <section className="tutorial-how">
        <h3><span />HOW IT WORKS<span /></h3>
        <div className="tutorial-steps">
          <div><PixelQuestionIcon /><span>WUT ASKS<br />QUESTIONS</span></div>
          <PixelArrowIcon direction="right" className="tutorial-step-arrow" />
          <div><PixelSearchIcon /><span>WE FIND<br />THE GAPS</span></div>
          <PixelArrowIcon direction="right" className="tutorial-step-arrow" />
          <div><PixelBookIcon /><span>YOU TEACH<br />BETTER</span></div>
          <PixelArrowIcon direction="right" className="tutorial-step-arrow" />
          <div><PixelRobot /><span>WUT<br />UNDERSTANDS</span></div>
        </div>
      </section>

      <section className="tutorial-tips">
        <h3><PixelLightbulbIcon />TIPS FOR TEACHERS</h3>
        <ul>
          <li>USE SIMPLE WORDS</li>
          <li>GIVE REAL EXAMPLES</li>
          <li>DON&apos;T WORRY IF YOU&apos;RE NOT PERFECT</li>
          <li>TAKE YOUR TIME!</li>
        </ul>
      </section>
    </div>
  );
}

export function TeachingLobby({ studyModule, onReturnToReading, onStartConversation }: TeachingLobbyProps) {
  const [selectedStudent, setSelectedStudent] = useState<ClassroomStudentId | null>(null);
  const [isTutorialOpen, setIsTutorialOpen] = useState(true);

  useEffect(() => {
    if (!selectedStudent) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reducedMotion) {
      onStartConversation();
      return;
    }

    const timer = window.setTimeout(onStartConversation, ZOOM_DURATION);
    return () => window.clearTimeout(timer);
  }, [selectedStudent, onStartConversation]);

  function selectStudent(studentId: ClassroomStudentId) {
    if (selectedStudent) return;
    setSelectedStudent(studentId);
  }

  return (
    <main className={`teaching-lobby ${isTutorialOpen ? "is-tutorial-open" : "is-tutorial-collapsed"}`}>
      <section className="lobby-summary" aria-label="Teaching session overview">
        <div className="lobby-summary-card lobby-module-card">
          <PixelDocumentIcon className="lobby-module-icon" />
          <div>
            <span>CURRENT MODULE</span>
            <h1>{studyModule.title}</h1>
            <p>TOPIC: {studyModule.document.title}</p>
          </div>
        </div>

        <div className="lobby-summary-card lobby-progress-card">
          <span>SESSION PROGRESS</span>
          <SessionProgress />
        </div>

        <button
          type="button"
          className="lobby-summary-card lobby-back-card"
          disabled={Boolean(selectedStudent)}
          onClick={onReturnToReading}
        >
          <PixelReturnIcon className="lobby-return-icon" />
          <span>BACK TO<br />LEARNING MATERIAL</span>
        </button>
      </section>

      <section className="classroom-viewport" aria-label="Select a student to teach">
        <div className={`classroom-scene${selectedStudent ? ` is-zooming focus-${selectedStudent}` : ""}`}>
          <img src="/images/wut-classroom.png" alt="A pixel-art classroom with students waiting to learn" />
          <div className="classroom-board-copy" aria-hidden="true">
            <span>TEACH WUT</span>
            <span>HELP WUT UNDERSTAND</span>
            <span>DISCOVER YOUR GAPS</span>
          </div>
          {STUDENTS.map((student) => (
            <button
              key={student.id}
              type="button"
              className={`student-question student-question--${student.id}`}
              aria-label={student.label}
              disabled={Boolean(selectedStudent)}
              onClick={() => selectStudent(student.id)}
            >
              ?
            </button>
          ))}
        </div>
      </section>

      <section className={`tutorial-drawer${isTutorialOpen ? " is-open" : " is-collapsed"}`} aria-label="Teaching tutorial">
        {isTutorialOpen ? (
          <TutorialContent disabled={Boolean(selectedStudent)} onCollapse={() => setIsTutorialOpen(false)} />
        ) : (
          <div className="tutorial-bar">
            <span>TUTORIAL BAR</span>
            <button
              type="button"
              className="tutorial-disclosure tutorial-disclosure--expand"
              aria-label="Expand tutorial"
              aria-expanded="false"
              aria-controls="teaching-tutorial-content"
              disabled={Boolean(selectedStudent)}
              onClick={() => setIsTutorialOpen(true)}
            >
              <PixelArrowIcon direction="left" />
            </button>
          </div>
        )}
      </section>

      <p className="sr-only" role="status" aria-live="polite">
        {selectedStudent ? "ENTERING CONVERSATION..." : ""}
      </p>
    </main>
  );
}
