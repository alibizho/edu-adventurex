import { useEffect } from "react";
import { PixelReturnIcon } from "../../../components/visuals/PixelIcons";
import type { StudyModule } from "../study.types";

const EXIT_DURATION_MS = 3600;

type SessionExitSceneProps = {
  studyModule: StudyModule;
  onComplete: () => void;
};

function PixelDocumentIcon() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M13 4h28l11 11v45H13V4zm7 7v42h25V20H36v-9H20zm23 2 4 4h-4v-4z" fill="currentColor" />
      <path d="M25 25h15v5H25zm0 10h15v5H25zm0 10h11v5H25z" fill="var(--paper)" />
    </svg>

  );
}

function PixelWalker({ index }: { index: number }) {
  return (
    <div className={`pixel-walker pixel-walker--${index + 1}`} aria-hidden="true">
      <svg viewBox="0 0 44 92" shapeRendering="crispEdges">
        <circle cx="22" cy="12" r="10" />
        <path d="M14 23h16v28H14z" />
        <g className="walker-pose walker-pose--a">
          <path d="M14 27 4 46l6 3 10-17zM30 27l10 18-6 4-10-17zM15 50 9 83h8l8-31zM27 50l1 34h8l-1-35z" />
        </g>

        <g className="walker-pose walker-pose--b">
          <path d="M14 27 8 47l7 2 6-18zM30 27l7 21 7-3-9-19zM16 49l1 35h8l-1-34zM28 51 18 82h8l10-28z" />
        </g>

      </svg>

    </div>

  );
}

function ExitClassroom() {
  return (
    <section className="exit-classroom" aria-label="Classroom closing animation" aria-busy="true">
      <div className="exit-clock" aria-hidden="true"><i /></div>

      <div className="exit-plant" aria-hidden="true"><i /><b /><b /><b /><span /></div>

      <div className="exit-blackboard">
        <p>CONGRATULATIONS ON<br />FINISHING THE SESSION!<br />CLASS IS OVER~</p>

        <span aria-hidden="true">♡</span>

      </div>

      <div className="exit-window" aria-hidden="true"><i /><i /><i /><i /></div>

      <div className="exit-room-line" aria-hidden="true" />
      <div className="exit-desk exit-desk--left" aria-hidden="true"><i /></div>

      <div className="exit-desk exit-desk--middle" aria-hidden="true"><i /></div>

      <div className="exit-desk exit-desk--right" aria-hidden="true"><i /></div>

      <svg className="exit-podium" viewBox="0 0 370 110" aria-hidden="true" shapeRendering="crispEdges">
        <path d="M50 5h270l45 32H5z" />
        <path d="M24 37h322v68H24z" />
      </svg>

      {Array.from({ length: 4 }, (_, index) => <PixelWalker key={index} index={index} />)}

    </section>

  );
}

export function SessionExitScene({ studyModule, onComplete }: SessionExitSceneProps) {
  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const timer = window.setTimeout(onComplete, reducedMotion ? 0 : EXIT_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [onComplete]);

  return (
    <main className="session-exit" aria-label="Teaching session complete">
      <section className="lobby-summary" aria-label="Completed teaching session overview">
        <div className="lobby-summary-card lobby-module-card">
          <span className="lobby-module-icon"><PixelDocumentIcon /></span>

          <div>
            <span>CURRENT MODULE</span>

            <h1>{studyModule.title}</h1>

            <p>TOPIC: {studyModule.document.title}</p>

          </div>

        </div>

        <div className="lobby-summary-card lobby-progress-card">
          <span>SESSION PROGRESS</span>

          <div className="lobby-progress" role="progressbar" aria-label="Session progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={100}>
            <div className="lobby-progress-blocks" aria-hidden="true">
              {Array.from({ length: 10 }, (_, index) => <i key={index} className="is-filled" />)}
            </div>

            <strong>100%</strong>

          </div>

        </div>

        <button type="button" className="lobby-summary-card lobby-back-card" disabled>
          <span className="lobby-return-icon"><PixelReturnIcon /></span>

          <span>BACK TO<br />LEARNING MATERIAL</span>

        </button>

      </section>

      <ExitClassroom />

      <div className="exit-tutorial-bar" aria-hidden="true">
        <span>TUTORIAL BAR</span><i>◀</i>
      </div>

      <p className="sr-only" role="status" aria-live="polite">
        SESSION COMPLETE. STUDENTS ARE LEAVING THE CLASSROOM.
      </p>

    </main>

  );
}
