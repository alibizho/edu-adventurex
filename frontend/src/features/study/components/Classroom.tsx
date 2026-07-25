import { useEffect, useRef, useState, type CSSProperties } from "react";
import { PixelMicIcon, PixelReturnIcon } from "../../../components/visuals/PixelIcons";
import { PixelRobot } from "../../../components/visuals/PixelRobot";
import type { ClassObjective } from "../../learning-data/backend.types";
import { POSE_SOURCE, type ClassroomCast } from "../classroom.cast";
import { SEATS, SEAT_BY_ID, type SeatId } from "../classroom.seats";
import { formatElapsed, type MeterRef, type RecorderState } from "../usePressToTalkRecorder";
import type { StudyModule } from "../study.types";
import { ObjectiveChecklist } from "./ObjectiveChecklist";

type PixelIconProps = {
  className?: string;
};

const ZOOM_DURATION = 520;

type LiveSession = {
  raisedHands: readonly SeatId[];
  recorderState: RecorderState;
  meterRef: MeterRef;
  elapsedSeconds: number;
  queueDepth: number;
  isBusy: boolean;
  isResetting: boolean;
  lastHeard: string | null;
  explanation: string | null;
  turnCount: number;
  error: string | null;
  voiceAvailable: boolean;
  lastConfidence: number | null;
  onToggleMic: () => void;
  onDiscardRecording: () => void;
  onDismissExplanation: () => void;
  onTextAnswer: (text: string) => void;
  onReset: () => void;
  onFinish: () => void;
};

type ClassroomProps = {
  studyModule: StudyModule;
  readiness: number;
  cast: ClassroomCast;
  objectives?: readonly ClassObjective[];
  coveredObjectives?: readonly string[];
  objectiveEvidence?: Record<string, string>;
  onEnterZoom: (seatId: SeatId) => void;
  onReturnToReading: () => void;
  live?: LiveSession;
};

function PixelDocumentIcon({ className }: PixelIconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" shapeRendering="crispEdges">
      <path d="M13 4h28l11 11v45H13V4zm7 7v42h25V20H36v-9H20zm23 2 4 4h-4v-4z" fill="currentColor" />
      <path d="M25 25h15v5H25zm0 10h15v5H25zm0 10h11v5H25z" fill="var(--paper)" />
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

function SessionProgress({ readiness }: { readiness: number }) {
  const filled = Math.round((Math.min(100, Math.max(0, readiness)) / 100) * 10);
  return (
    <div className="lobby-progress" role="progressbar" aria-label="Session progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={readiness}>
      <div className="lobby-progress-blocks" aria-hidden="true">
        {Array.from({ length: 10 }, (_, index) => <i key={index} className={index < filled ? "is-filled" : ""} />)}

      </div>

      <strong>{readiness}%</strong>

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
            <h3>PRESS THE MIC AND TEACH</h3>

            <p>PRESS AGAIN TO SEND IT.<br />WHEN A STUDENT GETS LOST<br />A ? POPS UP — CLICK IT</p>

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

          <li>PRESS THE MIC AGAIN TO SEND</li>

          <li>TAKE YOUR TIME — NOTHING IS SENT UNTIL YOU SAY SO</li>

        </ul>

      </section>

    </div>

  );
}

function ClassroomTextInput({ isBusy, onSend }: { isBusy: boolean; onSend: (text: string) => void }) {
  const [draft, setDraft] = useState("");

  function send() {
    const trimmed = draft.trim();
    if (!trimmed || isBusy) return;
    setDraft("");
    onSend(trimmed);
  }

  return (
    <div className="backend-text-control classroom-text-control">
      <label htmlFor="classroom-answer">GPU VOICE ANALYSIS OFFLINE — TYPE WHAT YOU WOULD SAY</label>

      <textarea
        id="classroom-answer"
        value={draft}
        maxLength={2000}
        disabled={isBusy}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            send();
          }
        }}
      />

      <button type="button" className="solid-action" disabled={!draft.trim() || isBusy} onClick={send}>
        {isBusy ? "SENDING..." : "TEACH THIS"}
      </button>

    </div>

  );
}

function classStatus(
  live: LiveSession,
  openGoals: readonly ClassObjective[],
  allCovered: boolean,
): { text: string; tone: "clear" | "unsure" | "done" } {
  if (live.raisedHands.length > 0) {
    return { text: `${live.raisedHands.length} HAND${live.raisedHands.length > 1 ? "S" : ""} UP — CLICK THE ?`, tone: "unsure" };
  }
  if (live.lastConfidence !== null && live.lastConfidence < 0.5) {
    return { text: "THAT SOUNDED UNSURE — SOMEONE WILL ASK", tone: "unsure" };
  }
  if (allCovered) return { text: "THE CLASS HAS IT — YOU CAN FINISH", tone: "done" };
  if (openGoals.length > 0) {
    return { text: `FOLLOWING YOU · ${openGoals.length} GOAL${openGoals.length > 1 ? "S" : ""} LEFT: ${openGoals[0].text}`, tone: "clear" };
  }
  return { text: "THE CLASS IS FOLLOWING", tone: "clear" };
}

function micCaption(state: RecorderState, isBusy: boolean, queueDepth: number, elapsedSeconds: number) {
  if (state === "recording") return `RECORDING ${formatElapsed(elapsedSeconds)} — PRESS AGAIN TO SEND`;
  if (state === "starting") return "OPENING THE MIC...";
  if (queueDepth > 3) return `FALLING BEHIND — ${queueDepth} CLIPS QUEUED`;
  if (isBusy) return "THINKING...";
  return "PRESS THE MIC AND TEACH — PRESS AGAIN WHEN YOU'RE DONE";
}

export function Classroom({
  studyModule,
  readiness,
  cast,
  objectives = [],
  coveredObjectives = [],
  objectiveEvidence = {},
  onEnterZoom,
  onReturnToReading,
  live,
}: ClassroomProps) {
  const [zoomingTo, setZoomingTo] = useState<SeatId | null>(null);
  const [isTutorialOpen, setIsTutorialOpen] = useState(true);
  const [isConfirmingReset, setIsConfirmingReset] = useState(false);

  const onEnterZoomRef = useRef(onEnterZoom);
  onEnterZoomRef.current = onEnterZoom;

  useEffect(() => {
    if (!zoomingTo) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      onEnterZoomRef.current(zoomingTo);
      return;
    }
    const timer = window.setTimeout(() => onEnterZoomRef.current(zoomingTo), ZOOM_DURATION);
    return () => window.clearTimeout(timer);
  }, [zoomingTo]);

  const openGoals = objectives.filter((objective) => !coveredObjectives.includes(objective.id));
  const allCovered = objectives.length > 0 && openGoals.length === 0;
  const status = live ? classStatus(live, openGoals, allCovered) : null;
  const raised = live ? live.raisedHands : SEATS.map((seat) => seat.id);
  const isArmed = live ? live.recorderState !== "idle" : false;
  const focus = zoomingTo ? SEAT_BY_ID.get(zoomingTo) ?? null : null;

  return (
    <main className="teaching-lobby">
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
          <span>{objectives.length > 0 ? "CLASS GOALS" : "SESSION PROGRESS"}</span>

          <SessionProgress readiness={readiness} />

          {objectives.length > 0 && (
            <ObjectiveChecklist
              objectives={objectives}
              covered={coveredObjectives}
              evidence={objectiveEvidence}
            />

          )}
        </div>

        <button
          type="button"
          className="lobby-summary-card lobby-back-card"
          disabled={Boolean(zoomingTo)}
          onClick={onReturnToReading}
        >
          <PixelReturnIcon className="lobby-return-icon" />
          <span>BACK TO<br />LEARNING MATERIAL</span>

        </button>

      </section>

      {status && (
        <p className={`classroom-banner classroom-banner--${status.tone}`} role="status" aria-live="polite">
          {status.text}
        </p>

      )}

      <section className="classroom-viewport" aria-label="Your classroom">
        {}
        <div
          className={`classroom-scene${zoomingTo ? " is-zooming" : ""}`}
          style={focus ? { "--seat-x": focus.x / 100, "--seat-y": focus.base / 100 } as CSSProperties : undefined}
        >
          <img className="classroom-backdrop" src="/images/classroom.png" alt="" />
          {SEATS.map((seat) => {
            const hasQuestion = raised.includes(seat.id);
            const pose = hasQuestion ? "handup" : cast[seat.id].resting;
            return (
              <div
                key={seat.id}
                className={`classroom-seat classroom-seat--${pose}`}
                style={{ left: `${seat.x}%`, bottom: `${100 - seat.base}%` }}
              >
                <img src={POSE_SOURCE[pose]} alt="" aria-hidden="true" />
                {hasQuestion && (
                  <button
                    type="button"
                    className="student-question"
                    aria-label={`${seat.name} has a question — click to answer it`}
                    disabled={Boolean(zoomingTo)}
                    onClick={() => !zoomingTo && setZoomingTo(seat.id)}
                  >
                    ?
                  </button>

                )}
              </div>

            );
          })}
        </div>

      </section>

      {}
      {live?.explanation && (
        <section className="classroom-explanation" role="status" aria-live="polite">
          <div className="classroom-explanation-bot" aria-hidden="true"><PixelRobot /></div>

          <div className="classroom-explanation-body">
            <strong>YOU SAID YOU WEREN'T SURE — SO HERE IT IS</strong>
            <p>{live.explanation}</p>
          </div>
          <button type="button" className="outline-action" onClick={live.onDismissExplanation}>
            GOT IT
          </button>
        </section>
      )}

      {/* Outside .classroom-viewport on purpose: that box is a fixed aspect-ratio window with
          overflow:hidden, so anything after the scene gets clipped out of existence. */}
      {live && (live.voiceAvailable ? (
          <div className="classroom-controls">
            <button
              type="button"
              className={`voice-button${live.recorderState === "recording" ? " is-listening" : ""}${isArmed ? " is-armed" : ""}`}
              aria-label={live.recorderState === "recording" ? "Stop recording and send what you taught" : "Start recording"}
              aria-pressed={isArmed}
              onClick={live.onToggleMic}
            >
              <PixelMicIcon />
            </button>
            <div className="classroom-status">
              <strong>{micCaption(live.recorderState, live.isBusy, live.queueDepth, live.elapsedSeconds)}</strong>
              {isArmed && (
                <div className="mic-meter" aria-hidden="true">
                  <div className="mic-meter-fill" ref={live.meterRef} />
                </div>
              )}
              {live.lastHeard && <span className="classroom-heard">“{live.lastHeard}”</span>}
              {live.lastConfidence !== null && (
                <div className="clarity-row">
                  <span>CLARITY</span>
                  <span className="clarity-track" aria-hidden="true">
                    <i style={{ width: `${Math.round(live.lastConfidence * 100)}%` }} />
                  </span>
                  <span>{Math.round(live.lastConfidence * 100)}%</span>
                </div>
              )}
            </div>
            {live.recorderState === "recording" && (
              <button type="button" className="outline-action discard-take" onClick={live.onDiscardRecording}>
                DISCARD
              </button>
            )}
          </div>
        ) : (
          <ClassroomTextInput isBusy={live.isBusy} onSend={live.onTextAnswer} />
        ))}

      <section className={`tutorial-drawer${isTutorialOpen ? " is-open" : " is-collapsed"}`} aria-label="Teaching tutorial">
        {isTutorialOpen ? (
          <TutorialContent disabled={Boolean(zoomingTo)} onCollapse={() => setIsTutorialOpen(false)} />
        ) : (
          <div className="tutorial-bar">
            <span>TUTORIAL BAR</span>
            <button
              type="button"
              className="tutorial-disclosure tutorial-disclosure--expand"
              aria-label="Expand tutorial"
              aria-expanded="false"
              aria-controls="teaching-tutorial-content"
              disabled={Boolean(zoomingTo)}
              onClick={() => setIsTutorialOpen(true)}
            >
              <PixelArrowIcon direction="left" />
            </button>
          </div>
        )}
      </section>

      {/* Soft gate: finishing is always allowed. Covering everything changes what the button says
          and how loud it is, rather than being the only way out — a learner whose phrasing the
          judge missed must never be trapped in a class they already understand.

          Beside it, the other honest way out: a lesson that wandered off the topic can be thrown
          away and taught again, instead of being finished badly or carried by a transcript the
          learner no longer stands behind. */}
      {live && live.turnCount > 0 && (
        <div className="classroom-actions">
          <button
            type="button"
            className={`outline-action classroom-reset${isConfirmingReset ? " is-confirming" : ""}`}
            disabled={Boolean(zoomingTo) || live.isResetting}
            onClick={() => {
              if (!isConfirmingReset) return setIsConfirmingReset(true);
              setIsConfirmingReset(false);
              live.onReset();
            }}
          >
            {live.isResetting ? "STARTING OVER..."
              : isConfirmingReset ? "YES — WIPE THIS CLASS"
              : "START THIS CLASS OVER"}
          </button>
          {isConfirmingReset && !live.isResetting && (
            <button type="button" className="outline-action classroom-reset" onClick={() => setIsConfirmingReset(false)}>
              KEEP GOING
            </button>
          )}
          {/* Not while the mic is open: the recording in progress would be dropped unsent, and
              "finish" must never be how a teacher loses their last sentence. Reset stays live —
              throwing the recording away is exactly what it is for. */}
          <button
            type="button"
            className={`classroom-finish solid-action${allCovered ? " is-mastered" : ""}`}
            disabled={Boolean(zoomingTo) || live.isResetting || live.recorderState === "recording"}
            onClick={live.onFinish}
          >
            {allCovered ? "YOU'VE GOT IT — FINISH" : "FINISH TEACHING"}
          </button>
        </div>
      )}

      {live?.error && <p className="classroom-error" role="alert">{live.error}</p>}

      <p className="sr-only" role="status" aria-live="polite">
        {live && live.raisedHands.length > 0 ? `${live.raisedHands.length} student(s) have a question` : ""}
      </p>
    </main>
  );
}
