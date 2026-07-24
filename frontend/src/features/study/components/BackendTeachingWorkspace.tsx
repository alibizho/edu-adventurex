import { useEffect, useRef, useState } from "react";
import type { TeachingMessage } from "../../../app/session.types";
import { useWavRecorder } from "../useWavRecorder";

const UNKNOWN_PATTERN = /\b(i\s*(do not|don't)\s*know|not\s*sure|no\s*idea|cannot\s*explain)\b/i;

type TurnResult = { teacherText: string; studentText: string };

type BackendTeachingWorkspaceProps = {
  messages: readonly TeachingMessage[];
  turnCount: number;
  isVoiceAvailable: boolean;
  isBusy: boolean;
  error: string | null;
  onTextAnswer: (answer: string) => Promise<TurnResult | null>;
  onAudioAnswer: (audio: Blob) => Promise<TurnResult | null>;
  onFinish: (mode: "self-teaching" | "guided-explanation") => Promise<boolean>;
  onBackToMaterial: () => void;
};

export function BackendTeachingWorkspace({
  messages,
  turnCount,
  isVoiceAvailable,
  isBusy,
  error,
  onTextAnswer,
  onAudioAnswer,
  onFinish,
  onBackToMaterial,
}: BackendTeachingWorkspaceProps) {
  const [answer, setAnswer] = useState("");
  const [isCompleting, setIsCompleting] = useState(false);
  const [displayState, setDisplayState] = useState<"question" | "opening-book" | "explanation">("question");
  const [explanation, setExplanation] = useState("");
  const timerRef = useRef<number | null>(null);
  const recorder = useWavRecorder();
  const latestQuestion = [...messages].reverse().find(({ speaker }) => speaker === "student")?.text
    ?? "WHAT WOULD YOU LIKE TO TEACH ME?";

  useEffect(() => () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
  }, []);

  function showTurn(result: TurnResult | null) {
    if (!result || !UNKNOWN_PATTERN.test(result.teacherText)) return;
    setExplanation(result.studentText);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setDisplayState("explanation");
      return;
    }
    setDisplayState("opening-book");
    timerRef.current = window.setTimeout(() => setDisplayState("explanation"), 1120);
  }

  async function submitText() {
    const trimmed = answer.trim();
    if (!trimmed || isBusy) return;
    setAnswer("");
    showTurn(await onTextAnswer(trimmed));
  }

  async function toggleRecording() {
    if (isBusy) return;
    if (!recorder.isRecording) {
      await recorder.start();
      return;
    }
    const audio = await recorder.stop();
    if (audio) showTurn(await onAudioAnswer(audio));
  }

  async function finish(mode: "self-teaching" | "guided-explanation") {
    setIsCompleting(true);
    const completed = await onFinish(mode);
    if (!completed) setIsCompleting(false);
  }

  const isExplanation = displayState === "explanation";
  return (
    <section className={`conversation-stage conversation-stage--${displayState}`} aria-label="AI student conversation">
      <div className="conversation-character" aria-hidden="true">
        {displayState === "opening-book" ? (
          <div className="conversation-character-sprite is-opening-book" />
        ) : isExplanation ? (
          <div className="conversation-character-sprite is-holding-book" />
        ) : (
          <img src="/images/wut-student-fullbody.png" alt="" />
        )}
      </div>

      <div className={`student-speech${isExplanation ? " student-speech--explanation" : ""}`} role="status" aria-live="polite">
        <p>{isExplanation ? explanation : latestQuestion}</p>
      </div>

      {!isExplanation && (
        isVoiceAvailable ? (
          <div className="voice-control">
            <button
              type="button"
              className={`voice-button${recorder.isRecording ? " is-listening" : ""}`}
              disabled={isBusy}
              onClick={toggleRecording}
              aria-label={recorder.isRecording ? "Stop recording" : "Start recording"}
            >
              <svg viewBox="0 0 48 48" aria-hidden="true" shapeRendering="crispEdges">
                <path d="M18 7h12v4h4v17h-4v4H18v-4h-4V11h4zm2 4v17h8V11zM8 23h5v8h4v4h14v-4h4v-8h5v9h-4v4h-9v6h6v4H15v-4h7v-6h-9v-4H8z" />
              </svg>
            </button>
            <strong>{recorder.isRecording ? "CLICK TO STOP" : isBusy ? "PROCESSING..." : "CLICK TO SPEAK"}</strong>
            <span>16KHZ WAV INPUT: TONE AND PAUSE ANALYSIS ACTIVE.</span>
          </div>
        ) : (
          <div className="backend-text-control">
            <label htmlFor="backend-teaching-answer">GPU VOICE ANALYSIS OFFLINE — TYPE YOUR EXPLANATION</label>
            <textarea
              id="backend-teaching-answer"
              value={answer}
              maxLength={2000}
              disabled={isBusy}
              onChange={(event) => setAnswer(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitText();
                }
              }}
            />
            <button type="button" className="solid-action" disabled={!answer.trim() || isBusy} onClick={submitText}>
              {isBusy ? "SENDING..." : "SEND"}
            </button>
          </div>
        )
      )}

      {isExplanation ? (
        <button type="button" className="acknowledgement-button" disabled={isCompleting} onClick={() => finish("guided-explanation")}>
          {isCompleting ? "COMPLETING..." : "I GOT IT!"}
        </button>
      ) : turnCount > 0 ? (
        <button type="button" className="acknowledgement-button backend-finish-button" disabled={isBusy || isCompleting} onClick={() => finish("self-teaching")}>
          {isCompleting ? "COMPLETING..." : "FINISH TEACHING"}
        </button>
      ) : null}

      {(error || recorder.error) && <p className="conversation-completion-error" role="alert">{error ?? recorder.error}</p>}
      <button type="button" className="back-to-material" disabled={isBusy || isCompleting} onClick={onBackToMaterial}>← BACK TO MATERIAL</button>
      <span className="conversation-turn">TURN {String(turnCount + 1).padStart(2, "0")}</span>
      <div className="conversation-footer" aria-hidden="true"><i /><i /><i /></div>
    </section>
  );
}
