import { useEffect, useRef, useState } from "react";
import { PixelMicIcon } from "../../../components/visuals/PixelIcons";
import type { TeachingMessage } from "../../../app/session.types";
import type { UnknownHelpScenario } from "../study.types";

const MOCK_VOICE_ANSWER = "Observation requires a measurement interaction, and that interaction changes the state we can observe.";
const LISTENING_DURATION_MS = 900;
const UNKNOWN_RESPONSE_DURATION_MS = 800;
const BOOK_OPENING_DURATION_MS = 1120;

type ConversationDisplayState =
  | "question"
  | "unknown-response"
  | "opening-book"
  | "explanation";

type TeachingWorkspaceProps = {
  messages: readonly TeachingMessage[];
  turnCount: number;
  isComplete: boolean;
  unknownHelp?: UnknownHelpScenario;
  onSubmit: (answer: string) => void;
  onFinish: () => Promise<boolean>;
  onBackToMaterial: () => void;
};

export function TeachingWorkspace({
  messages,
  turnCount,
  isComplete,
  unknownHelp,
  onSubmit,
  onFinish,
  onBackToMaterial,
}: TeachingWorkspaceProps) {
  const [isListening, setIsListening] = useState(false);
  const [isCompleting, setIsCompleting] = useState(false);
  const [completionError, setCompletionError] = useState<string | null>(null);
  const [displayState, setDisplayState] = useState<ConversationDisplayState>("question");
  const timersRef = useRef<number[]>([]);
  const latestQuestion = [...messages].reverse().find((message) => message.speaker === "student")?.text
    ?? "What would you like to teach me?";
  const isHelpSequenceActive = displayState !== "question";
  const isExplanation = displayState === "explanation";

  function clearConversationTimers() {
    timersRef.current.forEach((timerId) => window.clearTimeout(timerId));
    timersRef.current = [];
  }

  function schedule(callback: () => void, delay: number) {
    const timerId = window.setTimeout(callback, delay);
    timersRef.current.push(timerId);
  }

  useEffect(() => {
    return () => clearConversationTimers();
  }, []);

  function handleVoiceInput() {
    if (isListening || isHelpSequenceActive || isComplete) return;

    setIsListening(true);

    schedule(() => {
      setIsListening(false);

      if (!unknownHelp) {
        onSubmit(MOCK_VOICE_ANSWER);
        return;
      }

      setDisplayState("unknown-response");
      schedule(() => {
        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (reduceMotion) {
          setDisplayState("explanation");
          return;
        }

        setDisplayState("opening-book");
        schedule(() => setDisplayState("explanation"), BOOK_OPENING_DURATION_MS);
      }, UNKNOWN_RESPONSE_DURATION_MS);
    }, LISTENING_DURATION_MS);
  }

  function handleBackToMaterial() {
    if (isCompleting) return;
    clearConversationTimers();
    setIsListening(false);
    setDisplayState("question");
    onBackToMaterial();
  }

  async function handleAcknowledgement() {
    if (isCompleting) return;
    setCompletionError(null);
    setIsCompleting(true);
    try {
      const completed = await onFinish();
      if (!completed) {
        setCompletionError("SESSION COULD NOT BE COMPLETED. PLEASE TRY AGAIN.");
        setIsCompleting(false);
      }
    } catch {
      setCompletionError("SESSION COULD NOT BE COMPLETED. PLEASE TRY AGAIN.");
      setIsCompleting(false);
    }
  }

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

      <div
        className={`student-speech${isExplanation ? " student-speech--explanation" : ""}`}
        role="status"
        aria-live="polite"
      >
        <p>{isExplanation ? unknownHelp?.explanation : latestQuestion}</p>

      </div>

      {(displayState === "unknown-response" || displayState === "opening-book") && unknownHelp && (
        <div className="teacher-response" role="status" aria-live="polite">
          <p>{unknownHelp.response}</p>

        </div>

      )}

      <p className="sr-only" role="status" aria-live="polite">
        {displayState === "opening-book"
          ? "The student is opening a reference book."
          : ""}
      </p>

      {!isExplanation && (
        <div className="voice-control">
          <button
            type="button"
            className={`voice-button${isListening ? " is-listening" : ""}`}
            disabled={isListening || isHelpSequenceActive || isComplete}
            onClick={handleVoiceInput}
            aria-label={isListening ? "Listening" : "Start mock voice input"}
          >
            <PixelMicIcon />
          </button>

          <strong>{isListening ? "LISTENING..." : isComplete ? "SESSION COMPLETE" : "CLICK TO SPEAK"}</strong>

          <span>Voice input only: tone and pause analysis active.</span>

        </div>

      )}

      {isExplanation && unknownHelp && (
        <button
          type="button"
          className="acknowledgement-button"
          disabled={isCompleting}
          aria-busy={isCompleting}
          onClick={handleAcknowledgement}
        >
          {isCompleting ? "COMPLETING..." : unknownHelp.acknowledgementLabel}
        </button>

      )}

      {completionError && (
        <p className="conversation-completion-error" role="alert">{completionError}</p>

      )}

      <button type="button" className="back-to-material" disabled={isCompleting} onClick={handleBackToMaterial}>
        <span aria-hidden="true">←</span> BACK TO MATERIAL

      </button>

      <span className="conversation-turn" aria-label={`Teaching turn ${turnCount + 1}`}>
        TURN {String(turnCount + 1).padStart(2, "0")}
      </span>

      <div className="conversation-footer" aria-hidden="true">
        <i /><i /><i />
      </div>

    </section>

  );
}
