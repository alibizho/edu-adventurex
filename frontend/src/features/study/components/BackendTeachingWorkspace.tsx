import { useEffect, useRef, useState } from "react";
import { PixelMicIcon } from "../../../components/visuals/PixelIcons";
import type { TargetedQuestion } from "../../learning-data/backend.types";
import { useContinuousRecorder, type SpeechProsody } from "../useContinuousRecorder";

type TurnResult = { teacherText: string; studentText: string; isOver: boolean };

type BackendTeachingWorkspaceProps = {
  seatName: string;
  question: TargetedQuestion;
  isBusy: boolean;
  error: string | null;
  voiceAvailable: boolean;
  onAudioAnswer: (audio: Blob, prosody: SpeechProsody) => Promise<TurnResult | null>;
  onTextAnswer: (text: string) => Promise<TurnResult | null>;
  onBackToClass: () => void;
};

/**
 * One student, up close. You got here by clicking the `?` over their head, so there is exactly one
 * thing to do: answer what they asked. The mic is armed on arrival — this is a conversation, not a
 * form, and making the teacher hunt for a button breaks the momentum they had while teaching.
 *
 * It really is a conversation: the backend grades each answer against the question's answer key, so
 * a vague one earns a follow-up on the same concept rather than being accepted. The exchange ends
 * when the student is satisfied (or has run out of follow-ups) — `isOver` on the turn result.
 */
export function BackendTeachingWorkspace({
  seatName,
  question,
  isBusy,
  error,
  voiceAvailable,
  onAudioAnswer,
  onTextAnswer,
  onBackToClass,
}: BackendTeachingWorkspaceProps) {
  // The thread so far, oldest first. `isOver` is tracked separately from "has the student said
  // anything": a reply that comes with a follow-up means the conversation continues.
  const [turns, setTurns] = useState<{ speaker: "you" | "student"; text: string }[]>([]);
  const [isOver, setIsOver] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [draft, setDraft] = useState("");

  // The audio path runs from onaudioprocess, outside React's schedule, so it cannot read the state
  // above: between a send resolving and React committing the reset, a closing utterance would see a
  // stale `true` and be dropped silently — the teacher's sentence vanishes and nobody replies.
  const isSendingRef = useRef(false);

  const startSending = () => {
    isSendingRef.current = true;
    setIsSending(true);
  };
  const finishSending = () => {
    isSendingRef.current = false;
    setIsSending(false);
  };

  const recordTurn = (result: TurnResult | null) => {
    if (!result) return;
    setTurns((current) => [
      ...current,
      { speaker: "you", text: result.teacherText },
      { speaker: "student", text: result.studentText },
    ]);
    if (result.isOver) setIsOver(true);
  };

  const recorder = useContinuousRecorder({
    onUtterance: (audio, prosody) => {
      if (isSendingRef.current) return;
      startSending();
      void onAudioAnswer(audio, prosody).then(recordTurn).finally(finishSending);
    },
  });

  const { arm, disarm } = recorder;
  useEffect(() => {
    if (!voiceAvailable) return;
    void arm();
    return disarm;
  }, [arm, disarm, voiceAvailable]);

  function sendText() {
    const trimmed = draft.trim();
    if (!trimmed || isSending) return;
    setDraft("");
    startSending();
    void onTextAnswer(trimmed).then(recordTurn).finally(finishSending);
  }

  // Stop listening only once the student is actually satisfied. Disarming after every reply is what
  // made this a one-shot form: a follow-up would arrive with no way left to answer it.
  useEffect(() => { if (isOver) disarm(); }, [isOver, disarm]);

  const busy = isBusy || isSending;
  // What the student is saying right now: their newest line, or the question that got you here.
  const lastStudentLine = [...turns].reverse().find((turn) => turn.speaker === "student")?.text;

  return (
    <section className="conversation-stage conversation-stage--question" aria-label={`Conversation with ${seatName}`}>
      <div className="conversation-character" aria-hidden="true">
        <img src="/images/wut-student-fullbody.png" alt="" />
      </div>

      <div className="student-speech" role="status" aria-live="polite">
        <strong className="student-speech-name">{seatName}</strong>
        <p>{lastStudentLine ?? question.text}</p>
        {turns.length > 2 && (
          <ol className="student-speech-thread">
            {turns.slice(0, -2).map((turn, index) => (
              <li key={index} className={`thread-turn thread-turn--${turn.speaker}`}>
                <span>{turn.speaker === "you" ? "YOU" : seatName}</span> {turn.text}
              </li>
            ))}
          </ol>
        )}
      </div>

      {isOver ? (
        <button type="button" className="acknowledgement-button" onClick={onBackToClass}>
          BACK TO CLASS
        </button>
      ) : !voiceAvailable ? (
        <div className="backend-text-control">
          <label htmlFor="zoom-answer">GPU VOICE ANALYSIS OFFLINE — TYPE YOUR ANSWER</label>
          <textarea
            id="zoom-answer"
            value={draft}
            maxLength={2000}
            disabled={busy}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendText();
              }
            }}
          />
          <button type="button" className="solid-action" disabled={!draft.trim() || busy} onClick={sendText}>
            {busy ? "SENDING..." : "ANSWER"}
          </button>
        </div>
      ) : (
        <div className="voice-control">
          {/* A real button, not decoration. It used to be an aria-hidden <div> with no handler, so
              a teacher who wanted to stop talking — or restart after the mic dropped — had nothing
              to press and no way to tell the room was still listening. */}
          <button
            type="button"
            className={`voice-button${recorder.state === "speaking" ? " is-listening" : ""}${recorder.isArmed ? " is-armed" : ""}`}
            aria-label={recorder.isArmed ? "Stop answering" : "Start answering"}
            aria-pressed={recorder.isArmed}
            onClick={() => (recorder.isArmed ? disarm() : void arm())}
          >
            <PixelMicIcon />
          </button>
          <strong>
            {busy ? "THINKING..."
              : !recorder.isArmed ? "CLICK THE MIC TO ANSWER"
              : recorder.state === "calibrating" ? "LISTENING TO THE ROOM..."
              : recorder.state === "speaking" ? "HEARING YOU — PAUSE WHEN DONE"
              : "ANSWER OUT LOUD"}
          </strong>
          {recorder.isArmed && (
            <div className="mic-meter" aria-hidden="true">
              <div className="mic-meter-fill" ref={recorder.meterRef} />
            </div>
          )}
          {question.anomaly_type && <span>DETECTED: {question.anomaly_type.replaceAll("_", " ").toUpperCase()}</span>}
        </div>
      )}

      {(error || recorder.error) && <p className="conversation-completion-error" role="alert">{error ?? recorder.error}</p>}
      <button type="button" className="back-to-material" disabled={busy} onClick={onBackToClass}>← BACK TO CLASS</button>
      <div className="conversation-footer" aria-hidden="true"><i /><i /><i /></div>
    </section>
  );
}
