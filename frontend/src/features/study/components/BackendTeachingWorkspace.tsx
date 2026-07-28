import { useEffect, useRef, useState } from "react";
import { PixelMicIcon } from "../../../components/visuals/PixelIcons";
import type { TargetedQuestion } from "../../learning-data/backend.types";
import { formatElapsed, usePressToTalkRecorder, type SpeechProsody } from "../usePressToTalkRecorder";

type TurnResult = { teacherText: string; studentText: string; isOver: boolean };

type BackendTeachingWorkspaceProps = {
  seatName: string;
  sprite: string;
  question: TargetedQuestion;
  isBusy: boolean;
  error: string | null;
  voiceAvailable: boolean;
  onAudioAnswer: (audio: Blob, prosody: SpeechProsody) => Promise<TurnResult | null>;
  onTextAnswer: (text: string) => Promise<TurnResult | null>;
  onBackToClass: () => void;
};

export function BackendTeachingWorkspace({
  seatName,
  sprite,
  question,
  isBusy,
  error,
  voiceAvailable,
  onAudioAnswer,
  onTextAnswer,
  onBackToClass,
}: BackendTeachingWorkspaceProps) {
  const [turns, setTurns] = useState<{ speaker: "you" | "student"; text: string }[]>([]);
  const [isOver, setIsOver] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [draft, setDraft] = useState("");

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

  const recorder = usePressToTalkRecorder({
    onUtterance: (audio, prosody) => {
      if (isSendingRef.current) return;
      startSending();
      void onAudioAnswer(audio, prosody).then(recordTurn).finally(finishSending);
    },
  });

  const { cancel, toggle } = recorder;

  function sendText() {
    const trimmed = draft.trim();
    if (!trimmed || isSending) return;
    setDraft("");
    startSending();
    void onTextAnswer(trimmed).then(recordTurn).finally(finishSending);
  }

  useEffect(() => { if (isOver) cancel(); }, [isOver, cancel]);

  const busy = isBusy || isSending;
  const lastStudentLine = [...turns].reverse().find((turn) => turn.speaker === "student")?.text;

  return (
    <section className="conversation-stage conversation-stage--question" aria-label={`Conversation with ${seatName}`}>
      <div className="conversation-character" aria-hidden="true">
        <img src={sprite} alt="" />
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
          {}
          <button
            type="button"
            className={`voice-button${recorder.isRecording ? " is-listening" : ""}${recorder.isArmed ? " is-armed" : ""}`}
            aria-label={recorder.isRecording ? "Stop recording and send your answer" : "Record your answer"}
            aria-pressed={recorder.isArmed}
            disabled={busy}
            onClick={toggle}
          >
            <PixelMicIcon />
          </button>

          <strong>
            {busy ? "THINKING..."
              : recorder.state === "starting" ? "OPENING THE MIC..."
              : recorder.isRecording ? `RECORDING ${formatElapsed(recorder.elapsedSeconds)} — PRESS AGAIN TO SEND`
              : "PRESS THE MIC TO ANSWER"}
          </strong>

          {recorder.isArmed && (
            <div className="mic-meter" aria-hidden="true">
              <div className="mic-meter-fill" ref={recorder.meterRef} />

            </div>

          )}
          {recorder.isRecording && (
            <button type="button" className="outline-action discard-take" onClick={cancel}>
              DISCARD
            </button>

          )}
          {question.anomaly_type && <span>DETECTED: {question.anomaly_type.replaceAll("_", " ").toUpperCase()}</span>}

        </div>

      )}

      {(error || recorder.error) && <p className="conversation-completion-error" role="alert">{error ?? recorder.error}</p>}

      {!isOver && (
        <button type="button" className="back-to-material" disabled={busy} onClick={onBackToClass}>← BACK TO CLASS</button>
      )}

      <div className="conversation-footer" aria-hidden="true"><i /><i /><i /></div>

    </section>

  );
}
