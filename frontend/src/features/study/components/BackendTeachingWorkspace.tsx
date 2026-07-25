import { useEffect, useState } from "react";
import { PixelMicIcon } from "../../../components/visuals/PixelIcons";
import type { TargetedQuestion } from "../../learning-data/backend.types";
import { useContinuousRecorder, type SpeechProsody } from "../useContinuousRecorder";

type TurnResult = { teacherText: string; studentText: string };

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
  const [reply, setReply] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [draft, setDraft] = useState("");

  const recorder = useContinuousRecorder({
    onUtterance: (audio, prosody) => {
      if (isSending) return;
      setIsSending(true);
      void onAudioAnswer(audio, prosody)
        .then((result) => { if (result) setReply(result.studentText); })
        .finally(() => setIsSending(false));
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
    setIsSending(true);
    void onTextAnswer(trimmed)
      .then((result) => { if (result) setReply(result.studentText); })
      .finally(() => setIsSending(false));
  }

  // Once they've understood, stop listening — otherwise the room keeps recording a teacher who has
  // moved on and is now talking to someone else.
  useEffect(() => { if (reply) disarm(); }, [reply, disarm]);

  const busy = isBusy || isSending;
  const meter = Math.min(100, Math.round(recorder.level * 900));

  return (
    <section className="conversation-stage conversation-stage--question" aria-label={`Conversation with ${seatName}`}>
      <div className="conversation-character" aria-hidden="true">
        <img src="/images/wut-student-fullbody.png" alt="" />
      </div>

      <div className="student-speech" role="status" aria-live="polite">
        <strong className="student-speech-name">{seatName}</strong>
        <p>{reply ?? question.text}</p>
      </div>

      {reply ? (
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
          <div className={`voice-button is-armed${recorder.state === "speaking" ? " is-listening" : ""}`} aria-hidden="true">
            <PixelMicIcon />
          </div>
          <strong>
            {busy ? "THINKING..."
              : recorder.state === "calibrating" ? "LISTENING TO THE ROOM..."
              : recorder.state === "speaking" ? "HEARING YOU — PAUSE WHEN DONE"
              : "ANSWER OUT LOUD"}
          </strong>
          <div className="mic-meter" aria-hidden="true">
            <div className="mic-meter-fill" style={{ width: `${meter}%` }} />
          </div>
          {question.anomaly_type && <span>DETECTED: {question.anomaly_type.replaceAll("_", " ").toUpperCase()}</span>}
        </div>
      )}

      {(error || recorder.error) && <p className="conversation-completion-error" role="alert">{error ?? recorder.error}</p>}
      <button type="button" className="back-to-material" disabled={busy} onClick={onBackToClass}>← BACK TO CLASS</button>
      <div className="conversation-footer" aria-hidden="true"><i /><i /><i /></div>
    </section>
  );
}
