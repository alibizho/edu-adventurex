import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { SessionCompletionMode, TeachingMessage } from "../../app/session.types";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { apiMessage } from "../learning-data/apiClient";
import { backendLearningDataSource } from "../learning-data/backendLearningDataSource";
import {
  classChecklist,
  classSessionId,
  type BackendClassProgress,
  type ClassUnit,
  type GrowthPath,
  type TargetedQuestion,
} from "../learning-data/backend.types";
import { useClassroomCast } from "./classroom.cast";
import { SEATS, seatName, type SeatId } from "./classroom.seats";
import { BackendTeachingWorkspace } from "./components/BackendTeachingWorkspace";
import { Classroom } from "./components/Classroom";
import { MarkdownNotes } from "./components/MarkdownNotes";
import { ModuleBanner } from "./components/ModuleBanner";
import { SessionExitScene } from "./components/SessionExitScene";
import { usePressToTalkRecorder, type SpeechProsody } from "./usePressToTalkRecorder";
import type { StudyModule } from "./study.types";

const TOOLS = [
  { id: "map", label: "Map" },
  { id: "tutorial", label: "Tutorial" },
  { id: "reset", label: "Reset" },
] as const;

function message(speaker: TeachingMessage["speaker"], text: string): TeachingMessage {
  return { id: crypto.randomUUID(), speaker, text, createdAt: new Date().toISOString() };
}

function createVisualModule(path: GrowthPath, unit: ClassUnit): StudyModule {
  return {
    conceptId: "subject",
    moduleLabel: "CURRENT MODULE",
    title: path.confirmed_topic.toUpperCase(),
    student: { name: "AI STUDENT", readiness: 0 },
    tools: TOOLS,
    status: { learningLabel: "LEARNING_MATERIAL_PHASE", readyLabel: "READY_TO_TEACH", meta: "BACKEND: CONNECTED" },
    document: {
      id: `${path.path_id}_${unit.class_id}`.toUpperCase(),
      page: 1,
      pageCount: 1,
      title: unit.title.toUpperCase(),
      introduction: unit.objective,
      detail: unit.teacher_notes,
      figureCaption: "DYNAMIC LEARNING MATERIAL",
      readyLabel: "READY TO TEACH",
      completedLabel: "READY",
    },
    teaching: {
      initialQuestion: `CAN YOU TEACH ME ABOUT ${unit.title.toUpperCase()}?`,
      followUps: [],
      fallbackQuestion: "CAN YOU EXPLAIN THAT FROM ANOTHER ANGLE?",
      gaps: [],
    },
  };
}

type RaisedHand = { seatId: SeatId; question: TargetedQuestion; askedAt: number };

export function BackendStudyExperience({ pathId, classId }: { pathId: string; classId: string }) {
  const navigate = useNavigate();
  const [path, setPath] = useState<GrowthPath | null>(null);
  const [unit, setUnit] = useState<ClassUnit | null>(null);
  const [progress, setProgress] = useState<BackendClassProgress | null>(null);
  const [phase, setPhase] = useState<"reading" | "classroom" | "zoom">("reading");
  const [zoomSeat, setZoomSeat] = useState<SeatId | null>(null);
  const [hands, setHands] = useState<RaisedHand[]>([]);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [lastHeard, setLastHeard] = useState<string | null>(null);
  /** What the class told the teacher after they admitted they were stuck. */
  const [explanation, setExplanation] = useState<string | null>(null);
  const [lastConfidence, setLastConfidence] = useState<number | null>(null);
  const [turnCount, setTurnCount] = useState(0);
  // Readiness is NOT computed here any more. It is the share of class objectives the backend
  // judged as actually explained; a local turn-count estimate would just disagree with it.
  const readiness = progress?.readiness ?? 0;
  const [queueDepth, setQueueDepth] = useState(0);
  const [voiceAvailable, setVoiceAvailable] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const visualModule = useMemo(() => path && unit ? createVisualModule(path, unit) : null, [path, unit]);
  const sessionId = classSessionId(pathId, classId);
  // Held here rather than inside either view: both unmount every time the camera moves between
  // them, and re-dealing on the way would change who is sitting where mid-lesson.
  const cast = useClassroomCast(sessionId);

  // Recording and analysis run at different speeds: a chunk takes seconds to come back and the
  // teacher keeps talking through it. Chunks queue here and drain one at a time — the GPU
  // serializes anyway, so parallel uploads would only queue further downstream.
  const queueRef = useRef<{ audio: Blob; prosody: SpeechProsody }[]>([]);
  const drainingRef = useRef(false);
  const chunkIdRef = useRef(0);
  const transcriptRef = useRef<string[]>([]);
  transcriptRef.current = transcript;

  // Bumped by every reset. A teaching turn that was already in flight when the class was thrown
  // away compares this before touching state, so its answer can't land in the fresh session and
  // put back a sentence the learner just deleted.
  const generationRef = useRef(0);

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setError(null);
    // The GPU probe crosses a tunnel to a rented box and can take the better part of a minute when
    // that box is cold. It rides on its own promise rather than inside the Promise.all below so the
    // material still renders promptly; voice simply switches on when the answer lands.
    backendLearningDataSource
      .confusionHealth()
      .then((confusion) => { if (active) setVoiceAvailable(Boolean(confusion.reachable)); })
      .catch(() => { if (active) setVoiceAvailable(false); });

    Promise.all([
      backendLearningDataSource.getPath(pathId),
      backendLearningDataSource.getMemory(pathId),
    ])
      .then(async ([loadedPath, loadedMemory]) => {
        const selected = loadedPath.classes.find((candidate) => candidate.class_id === classId);
        if (!selected) throw new Error("THE REQUESTED CLASS DOES NOT EXIST.");
        const notes = selected.notes_generated
          ? selected
          : await backendLearningDataSource.generateNotes(pathId, classId);
        if (!active) return;
        const loaded = loadedMemory.class_progress[classId] ?? null;
        setPath(loadedPath);
        setUnit(notes);
        setProgress(loaded);
        setTurnCount(loaded?.turn_count ?? 0);
        localStorage.setItem("wut:active-path", pathId);
        localStorage.setItem("wut:active-class", classId);
      })
      .catch((caught) => { if (active) setError(apiMessage(caught)); })
      .finally(() => { if (active) setIsLoading(false); });
    return () => { active = false; };
  }, [classId, pathId]);

  const refreshProgress = useCallback(async () => {
    try {
      const memory = await backendLearningDataSource.getMemory(pathId);
      setProgress(memory.class_progress[classId] ?? null);
    } catch {
      // A missed poll just means the checklist updates on the next one.
    }
  }, [classId, pathId]);

  // Objective coverage is judged in a background task, so it can't ride back on the teaching
  // response. Poll for it whenever the class is live — a cheap DB read, no LLM.
  //
  // Keeps running through a one-to-one: answering a student's question is teaching too, and the
  // backend credits objectives from it, so freezing the checklist while zoomed in was half the
  // reported lag. The other half was the interval — the backend now judges after every chunk, and
  // a 5s poll on top of that put a checkmark up to five seconds behind a verdict already sitting
  // in the database.
  useEffect(() => {
    if (phase === "reading") return;
    const timer = window.setInterval(() => void refreshProgress(), 2_000);
    return () => window.clearInterval(timer);
  }, [phase, refreshProgress]);

  /**
   * Give a new question to one of the students who isn't already holding one, picked at random.
   *
   * In seat order it was always the same student: the first question of every class came from the
   * back-left desk, the second from the one beside it, and a learner who only ever triggers one or
   * two questions a class never saw a hand anywhere else. Which desk asks carries no meaning — the
   * backend knows nothing about seats — so it may as well come from a different corner each time.
   */
  const raiseHand = useCallback((question: TargetedQuestion) => {
    // Rolled out here on purpose: React invokes state updaters twice in development, and a
    // Math.random() inside would pick a different seat on each pass.
    const roll = Math.random();
    setHands((current) => {
      const taken = new Set(current.map((hand) => hand.seatId));
      const free = SEATS.filter((seat) => !taken.has(seat.id));
      const entry = { question, askedAt: Date.now() };
      if (free.length > 0) {
        return [...current, { seatId: free[Math.floor(roll * free.length)].id, ...entry }];
      }
      // Every seat is occupied: the teacher is talking past six unanswered questions. Retire the
      // oldest — a stale probe about something said minutes ago is worth less than this one.
      const oldest = current.reduce((a, b) => (a.askedAt <= b.askedAt ? a : b));
      return [...current.filter((hand) => hand !== oldest), { seatId: oldest.seatId, ...entry }];
    });
  }, []);

  const drain = useCallback(async () => {
    if (drainingRef.current) return;
    drainingRef.current = true;
    setIsBusy(true);
    try {
      while (queueRef.current.length > 0) {
        const { audio, prosody } = queueRef.current.shift()!;
        setQueueDepth(queueRef.current.length);
        const generation = generationRef.current;
        try {
          const response = await backendLearningDataSource.teachAudio(
            pathId, classId, audio, chunkIdRef.current, transcriptRef.current.slice(-6), true, prosody,
          );
          if (generationRef.current !== generation) continue;   // the class was reset under us
          chunkIdRef.current += 1;
          if (response.degraded || !response.analysis.text.trim()) {
            setVoiceAvailable(false);
            setError("GPU VOICE ANALYSIS IS OFFLINE — THE CLASS CANNOT HEAR YOU.");
            continue;
          }
          const heard = response.analysis.text.trim();
          setLastHeard(heard);
          setLastConfidence(response.analysis.confidence);
          setTranscript((current) => [...current, heard]);
          setTurnCount((current) => current + 1);
          if (response.asked && response.question) raiseHand(response.question);
          // The teacher said they didn't know, and the class answered instead of asking again.
          // This was being fetched and dropped on the floor, so admitting you were stuck got you
          // silence — the one response guaranteed not to help.
          if (response.explained && response.student_reply.trim()) {
            setExplanation(response.student_reply.trim());
          }
          // Every chunk now triggers a coverage check on the backend. Refresh per chunk rather
          // than once the whole queue empties: while the teacher keeps talking the queue may not
          // empty for a minute, and the checklist would sit still through all of it.
          void refreshProgress();
        } catch (caught) {
          if (generationRef.current === generation) setError(apiMessage(caught));
        }
      }
    } finally {
      drainingRef.current = false;
      setIsBusy(false);
      setQueueDepth(0);
      void refreshProgress();
    }
  }, [classId, pathId, raiseHand, refreshProgress]);

  /** Typed fallback when the GPU is down. Same shape as a spoken chunk, minus the audio analysis. */
  const teachText = useCallback(async (text: string) => {
    setIsBusy(true);
    setError(null);
    try {
      const response = await backendLearningDataSource.teachText(pathId, classId, text);
      setLastHeard(text);
      setTranscript((current) => [...current, text]);
      setTurnCount((current) => current + 1);
      if (response.asked && response.question) raiseHand(response.question);
      // Same flag as the audio path. On this one the reply is usually a student making
      // conversation, so the flag is the only thing that separates chatter from the answer.
      if (response.explained && response.student_reply.trim()) {
        setExplanation(response.student_reply.trim());
      }
    } catch (caught) {
      setError(apiMessage(caught));
    } finally {
      setIsBusy(false);
    }
  }, [classId, pathId, raiseHand]);

  const handleUtterance = useCallback((audio: Blob, prosody: SpeechProsody) => {
    queueRef.current.push({ audio, prosody });
    setQueueDepth(queueRef.current.length);
    void drain();
  }, [drain]);

  const recorder = usePressToTalkRecorder({ onUtterance: handleUtterance });
  const { cancel: cancelRecording } = recorder;

  // The mic belongs to the room. Entering a one-to-one conversation hands it to that view, and
  // leaving the class entirely must not leave the browser recording indicator on.
  useEffect(() => {
    if (phase !== "classroom") cancelRecording();
  }, [phase, cancelRecording]);

  const zoomHand = hands.find((hand) => hand.seatId === zoomSeat) ?? null;

  const resolveHand = useCallback((seatId: SeatId) => {
    setHands((current) => current.filter((hand) => hand.seatId !== seatId));
  }, []);

  // Stable identity: Classroom starts a 520 ms transition timer when a seat is clicked, and an
  // inline arrow here re-created that timer on every re-render of this component.
  const handleEnterZoom = useCallback((seatId: SeatId) => {
    setZoomSeat(seatId);
    setPhase("zoom");
  }, []);

  /**
   * Shared tail of both answer paths. The hand is NOT retired here, on any outcome.
   *
   * Retiring it is what removes this seat from `hands`, and `zoomHand` is looked up in that list —
   * so the moment the hand goes, the conversation unmounts and the room replaces it with "THAT
   * QUESTION IS RESOLVED". The final turn is the one that matters most: it carries the answer the
   * student gives when the teacher says they don't know, or when the thread runs out of tries.
   * Retiring on `conversation_over` deleted the view mid-sentence and the learner never read it.
   *
   * Walking out is what ends the exchange — both exits from the conversation call onBackToClass,
   * which retires the hand there.
   */
  const settleTurn = useCallback(async (
    hand: RaisedHand,
    answer: string,
    studentReply: string,
    followUp: TargetedQuestion | null,
    conversationOver: boolean,
  ) => {
    setTranscript((current) => [...current, answer]);
    if (followUp) {
      // Same seat, new question: the student stays put and presses on the same concept.
      setHands((current) => current.map((item) => (
        item.seatId === hand.seatId ? { ...item, question: followUp, askedAt: Date.now() } : item
      )));
    }
    return {
      teacherText: answer,
      studentText: studentReply || "OH — THAT MAKES SENSE NOW!",
      isOver: conversationOver,
    };
  }, [resolveHand]);

  async function answerZoomed(audio: Blob, prosody: SpeechProsody) {
    if (!zoomHand) return null;
    setIsBusy(true);
    setError(null);
    try {
      // Not silent: this student asked you something directly and should answer back.
      const response = await backendLearningDataSource.teachAudio(
        pathId, classId, audio, chunkIdRef.current, transcriptRef.current.slice(-6), false, prosody,
        zoomHand.question.id,
      );
      chunkIdRef.current += 1;
      if (response.degraded || !response.analysis.text.trim()) {
        setError("GPU VOICE ANALYSIS IS OFFLINE — THE STUDENT COULD NOT HEAR YOU.");
        return null;
      }
      return await settleTurn(
        zoomHand, response.analysis.text.trim(), response.student_reply,
        response.question,
        // A follow-up IS the conversation continuing, whatever the flag says. Only fall back to
        // ending it when there is genuinely nothing left to answer: closing the exchange is the
        // damaging default, because the `?` goes with it and the question cannot be retried.
        response.question ? false : response.conversation_over ?? true,
      );
    } catch (caught) {
      setError(apiMessage(caught));
      return null;
    } finally {
      setIsBusy(false);
    }
  }

  async function answerZoomedText(text: string) {
    if (!zoomHand) return null;
    setIsBusy(true);
    setError(null);
    try {
      // The text route has no grading path on the backend, so one answer settles it.
      const response = await backendLearningDataSource.teachText(pathId, classId, text);
      await backendLearningDataSource
        .answerQuestion(sessionId, zoomHand.question.id, text)
        .catch(() => undefined);
      return await settleTurn(zoomHand, text, response.student_reply, null, true);
    } catch (caught) {
      setError(apiMessage(caught));
      return null;
    } finally {
      setIsBusy(false);
    }
  }

  /**
   * Throw this class's session away and teach it again from nothing.
   *
   * For the lesson that went off the rails under its own power — a wrong tangent, a wrong topic,
   * ten minutes of thinking out loud. Without this the only exits were finishing a class the
   * learner knew was bad (and having it graded and folded into cross-class memory) or abandoning
   * it half-taught, and both leave the AI students holding questions about speech nobody stands
   * behind. Scoped to this class: the rest of the course is untouched.
   */
  async function resetClass() {
    if (isResetting) return;
    setIsResetting(true);
    // Everything still queued or still being recorded belongs to the lesson being deleted.
    generationRef.current += 1;
    queueRef.current = [];
    cancelRecording();
    try {
      const memory = await backendLearningDataSource.resetClass(pathId, classId);
      setProgress(memory.class_progress[classId] ?? null);
      setTranscript([]);
      setHands([]);
      setZoomSeat(null);
      setLastHeard(null);
      setExplanation(null);
      setLastConfidence(null);
      setTurnCount(0);
      setQueueDepth(0);
      chunkIdRef.current = 0;
      setError(null);
    } catch (caught) {
      setError(apiMessage(caught));
    } finally {
      setIsResetting(false);
    }
  }

  async function finish(completionMode: SessionCompletionMode) {
    if (isBusy) return false;
    cancelRecording();
    setIsBusy(true);
    setError(null);
    try {
      const updatedMemory = await backendLearningDataSource.endClass(pathId, classId, completionMode);
      // Keep the readiness the learner earned — the backend no longer stamps 100 on every class.
      setProgress(updatedMemory.class_progress[classId] ?? null);
      backendLearningDataSource.startAnalysis(sessionId).catch(() => undefined);
      setIsClosing(true);
      return true;
    } catch (caught) {
      setError(apiMessage(caught));
      return false;
    } finally {
      setIsBusy(false);
    }
  }

  const handleExitComplete = useCallback(() => {
    navigate(`${ROUTES.summary}?path=${encodeURIComponent(pathId)}&class=${encodeURIComponent(classId)}`, { replace: true });
  }, [classId, navigate, pathId]);

  if (isLoading) return <div className="screen study-screen"><AppHeader /><div className="backend-page-state retro-panel">GENERATING LEARNING MATERIAL...</div></div>;
  if (!path || !unit || !visualModule) return (
    <div className="screen study-screen"><AppHeader /><div className="backend-page-state retro-panel" role="alert"><strong>LEARNING MATERIAL UNAVAILABLE</strong><p>{error}</p><button className="solid-action" onClick={() => navigate(ROUTES.material)}>BACK HOME</button></div></div>
  );

  const isZoom = phase === "zoom";
  return (
    <div className={`screen study-screen${phase === "classroom" || isClosing ? " study-screen--lobby" : ""}${isZoom ? " study-screen--conversation" : ""}`}>
      <AppHeader />
      {isClosing ? (
        <SessionExitScene studyModule={visualModule} onComplete={handleExitComplete} />
      ) : phase === "classroom" ? (
        <Classroom
          studyModule={visualModule}
          readiness={readiness}
          cast={cast}
          objectives={classChecklist(unit)}
          coveredObjectives={progress?.covered_objectives ?? []}
          objectiveEvidence={progress?.objective_evidence ?? {}}
          onReturnToReading={() => setPhase("reading")}
          onEnterZoom={handleEnterZoom}
          live={{
            raisedHands: hands.map((hand) => hand.seatId),
            recorderState: recorder.state,
            meterRef: recorder.meterRef,
            elapsedSeconds: recorder.elapsedSeconds,
            queueDepth,
            isBusy,
            isResetting,
            lastHeard,
            explanation,
            turnCount,
            error: error ?? recorder.error,
            voiceAvailable,
            lastConfidence,
            onToggleMic: recorder.toggle,
            onDiscardRecording: cancelRecording,
            onDismissExplanation: () => setExplanation(null),
            onTextAnswer: (text) => void teachText(text),
            onReset: () => void resetClass(),
            onFinish: () => void finish("self-teaching"),
          }}
        />
      ) : (
        <>
          {/* No sidebar. It carried an AI-STUDENT card with a readiness bar, and Map / Tutorial /
              Reset — a duplicate of the header's MAP, a tip, and a button that reset only the
              local view. None of it belongs beside the material you are here to read. */}
          <main className="study-layout">
            <section className={isZoom ? "conversation-canvas" : "study-canvas halftone-screen"}>
              {isZoom && zoomHand ? (
                <BackendTeachingWorkspace
                  seatName={seatName(zoomHand.seatId)}
                  sprite={cast[zoomHand.seatId].sprite}
                  question={zoomHand.question}
                  isBusy={isBusy}
                  error={error}
                  voiceAvailable={voiceAvailable}
                  onAudioAnswer={answerZoomed}
                  onTextAnswer={answerZoomedText}
                  onBackToClass={() => {
                    // Walking away is what retires the hand — the student keeps it while there is
                    // still a follow-up outstanding.
                    resolveHand(zoomHand.seatId);
                    setZoomSeat(null);
                    setPhase("classroom");
                  }}
                />
              ) : isZoom ? (
                <div className="backend-page-state retro-panel">
                  <strong>THAT QUESTION IS RESOLVED</strong>
                  <button className="solid-action" onClick={() => { setZoomSeat(null); setPhase("classroom"); }}>BACK TO CLASS</button>
                </div>
              ) : (
                <>
                  <ModuleBanner label="CURRENT MODULE" title={path.confirmed_topic.toUpperCase()} />
                  <article className="document-panel retro-panel backend-document-panel">
                    <header className="document-meta"><span>DOC_ID: {visualModule.document.id}</span><span>LIVE BACKEND</span></header>
                    <h2>{unit.title.toUpperCase()}</h2>
                    <p>{unit.objective}</p>
                    <MarkdownNotes source={unit.teacher_notes} />
                    <button type="button" className="solid-action ready-action" onClick={() => setPhase("classroom")}>READY TO TEACH</button>
                  </article>
                </>
              )}
            </section>
          </main>
        </>
      )}
    </div>
  );
}
