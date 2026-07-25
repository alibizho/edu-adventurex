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
  const [explanation, setExplanation] = useState<string | null>(null);
  const [lastConfidence, setLastConfidence] = useState<number | null>(null);
  const [turnCount, setTurnCount] = useState(0);
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
  const cast = useClassroomCast(sessionId);

  const queueRef = useRef<{ audio: Blob; prosody: SpeechProsody }[]>([]);
  const drainingRef = useRef(false);
  const chunkIdRef = useRef(0);
  const transcriptRef = useRef<string[]>([]);
  transcriptRef.current = transcript;

  const generationRef = useRef(0);

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setError(null);
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
    }
  }, [classId, pathId]);

  useEffect(() => {
    if (phase === "reading") return;
    const timer = window.setInterval(() => void refreshProgress(), 2_000);
    return () => window.clearInterval(timer);
  }, [phase, refreshProgress]);

  const raiseHand = useCallback((question: TargetedQuestion) => {
    const roll = Math.random();
    setHands((current) => {
      const taken = new Set(current.map((hand) => hand.seatId));
      const free = SEATS.filter((seat) => !taken.has(seat.id));
      const entry = { question, askedAt: Date.now() };
      if (free.length > 0) {
        return [...current, { seatId: free[Math.floor(roll * free.length)].id, ...entry }];
      }
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
          if (generationRef.current !== generation) continue;
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
          if (response.explained && response.student_reply.trim()) {
            setExplanation(response.student_reply.trim());
          }
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

  const teachText = useCallback(async (text: string) => {
    setIsBusy(true);
    setError(null);
    try {
      const response = await backendLearningDataSource.teachText(pathId, classId, text);
      setLastHeard(text);
      setTranscript((current) => [...current, text]);
      setTurnCount((current) => current + 1);
      if (response.asked && response.question) raiseHand(response.question);
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

  useEffect(() => {
    if (phase !== "classroom") cancelRecording();
  }, [phase, cancelRecording]);

  const zoomHand = hands.find((hand) => hand.seatId === zoomSeat) ?? null;

  const resolveHand = useCallback((seatId: SeatId) => {
    setHands((current) => current.filter((hand) => hand.seatId !== seatId));
  }, []);

  const handleEnterZoom = useCallback((seatId: SeatId) => {
    setZoomSeat(seatId);
    setPhase("zoom");
  }, []);

  const settleTurn = useCallback(async (
    hand: RaisedHand,
    answer: string,
    studentReply: string,
    followUp: TargetedQuestion | null,
    conversationOver: boolean,
  ) => {
    setTranscript((current) => [...current, answer]);
    if (followUp) {
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

  async function resetClass() {
    if (isResetting) return;
    setIsResetting(true);
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
          {}
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
