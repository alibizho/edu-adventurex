import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { SessionCompletionMode, TeachingMessage } from "../../app/session.types";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import { apiMessage } from "../learning-data/apiClient";
import { backendLearningDataSource } from "../learning-data/backendLearningDataSource";
import { classSessionId, type ClassUnit, type GrowthPath, type PathMemory } from "../learning-data/backend.types";
import { BackendTeachingWorkspace } from "./components/BackendTeachingWorkspace";
import { MarkdownNotes } from "./components/MarkdownNotes";
import { ModuleBanner } from "./components/ModuleBanner";
import { SessionExitScene } from "./components/SessionExitScene";
import { StudentSidebar } from "./components/StudentSidebar";
import { TeachingLobby } from "./components/TeachingLobby";
import type { StudyModule, StudyToolId } from "./study.types";

const TOOLS = [
  { id: "progress", label: "Progress" },
  { id: "tutorial", label: "Tutorial" },
  { id: "reset", label: "Reset" },
] as const;
const UNKNOWN_PATTERN = /\b(i\s*(do not|don't)\s*know|not\s*sure|no\s*idea|cannot\s*explain)\b/i;

function message(speaker: TeachingMessage["speaker"], text: string): TeachingMessage {
  return { id: crypto.randomUUID(), speaker, text, createdAt: new Date().toISOString() };
}

function plainNotes(notes: string) {
  return notes
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[#*_>`\[\]()!-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 1100);
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

export function BackendStudyExperience({ pathId, classId }: { pathId: string; classId: string }) {
  const navigate = useNavigate();
  const [path, setPath] = useState<GrowthPath | null>(null);
  const [unit, setUnit] = useState<ClassUnit | null>(null);
  const [memory, setMemory] = useState<PathMemory | null>(null);
  const [phase, setPhase] = useState<"reading" | "lobby" | "teaching">("reading");
  const [messages, setMessages] = useState<TeachingMessage[]>([]);
  const [turnCount, setTurnCount] = useState(0);
  const [readiness, setReadiness] = useState(0);
  const [voiceAvailable, setVoiceAvailable] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toolMessage, setToolMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const visualModule = useMemo(() => path && unit ? createVisualModule(path, unit) : null, [path, unit]);

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setError(null);
    Promise.all([
      backendLearningDataSource.getPath(pathId),
      backendLearningDataSource.getMemory(pathId),
      backendLearningDataSource.confusionHealth().catch(() => ({ reachable: false })),
    ])
      .then(async ([loadedPath, loadedMemory, confusion]) => {
        const selected = loadedPath.classes.find((candidate) => candidate.class_id === classId);
        if (!selected) throw new Error("THE REQUESTED CLASS DOES NOT EXIST.");
        const notes = selected.notes_generated
          ? selected
          : await backendLearningDataSource.generateNotes(pathId, classId);
        if (!active) return;
        const progress = loadedMemory.class_progress[classId];
        setPath(loadedPath);
        setUnit(notes);
        setMemory(loadedMemory);
        setReadiness(progress?.readiness ?? 0);
        setTurnCount(progress?.turn_count ?? 0);
        setVoiceAvailable(Boolean(confusion.reachable));
        setMessages([message("student", `CAN YOU TEACH ME ABOUT ${notes.title.toUpperCase()}?`)]);
        localStorage.setItem("wut:active-path", pathId);
        localStorage.setItem("wut:active-class", classId);
      })
      .catch((caught) => { if (active) setError(apiMessage(caught)); })
      .finally(() => { if (active) setIsLoading(false); });
    return () => { active = false; };
  }, [classId, pathId]);

  const applyTurn = useCallback((teacherText: string, studentText: string) => {
    setMessages((current) => [...current, message("teacher", teacherText), message("student", studentText)].slice(-41));
    setTurnCount((current) => {
      const next = current + 1;
      setReadiness(Math.min(95, 25 + next * 15));
      return next;
    });
    return { teacherText, studentText };
  }, []);

  async function teachText(answer: string) {
    if (!unit || isBusy) return null;
    setIsBusy(true);
    setError(null);
    try {
      const response = await backendLearningDataSource.teachText(pathId, classId, answer);
      const studentText = UNKNOWN_PATTERN.test(answer)
        ? plainNotes(unit.teacher_notes) || response.student_reply
        : response.question?.text ?? response.student_reply;
      return applyTurn(answer, studentText);
    } catch (caught) {
      setError(apiMessage(caught));
      return null;
    } finally {
      setIsBusy(false);
    }
  }

  async function teachAudio(audio: Blob) {
    if (!unit || isBusy) return null;
    setIsBusy(true);
    setError(null);
    try {
      const history = messages.filter(({ speaker }) => speaker === "teacher").map(({ text }) => text);
      const response = await backendLearningDataSource.teachAudio(pathId, classId, audio, turnCount, history);
      if (response.degraded || !response.analysis.text.trim()) {
        setVoiceAvailable(false);
        setError("GPU VOICE ANALYSIS IS OFFLINE. TEXT INPUT HAS BEEN ENABLED.");
        return null;
      }
      const teacherText = response.analysis.text.trim();
      const expanded = response.analysis.curriculum_update?.added_concepts ?? [];
      if (expanded.length > 0) {
        setToolMessage(`CURRICULUM EXPANDED: ${expanded.join(", ").toUpperCase()}`);
      } else if (response.analysis.anomalies.length > 0) {
        const labels = response.analysis.anomalies.map(({ type }) => type.replaceAll("_", " "));
        setToolMessage(`LIVE ANALYSIS: ${labels.join(" / ").toUpperCase()}`);
      } else {
        setToolMessage("LIVE ANALYSIS: EXPLANATION CLEAR.");
      }
      const studentText = UNKNOWN_PATTERN.test(teacherText)
        ? plainNotes(unit.teacher_notes) || response.student_reply
        : response.question?.text ?? response.student_reply;
      return applyTurn(teacherText, studentText);
    } catch (caught) {
      setVoiceAvailable(false);
      setError(`${apiMessage(caught)} TEXT INPUT HAS BEEN ENABLED.`);
      return null;
    } finally {
      setIsBusy(false);
    }
  }

  async function finish(completionMode: SessionCompletionMode) {
    if (isBusy) return false;
    setIsBusy(true);
    setError(null);
    try {
      const updatedMemory = await backendLearningDataSource.endClass(pathId, classId, completionMode);
      setMemory(updatedMemory);
      setReadiness(100);
      const sessionId = classSessionId(pathId, classId);
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

  function handleToolAction(toolId: StudyToolId) {
    if (toolId === "progress") {
      navigate(`${ROUTES.progress}?path=${encodeURIComponent(pathId)}`);
      return;
    }
    if (toolId === "tutorial") {
      setToolMessage(phase === "reading" ? "TIP: READ THE MATERIAL, THEN SELECT READY TO TEACH." : "TIP: EXPLAIN THE IDEA IN YOUR OWN WORDS.");
      return;
    }
    setPhase("reading");
    setMessages([message("student", `CAN YOU TEACH ME ABOUT ${unit?.title.toUpperCase() ?? "THIS TOPIC"}?`)]);
    setToolMessage("LOCAL VIEW RESET. BACKEND HISTORY IS RETAINED.");
  }

  if (isLoading) return <div className="screen study-screen"><AppHeader /><div className="backend-page-state retro-panel">GENERATING LEARNING MATERIAL...</div></div>;
  if (!path || !unit || !visualModule) return (
    <div className="screen study-screen"><AppHeader /><div className="backend-page-state retro-panel" role="alert"><strong>LEARNING MATERIAL UNAVAILABLE</strong><p>{error}</p><button className="solid-action" onClick={() => navigate(ROUTES.material)}>BACK HOME</button></div></div>
  );

  const isConversation = phase === "teaching";
  return (
    <div className={`screen study-screen${phase === "lobby" || isClosing ? " study-screen--lobby" : ""}${isConversation ? " study-screen--conversation" : ""}`}>
      <AppHeader />
      {isClosing ? (
        <SessionExitScene studyModule={visualModule} onComplete={handleExitComplete} />
      ) : phase === "lobby" ? (
        <TeachingLobby studyModule={visualModule} onReturnToReading={() => setPhase("reading")} onStartConversation={() => setPhase("teaching")} />
      ) : (
        <>
          <main className="study-layout">
            <StudentSidebar
              student={{ name: "AI STUDENT", readiness }}
              tools={TOOLS}
              message={toolMessage}
              onToolAction={handleToolAction}
              avatarVariant={isConversation ? "student" : "robot"}
            />
            <section className={isConversation ? "conversation-canvas" : "study-canvas halftone-screen"}>
              {isConversation ? (
                <BackendTeachingWorkspace
                  messages={messages}
                  turnCount={turnCount}
                  isVoiceAvailable={voiceAvailable}
                  isBusy={isBusy}
                  error={error}
                  onTextAnswer={teachText}
                  onAudioAnswer={teachAudio}
                  onFinish={finish}
                  onBackToMaterial={() => setPhase("reading")}
                />
              ) : (
                <>
                  <ModuleBanner label="CURRENT MODULE" title={path.confirmed_topic.toUpperCase()} />
                  <article className="document-panel retro-panel backend-document-panel">
                    <header className="document-meta"><span>DOC_ID: {visualModule.document.id}</span><span>LIVE BACKEND</span></header>
                    <h2>{unit.title.toUpperCase()}</h2>
                    <p>{unit.objective}</p>
                    <MarkdownNotes source={unit.teacher_notes} />
                    <button type="button" className="solid-action ready-action" onClick={() => setPhase("lobby")}>READY TO TEACH</button>
                  </article>
                </>
              )}
            </section>
          </main>
          {!isConversation && <StatusBar label="LEARNING_MATERIAL_PHASE" full meta={`PATH: ${path.path_id}`} />}
        </>
      )}
    </div>
  );
}
