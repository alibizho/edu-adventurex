import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { SessionCompletionMode } from "../../app/session.types";
import { ROUTES } from "../../app/routes";
import { useSession } from "../../app/SessionProvider";
import type { StudyModule, StudyToolId } from "./study.types";

export function useStudySession(studyModule: StudyModule) {
  const navigate = useNavigate();
  const {
    state,
    openLobby,
    returnToReading,
    startTeaching,
    submitAnswer,
    finishTeaching,
    resetConcept,
  } = useSession();
  const [toolMessage, setToolMessage] = useState<string | null>(null);
  const progress = state.concepts[studyModule.conceptId];

  function markReady() {
    openLobby(studyModule.conceptId);
    setToolMessage(null);
  }

  function handleReturnToReading() {
    returnToReading(studyModule.conceptId);
    setToolMessage(null);
  }

  function handleStartConversation() {
    startTeaching(studyModule.conceptId);
    setToolMessage(null);
  }

  function handleSubmitAnswer(answer: string) {
    submitAnswer(studyModule.conceptId, answer);
    setToolMessage(null);
  }

  async function handleFinish(mode: SessionCompletionMode = "self-teaching") {
    return finishTeaching(studyModule.conceptId, mode);
  }

  function handleToolAction(toolId: StudyToolId) {
    if (toolId === "progress") {
      navigate(`${ROUTES.progress}?concept=${studyModule.conceptId}`);
      return;
    }

    if (toolId === "tutorial") {
      setToolMessage(progress.session.phase === "reading"
        ? "TIP: READ THE MATERIAL, THEN SELECT READY TO TEACH."
        : progress.session.phase === "lobby"
          ? "TIP: SELECT A QUESTION MARK TO START THE CONVERSATION."
          : "TIP: ANSWER IN YOUR OWN WORDS. FINISH WHEN YOUR EXPLANATION IS COMPLETE.");
      return;
    }

    resetConcept(studyModule.conceptId);
    setToolMessage("SESSION RESET.");
  }

  const statusLabel = progress.session.phase === "reading"
    ? studyModule.status.learningLabel
    : progress.session.phase === "lobby"
      ? "SELECT_A_STUDENT"
    : progress.session.phase === "teaching"
      ? "TEACHING_IN_PROGRESS"
      : "SESSION_COMPLETE";

  return {
    student: { ...studyModule.student, readiness: progress.readiness },
    progress,
    statusLabel,
    toolMessage,
    markReady,
    returnToReading: handleReturnToReading,
    startConversation: handleStartConversation,
    submitAnswer: handleSubmitAnswer,
    finish: handleFinish,
    handleToolAction,
  };
}
