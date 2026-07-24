import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";
import type { StudyRouteState } from "../concepts/concepts.types";
import { LearningDocument } from "./components/LearningDocument";
import { ModuleBanner } from "./components/ModuleBanner";
import { StudentSidebar } from "./components/StudentSidebar";
import { SessionExitScene } from "./components/SessionExitScene";
import { TeachingLobby } from "./components/TeachingLobby";
import { TeachingWorkspace } from "./components/TeachingWorkspace";
import { getStudyModule, resolveStudyConceptId } from "./study.data";
import type { StudyModule } from "./study.types";
import { useStudySession } from "./useStudySession";
import { BackendStudyExperience } from "./BackendStudyExperience";

type StudyExperienceProps = {
  studyModule: StudyModule;
};

function StudyExperience({ studyModule }: StudyExperienceProps) {
  const navigate = useNavigate();
  const session = useStudySession(studyModule);
  const [isClosing, setIsClosing] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [session.progress.session.phase]);

  useEffect(() => {
    if (session.progress.session.phase === "complete" && !isClosing) {
      navigate(`${ROUTES.summary}?concept=${studyModule.conceptId}`, { replace: true });
    }
  }, [isClosing, navigate, session.progress.session.phase, studyModule.conceptId]);

  const handleExitComplete = useCallback(() => {
    navigate(`${ROUTES.summary}?concept=${studyModule.conceptId}`, { replace: true });
  }, [navigate, studyModule.conceptId]);

  async function handleGuidedCompletion() {
    const completed = await session.finish("guided-explanation");
    if (completed) setIsClosing(true);
    return completed;
  }

  const isLobby = session.progress.session.phase === "lobby";
  const isConversation = session.progress.session.phase === "teaching"
    || session.progress.session.phase === "complete";

  return (
    <div className={`screen study-screen${isLobby || isClosing ? " study-screen--lobby" : ""}${isConversation && !isClosing ? " study-screen--conversation" : ""}`}>
      <AppHeader />
      {isClosing ? (
        <SessionExitScene studyModule={studyModule} onComplete={handleExitComplete} />
      ) : isLobby ? (
        <TeachingLobby
          studyModule={studyModule}
          onReturnToReading={session.returnToReading}
          onStartConversation={session.startConversation}
        />
      ) : (
        <>
          <main className="study-layout">
            <StudentSidebar
              student={session.student}
              tools={studyModule.tools}
              message={session.toolMessage}
              onToolAction={session.handleToolAction}
              avatarVariant={isConversation ? "student" : "robot"}
            />

            <section className={isConversation ? "conversation-canvas" : "study-canvas halftone-screen"}>
              {isConversation ? (
                <TeachingWorkspace
                  messages={session.progress.session.messages}
                  turnCount={session.progress.session.turnCount}
                  isComplete={session.progress.session.phase === "complete"}
                  unknownHelp={studyModule.teaching.unknownHelp}
                  onSubmit={session.submitAnswer}
                  onFinish={handleGuidedCompletion}
                  onBackToMaterial={session.returnToReading}
                />
              ) : (
                <>
                  <ModuleBanner label={studyModule.moduleLabel} title={studyModule.title} />
                <LearningDocument
                  document={studyModule.document}
                  isReady={false}
                  onReady={session.markReady}
                />
                </>
              )}
            </section>
          </main>
          {!isConversation && (
            <StatusBar label={session.statusLabel} full meta={studyModule.status.meta} />
          )}
        </>
      )}
    </div>
  );
}

export function StudyPage() {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const routeState = location.state as StudyRouteState | null;
  const pathId = searchParams.get("path");
  const classId = searchParams.get("class");
  if (pathId && classId) {
    return <BackendStudyExperience key={`${pathId}:${classId}`} pathId={pathId} classId={classId} />;
  }
  const conceptId = resolveStudyConceptId(
    searchParams.get("concept"),
    routeState?.concept?.id,
  );
  const studyModule = getStudyModule(conceptId);

  return <StudyExperience key={studyModule.conceptId} studyModule={studyModule} />;
}
