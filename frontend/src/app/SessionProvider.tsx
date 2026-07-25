import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type ReactNode,
} from "react";
import { CONCEPTS } from "../features/concepts/concepts.data";
import type { ConceptId } from "../features/concepts/concepts.types";
import { mockLearningDataSource } from "../features/learning-data/mockLearningDataSource";
import { createSessionSummary } from "../features/summary/summary.data";
import { getStudyModule } from "../features/study/study.data";
import type {
  ConceptProgress,
  KnowledgeGap,
  KnowledgeMapSnapshot,
  SessionAction,
  SessionCompletionMode,
  SessionState,
  SessionSummary,
  TeachingMessage,
} from "./session.types";

const SESSION_STORAGE_KEY = "wut:p0-session:v3";
const LEGACY_SESSION_STORAGE_KEYS = ["wut:p0-session:v2", "wut:p0-session:v1"] as const;
const SESSION_VERSION = 3 as const;
const MAX_STORED_MESSAGES = 41;

type SessionContextValue = {
  state: SessionState;
  openLobby: (conceptId: ConceptId) => void;
  returnToReading: (conceptId: ConceptId) => void;
  startTeaching: (conceptId: ConceptId) => void;
  submitAnswer: (conceptId: ConceptId, answer: string) => void;
  finishTeaching: (conceptId: ConceptId, mode?: SessionCompletionMode) => Promise<boolean>;
  resetConcept: (conceptId: ConceptId) => void;
  startReview: (conceptId: ConceptId) => void;
  getSessionSummary: (conceptId: ConceptId) => Promise<SessionSummary | null>;
  getKnowledgeMap: () => Promise<KnowledgeMapSnapshot>;
};

const SessionContext = createContext<SessionContextValue | null>(null);

function createEmptyProgress(conceptId: ConceptId): ConceptProgress {
  return {
    conceptId,
    status: "not_started",
    readiness: 0,
    lastStudiedAt: null,
    session: {
      phase: "reading",
      messages: [],
      turnCount: 0,
      nextPromptIndex: 0,
      startedAt: null,
      completedAt: null,
      completionMode: null,
    },
    latestGaps: [],
    latestSummary: null,
  };
}

export function createInitialSessionState(): SessionState {
  const concepts = Object.fromEntries(
    CONCEPTS.map(({ id }) => [id, createEmptyProgress(id)]),
  ) as Record<ConceptId, ConceptProgress>;

  return { version: SESSION_VERSION, concepts };
}

function isTeachingMessage(value: unknown): value is TeachingMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<TeachingMessage>;
  return (
    typeof message.id === "string"
    && (message.speaker === "student" || message.speaker === "teacher")
    && typeof message.text === "string"
    && typeof message.createdAt === "string"
  );
}

function isKnowledgeGap(value: unknown): value is KnowledgeGap {
  if (!value || typeof value !== "object") return false;
  const gap = value as Partial<KnowledgeGap>;
  return (
    typeof gap.id === "string"
    && typeof gap.title === "string"
    && (gap.severity === "HIGH" || gap.severity === "MEDIUM" || gap.severity === "LOW")
    && typeof gap.whyItMatters === "string"
    && typeof gap.evidence === "string"
    && typeof gap.recommendedReview === "string"
  );
}

function isSessionSummary(value: unknown, conceptId: ConceptId): value is SessionSummary {
  if (!value || typeof value !== "object") return false;
  const summary = value as Partial<SessionSummary>;
  return (
    summary.conceptId === conceptId
    && typeof summary.moduleTitle === "string"
    && typeof summary.topicTitle === "string"
    && typeof summary.readiness === "number"
    && (summary.completionMode === "self-teaching" || summary.completionMode === "guided-explanation")
    && typeof summary.startedAt === "string"
    && typeof summary.completedAt === "string"
    && typeof summary.durationSeconds === "number"
    && Array.isArray(summary.milestones)
    && Array.isArray(summary.mastery)
  );
}

function normalizeConceptProgress(value: unknown, conceptId: ConceptId): ConceptProgress | null {
  if (!value || typeof value !== "object") return null;
  const progress = value as Partial<ConceptProgress>;
  const session = progress.session;
  if (
    progress.conceptId !== conceptId
    || (progress.status !== "not_started" && progress.status !== "in_progress" && progress.status !== "complete")
    || typeof progress.readiness !== "number"
    || progress.readiness < 0
    || progress.readiness > 100
    || (progress.lastStudiedAt !== null && typeof progress.lastStudiedAt !== "string")
    || !session
    || (session.phase !== "reading" && session.phase !== "lobby" && session.phase !== "teaching" && session.phase !== "complete")
    || !Array.isArray(session.messages)
    || !session.messages.every(isTeachingMessage)
    || typeof session.turnCount !== "number"
    || typeof session.nextPromptIndex !== "number"
    || (session.startedAt !== null && typeof session.startedAt !== "string")
    || (session.completedAt !== null && typeof session.completedAt !== "string")
    || !Array.isArray(progress.latestGaps)
    || !progress.latestGaps.every(isKnowledgeGap)
  ) return null;

  const completionMode = session.completionMode === "guided-explanation"
    ? "guided-explanation"
    : session.completionMode === "self-teaching"
      ? "self-teaching"
      : null;
  const normalized: ConceptProgress = {
    conceptId,
    status: progress.status,
    readiness: progress.readiness,
    lastStudiedAt: progress.lastStudiedAt,
    session: {
      phase: session.phase,
      messages: session.messages,
      turnCount: session.turnCount,
      nextPromptIndex: session.nextPromptIndex,
      startedAt: session.startedAt,
      completedAt: session.completedAt,
      completionMode,
    },
    latestGaps: progress.latestGaps,
    latestSummary: isSessionSummary(progress.latestSummary, conceptId) ? progress.latestSummary : null,
  };

  if (normalized.status === "complete" && !normalized.latestSummary && normalized.session.completedAt) {
    normalized.latestSummary = createSessionSummary(
      conceptId,
      normalized,
      completionMode ?? "self-teaching",
      normalized.session.completedAt,
    );
  }

  return normalized;
}

function hydrateSessionState(): SessionState {
  const fallback = createInitialSessionState();

  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY)
      ?? LEGACY_SESSION_STORAGE_KEYS.map((key) => sessionStorage.getItem(key)).find(Boolean);
    if (!raw) return fallback;

    const parsed = JSON.parse(raw) as { version?: number; concepts?: Partial<Record<ConceptId, unknown>> };
    if (![1, 2, 3].includes(parsed.version ?? -1) || !parsed.concepts) return fallback;

    const normalizedEntries = CONCEPTS.map(({ id }) => {
      const normalized = normalizeConceptProgress(parsed.concepts?.[id], id);
      return normalized ? [id, normalized] as const : null;
    });
    if (normalizedEntries.some((entry) => entry === null)) return fallback;

    return {
      version: SESSION_VERSION,
      concepts: Object.fromEntries(normalizedEntries as Array<readonly [ConceptId, ConceptProgress]>) as Record<ConceptId, ConceptProgress>,
    };
  } catch {
    return fallback;
  }
}

function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  const current = state.concepts[action.conceptId];

  if (action.type === "RESET_CONCEPT") {
    return {
      ...state,
      concepts: { ...state.concepts, [action.conceptId]: createEmptyProgress(action.conceptId) },
    };
  }

  if (action.type === "START_REVIEW") {
    return {
      ...state,
      concepts: {
        ...state.concepts,
        [action.conceptId]: {
          ...createEmptyProgress(action.conceptId),
          status: "in_progress",
          lastStudiedAt: action.startedAt,
          latestGaps: current.latestGaps,
          latestSummary: current.latestSummary,
        },
      },
    };
  }

  if (action.type === "OPEN_LOBBY") {
    if (current.session.phase !== "reading") return state;
    return {
      ...state,
      concepts: {
        ...state.concepts,
        [action.conceptId]: {
          ...current,
          status: "in_progress",
          lastStudiedAt: action.openedAt,
          session: { ...current.session, phase: "lobby" },
        },
      },
    };
  }

  if (action.type === "RETURN_TO_READING") {
    if (current.session.phase === "reading") return state;
    return {
      ...state,
      concepts: {
        ...state.concepts,
        [action.conceptId]: {
          ...current,
          session: { ...current.session, phase: "reading" },
        },
      },
    };
  }

  if (action.type === "START_TEACHING") {
    if (current.session.phase !== "lobby") return state;
    return {
      ...state,
      concepts: {
        ...state.concepts,
        [action.conceptId]: {
          ...current,
          status: "in_progress",
          readiness: 25,
          lastStudiedAt: action.message.createdAt,
          session: {
            phase: "teaching",
            messages: [action.message],
            turnCount: 0,
            nextPromptIndex: 0,
            startedAt: action.message.createdAt,
            completedAt: null,
            completionMode: null,
          },
        },
      },
    };
  }

  if (action.type === "SUBMIT_ANSWER") {
    if (current.session.phase !== "teaching") return state;
    const turnCount = current.session.turnCount + 1;
    const messages = [
      ...current.session.messages,
      action.teacherMessage,
      action.studentMessage,
    ].slice(-MAX_STORED_MESSAGES);

    return {
      ...state,
      concepts: {
        ...state.concepts,
        [action.conceptId]: {
          ...current,
          status: "in_progress",
          readiness: Math.min(95, 25 + turnCount * 15),
          lastStudiedAt: action.teacherMessage.createdAt,
          session: {
            ...current.session,
            messages,
            turnCount,
            nextPromptIndex: current.session.nextPromptIndex + 1,
          },
        },
      },
    };
  }

  if (action.type === "FINISH_TEACHING") {
    if (current.session.phase !== "teaching") return state;
    return {
      ...state,
      concepts: {
        ...state.concepts,
        [action.conceptId]: {
          ...current,
          status: "complete",
          readiness: 100,
          lastStudiedAt: action.completedAt,
          latestGaps: action.gaps,
          latestSummary: action.summary,
          session: {
            ...current.session,
            phase: "complete",
            completedAt: action.completedAt,
            completionMode: action.completionMode,
          },
        },
      },
    };
  }

  return state;
}

function createMessage(speaker: TeachingMessage["speaker"], text: string): TeachingMessage {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    speaker,
    text,
    createdAt: new Date().toISOString(),
  };
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(sessionReducer, undefined, hydrateSessionState);

  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(state));
      LEGACY_SESSION_STORAGE_KEYS.forEach((key) => sessionStorage.removeItem(key));
    } catch {
    }
  }, [state]);

  const openLobby = useCallback((conceptId: ConceptId) => {
    dispatch({ type: "OPEN_LOBBY", conceptId, openedAt: new Date().toISOString() });
  }, []);

  const returnToReading = useCallback((conceptId: ConceptId) => {
    dispatch({ type: "RETURN_TO_READING", conceptId });
  }, []);

  const startTeaching = useCallback((conceptId: ConceptId) => {
    const module = getStudyModule(conceptId);
    dispatch({
      type: "START_TEACHING",
      conceptId,
      message: createMessage("student", module.teaching.initialQuestion),
    });
  }, []);

  const submitAnswer = useCallback((conceptId: ConceptId, answer: string) => {
    const trimmedAnswer = answer.trim();
    if (!trimmedAnswer) return;

    const current = state.concepts[conceptId];
    if (current.session.phase !== "teaching") return;

    const module = getStudyModule(conceptId);
    const nextPrompt = module.teaching.followUps[current.session.nextPromptIndex]
      ?? module.teaching.fallbackQuestion;

    dispatch({
      type: "SUBMIT_ANSWER",
      conceptId,
      teacherMessage: createMessage("teacher", trimmedAnswer),
      studentMessage: createMessage("student", nextPrompt),
    });
  }, [state.concepts]);

  const finishTeaching = useCallback(async (
    conceptId: ConceptId,
    completionMode: SessionCompletionMode = "self-teaching",
  ) => {
    const current = state.concepts[conceptId];
    if (current.session.phase !== "teaching") return false;
    if (completionMode === "self-teaching" && current.session.turnCount < 1) return false;

    const completedAt = new Date().toISOString();
    const result = await mockLearningDataSource.completeSession({
      conceptId,
      progress: current,
      completionMode,
      completedAt,
    });

    dispatch({
      type: "FINISH_TEACHING",
      conceptId,
      completedAt,
      completionMode,
      gaps: result.gaps,
      summary: result.summary,
    });
    return true;
  }, [state.concepts]);

  const resetConcept = useCallback((conceptId: ConceptId) => {
    dispatch({ type: "RESET_CONCEPT", conceptId });
  }, []);

  const startReview = useCallback((conceptId: ConceptId) => {
    dispatch({ type: "START_REVIEW", conceptId, startedAt: new Date().toISOString() });
  }, []);

  const getSessionSummary = useCallback((conceptId: ConceptId) => (
    mockLearningDataSource.getSessionSummary(conceptId, state)
  ), [state]);

  const getKnowledgeMap = useCallback(() => (
    mockLearningDataSource.getKnowledgeMap(state)
  ), [state]);

  const value = useMemo<SessionContextValue>(() => ({
    state,
    openLobby,
    returnToReading,
    startTeaching,
    submitAnswer,
    finishTeaching,
    resetConcept,
    startReview,
    getSessionSummary,
    getKnowledgeMap,
  }), [
    state,
    openLobby,
    returnToReading,
    startTeaching,
    submitAnswer,
    finishTeaching,
    resetConcept,
    startReview,
    getSessionSummary,
    getKnowledgeMap,
  ]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;

}

export function useSession() {
  const context = useContext(SessionContext);
  if (!context) throw new Error("useSession must be used inside SessionProvider");
  return context;
}
