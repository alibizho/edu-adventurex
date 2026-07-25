import type {
  ConceptProgress,
  KnowledgeGap,
  KnowledgeMapSnapshot,
  SessionCompletionMode,
  SessionState,
  SessionSummary,
} from "../../app/session.types";
import type { ConceptId } from "../concepts/concepts.types";

export type CompleteSessionInput = {
  conceptId: ConceptId;
  progress: ConceptProgress;
  completionMode: SessionCompletionMode;
  completedAt: string;
};

export type CompleteSessionResult = {
  gaps: KnowledgeGap[];
  summary: SessionSummary;
};

export interface LearningDataSource {
  completeSession(input: CompleteSessionInput): Promise<CompleteSessionResult>;
  getSessionSummary(conceptId: ConceptId, state: SessionState): Promise<SessionSummary | null>;
  getKnowledgeMap(state: SessionState): Promise<KnowledgeMapSnapshot>;
}
