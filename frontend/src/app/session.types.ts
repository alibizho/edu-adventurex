import type { ConceptId } from "../features/concepts/concepts.types";

export type LearningPhase = "reading" | "lobby" | "teaching" | "complete";

export type ConceptProgressStatus = "not_started" | "in_progress" | "complete";

export type SessionCompletionMode = "self-teaching" | "guided-explanation";

export type TeachingMessage = {
  id: string;
  speaker: "student" | "teacher";
  text: string;
  createdAt: string;
};

export type KnowledgeGapSeverity = "HIGH" | "MEDIUM" | "LOW";

export type KnowledgeGap = {
  id: string;
  title: string;
  severity: KnowledgeGapSeverity;
  whyItMatters: string;
  evidence: string;
  recommendedReview: string;
};

export type ConceptMasteryIcon = "arrow" | "solid" | "circle" | "info";

export type ConceptMasteryMetric = {
  id: string;
  label: string;
  score: number;
  icon: ConceptMasteryIcon;
};

export type GrowthMilestone = {
  id: string;
  label: string;
  status: "completed" | "next" | "locked";
};

export type SessionSummary = {
  conceptId: ConceptId;
  moduleTitle: string;
  topicTitle: string;
  readiness: number;
  completionMode: SessionCompletionMode;
  startedAt: string;
  completedAt: string;
  durationSeconds: number;
  studentsTaught: number;
  questionsAnswered: number;
  gapsDiscovered: number;
  masterTitle: string;
  masterQuote: string;
  rank: string;
  milestones: GrowthMilestone[];
  mastery: ConceptMasteryMetric[];
};

export type KnowledgeDomainId = "maths" | "quantum-physics" | "case-studies" | "history";

export type KnowledgeDomain = {
  id: KnowledgeDomainId;
  label: string;
  conceptIds: ConceptId[];
};

export type KnowledgeMapDomainSnapshot = KnowledgeDomain & {
  readiness: number;
  completedCount: number;
  status: ConceptProgressStatus;
};

export type KnowledgeMapSnapshot = {
  domains: KnowledgeMapDomainSnapshot[];
  generatedAt: string;
};

export type TeachingSession = {
  phase: LearningPhase;
  messages: TeachingMessage[];
  turnCount: number;
  nextPromptIndex: number;
  startedAt: string | null;
  completedAt: string | null;
  completionMode: SessionCompletionMode | null;
};

export type ConceptProgress = {
  conceptId: ConceptId;
  status: ConceptProgressStatus;
  readiness: number;
  lastStudiedAt: string | null;
  session: TeachingSession;
  latestGaps: KnowledgeGap[];
  latestSummary: SessionSummary | null;
};

export type SessionState = {
  version: 3;
  concepts: Record<ConceptId, ConceptProgress>;
};

export type SessionAction =
  | { type: "OPEN_LOBBY"; conceptId: ConceptId; openedAt: string }
  | { type: "RETURN_TO_READING"; conceptId: ConceptId }
  | {
      type: "START_TEACHING";
      conceptId: ConceptId;
      message: TeachingMessage;
    }
  | {
      type: "SUBMIT_ANSWER";
      conceptId: ConceptId;
      teacherMessage: TeachingMessage;
      studentMessage: TeachingMessage;
    }
  | {
      type: "FINISH_TEACHING";
      conceptId: ConceptId;
      completedAt: string;
      completionMode: SessionCompletionMode;
      gaps: KnowledgeGap[];
      summary: SessionSummary;
    }
  | { type: "RESET_CONCEPT"; conceptId: ConceptId }
  | { type: "START_REVIEW"; conceptId: ConceptId; startedAt: string };
