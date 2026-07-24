import type { ConceptId } from "../concepts/concepts.types";
import type { KnowledgeGapSeverity } from "../../app/session.types";

export type StudentStatus = {
  name: string;
  readiness: number;
};

export type StudyToolId = "progress" | "tutorial" | "reset";

export type StudyToolConfig = {
  id: StudyToolId;
  label: string;
};

export type StudyRuntimeStatus = {
  learningLabel: "LEARNING_MATERIAL_PHASE";
  readyLabel: "READY_TO_TEACH";
  meta: string;
};

export type StudyDocument = {
  id: string;
  page: number;
  pageCount: number;
  title: string;
  introduction: string;
  detail: string;
  figureCaption: string;
  readyLabel: string;
  completedLabel: string;
};

export type KnowledgeGapTemplate = {
  id: string;
  title: string;
  severity: KnowledgeGapSeverity;
  whyItMatters: string;
  recommendedReview: string;
};

export type UnknownHelpScenario = {
  response: "I don't know...";
  explanation: string;
  acknowledgementLabel: "I GOT IT!";
};

export type TeachingScenario = {
  initialQuestion: string;
  followUps: readonly string[];
  fallbackQuestion: string;
  gaps: readonly KnowledgeGapTemplate[];
  unknownHelp?: UnknownHelpScenario;
};

export type StudyModule = {
  conceptId: ConceptId;
  moduleLabel: string;
  title: string;
  student: StudentStatus;
  tools: readonly StudyToolConfig[];
  status: StudyRuntimeStatus;
  document: StudyDocument;
  teaching: TeachingScenario;
};
