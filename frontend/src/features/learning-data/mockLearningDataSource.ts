import type {
  ConceptProgressStatus,
  KnowledgeDomain,
  KnowledgeMapDomainSnapshot,
  KnowledgeMapSnapshot,
  SessionState,
} from "../../app/session.types";
import { getStudyModule } from "../study/study.data";
import { createSessionSummary } from "../summary/summary.data";
import type { CompleteSessionInput, LearningDataSource } from "./learningDataSource";

export const KNOWLEDGE_DOMAINS: readonly KnowledgeDomain[] = [
  { id: "maths", label: "MATHS", conceptIds: ["fundamentals", "advanced-logic"] },
  { id: "quantum-physics", label: "QUANTUM PHYSICS", conceptIds: ["subject", "ethics", "practical-application", "tools"] },
  { id: "case-studies", label: "CASE STUDIES", conceptIds: ["case-studies"] },
  { id: "history", label: "HISTORY", conceptIds: ["history"] },
];

function resolveDomainStatus(statuses: readonly ConceptProgressStatus[]): ConceptProgressStatus {
  if (statuses.every((status) => status === "complete")) return "complete";
  if (statuses.some((status) => status !== "not_started")) return "in_progress";
  return "not_started";
}

export function createKnowledgeMapSnapshot(state: SessionState): KnowledgeMapSnapshot {
  const domains: KnowledgeMapDomainSnapshot[] = KNOWLEDGE_DOMAINS.map((domain) => {
    const concepts = domain.conceptIds.map((conceptId) => state.concepts[conceptId]);
    return {
      ...domain,
      readiness: Math.round(concepts.reduce((sum, concept) => sum + concept.readiness, 0) / concepts.length),
      completedCount: concepts.filter(({ status }) => status === "complete").length,
      status: resolveDomainStatus(concepts.map(({ status }) => status)),
    };
  });

  return { domains, generatedAt: new Date().toISOString() };
}

export const mockLearningDataSource: LearningDataSource = {
  async completeSession(input: CompleteSessionInput) {
    const module = getStudyModule(input.conceptId);
    const evidence = [...input.progress.session.messages]
      .reverse()
      .find(({ speaker }) => speaker === "teacher")?.text
      ?? (input.completionMode === "guided-explanation"
        ? "THE LEARNER REQUESTED A GUIDED EXPLANATION."
        : "NO TEACHING EVIDENCE RECORDED.");
    const gaps = module.teaching.gaps.map((gap) => ({ ...gap, evidence }));
    const summary = createSessionSummary(
      input.conceptId,
      { ...input.progress, latestGaps: gaps },
      input.completionMode,
      input.completedAt,
    );
    return { gaps, summary };
  },

  async getSessionSummary(conceptId, state) {
    return state.concepts[conceptId].latestSummary;
  },

  async getKnowledgeMap(state: SessionState) {
    return createKnowledgeMapSnapshot(state);
  },
};
