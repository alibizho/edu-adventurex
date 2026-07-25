import type {
  ConceptMasteryIcon,
  ConceptMasteryMetric,
  ConceptProgress,
  GrowthMilestone,
  SessionCompletionMode,
  SessionSummary,
} from "../../app/session.types";
import type { ConceptId } from "../concepts/concepts.types";
import { getStudyModule } from "../study/study.data";

type SummaryProfile = {
  masteryLabels: readonly [string, string, string, string];
  icons: readonly [ConceptMasteryIcon, ConceptMasteryIcon, ConceptMasteryIcon, ConceptMasteryIcon];
  nextTopics: readonly [string, string, string];
};

const DEFAULT_ICONS = ["arrow", "circle", "solid", "info"] as const;

export const SUMMARY_PROFILES: Record<ConceptId, SummaryProfile> = {
  fundamentals: {
    masteryLabels: ["DUALITY", "INTERFERENCE", "AMPLITUDES", "MEASUREMENT"],
    icons: DEFAULT_ICONS,
    nextTopics: ["SUPERPOSITION", "QUANTUM LOGIC", "MEASUREMENT TOOLS"],
  },
  ethics: {
    masteryLabels: ["PRIVACY", "UNCERTAINTY", "ACCOUNTABILITY", "COMMUNICATION"],
    icons: DEFAULT_ICONS,
    nextTopics: ["RESPONSIBLE SENSING", "SECURE COMPUTING", "PUBLIC IMPACT"],
  },
  "advanced-logic": {
    masteryLabels: ["SUPERPOSITION", "QUBIT LOGIC", "PHASE", "MEASUREMENT"],
    icons: DEFAULT_ICONS,
    nextTopics: ["QUANTUM GATES", "ENTANGLEMENT", "ALGORITHMS"],
  },
  "practical-application": {
    masteryLabels: ["QUBITS", "QUANTUM GATES", "INTERFERENCE", "READOUT"],
    icons: DEFAULT_ICONS,
    nextTopics: ["QUANTUM CIRCUITS", "ERROR CORRECTION", "ALGORITHMS"],
  },
  "case-studies": {
    masteryLabels: ["PATHS", "INTERFERENCE", "DECOHERENCE", "EVIDENCE"],
    icons: DEFAULT_ICONS,
    nextTopics: ["BELL TESTS", "QUANTUM ERASER", "DELAYED CHOICE"],
  },
  subject: {
    masteryLabels: ["SUPERPOSITION", "ENTANGLEMENT", "WAVE FUNCTION", "UNCERTAINTY"],
    icons: DEFAULT_ICONS,
    nextTopics: ["RELATIVITY", "STRING THEORY", "THERMODYNAMICS"],
  },
  tools: {
    masteryLabels: ["DETECTORS", "CALIBRATION", "NOISE", "READOUT"],
    icons: DEFAULT_ICONS,
    nextTopics: ["CRYOGENICS", "QUANTUM CONTROL", "ERROR ANALYSIS"],
  },
  history: {
    masteryLabels: ["PLANCK", "EINSTEIN", "SCHRÖDINGER", "BELL"],
    icons: DEFAULT_ICONS,
    nextTopics: ["COPENHAGEN", "EPR DEBATE", "MODERN TESTS"],
  },
};

function createMastery(progress: ConceptProgress, conceptId: ConceptId): ConceptMasteryMetric[] {
  const profile = SUMMARY_PROFILES[conceptId];
  const scores = progress.latestGaps.length
    ? [85, 100, 65, 100]
    : [100, 92, 84, 96];

  return profile.masteryLabels.map((label, index) => ({
    id: `${conceptId}-${label.toLowerCase().replaceAll(" ", "-")}`,
    label,
    score: scores[index],
    icon: profile.icons[index],
  }));
}

function createMilestones(conceptId: ConceptId): GrowthMilestone[] {
  const module = getStudyModule(conceptId);
  return [
    { id: conceptId, label: module.title, status: "completed" },
    ...SUMMARY_PROFILES[conceptId].nextTopics.map((label, index) => ({
      id: `${conceptId}-next-${index + 1}`,
      label,
      status: index === 0 ? "next" as const : "locked" as const,
    })),
  ];
}

export function createSessionSummary(
  conceptId: ConceptId,
  progress: ConceptProgress,
  completionMode: SessionCompletionMode,
  completedAt: string,
): SessionSummary {
  const module = getStudyModule(conceptId);
  const startedAt = progress.session.startedAt ?? completedAt;
  const durationSeconds = Math.max(
    0,
    Math.round((new Date(completedAt).getTime() - new Date(startedAt).getTime()) / 1000),
  );
  const questionsAnswered = completionMode === "guided-explanation"
    ? Math.max(1, progress.session.turnCount)
    : progress.session.turnCount;

  return {
    conceptId,
    moduleTitle: module.title,
    topicTitle: module.document.title,
    readiness: 100,
    completionMode,
    startedAt,
    completedAt,
    durationSeconds,
    studentsTaught: 1,
    questionsAnswered,
    gapsDiscovered: module.teaching.gaps.length,
    masterTitle: `MASTER OF ${module.document.title}`,
    masterQuote: `You explained ${module.document.title.toLowerCase()} with clear cause and effect. Your student is now ready to question the next layer of the topic.`,
    rank: "ELITE TEACHER",
    milestones: createMilestones(conceptId),
    mastery: createMastery(progress, conceptId),
  };
}
