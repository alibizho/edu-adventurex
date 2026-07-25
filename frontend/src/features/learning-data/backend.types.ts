export type BackendClassProgress = {
  status: "not_started" | "in_progress" | "complete";
  readiness: number;
  turn_count: number;
  started_at: number | null;
  completed_at: number | null;
  completion_mode: "self-teaching" | "guided-explanation" | null;
  analysis_status: "not_started" | "pending" | "running" | "complete" | "failed";
  analysis_error: string | null;
  covered_objectives: string[];
  objective_evidence: Record<string, string>;
  passed_on_mastery: boolean;
};

export type PathMemory = {
  path_id: string;
  covered_concepts: string[];
  asked_questions: string[];
  understood: string[];
  struggled: string[];
  expanded_concepts: string[];
  class_progress: Record<string, BackendClassProgress>;
};

export type ClassObjective = {
  id: string;
  text: string;
};

export type ClassUnit = {
  class_id: string;
  title: string;
  objective: string;
  objectives: ClassObjective[];
  difficulty: string;
  prerequisites: string[];
  teacher_notes: string;
  notes_generated: boolean;
};

export function classChecklist(unit: ClassUnit): ClassObjective[] {
  return unit.objectives.length > 0 ? unit.objectives : [{ id: "o1", text: unit.objective }];
}

export type GrowthPath = {
  path_id: string;
  original_input: string;
  confirmed_topic: string;
  total_classes: number;
  recommended_order: string[];
  classes: ClassUnit[];
  source_material_summary: string | null;
};

export type ScopeSuggestion = {
  topic: string;
  rationale: string;
  suggested_classes: number;
};

export type TopicScope = {
  is_broad: boolean;
  suggestions: ScopeSuggestion[];
  confirmed_topic: string;
  suggested_classes: number;
};

export type MaterialExtraction = {
  material_text: string;
  files: Array<{
    name: string;
    media_type: string;
    size: number;
    extracted_characters: number;
  }>;
  warnings: string[];
  truncated: boolean;
};

export type BackendSegment = {
  id: number;
  idx: number;
  text: string;
  t_start: number | null;
  t_end: number | null;
};

export type AnalysisWordScore = {
  word: string;
  hesitation_zscore: number;
  is_anomaly: boolean;
  pace_zscore: number;
  is_bottleneck: boolean;
  attention_entropy: number;
  is_scattered: boolean;
};

export type StudentQuestion = {
  question_text: string;
  target_concept: string;
  anomaly_type: string;
};

export type CurriculumUpdate = {
  added_concepts: string[];
};

export type ChunkAnalysis = {
  chunk_id: number;
  text: string;
  confidence: number;
  anomalies: Array<{ type: string; source: string; score: number; evidence: string | null }>;
  localized_target: string | null;
  detail: AnalysisWordScore[];
  student_question: StudentQuestion | null;
  curriculum_update: CurriculumUpdate | null;
};

export type TargetedQuestion = {
  id: number;
  chunk_id: number;
  text: string;
  anomaly_type: string | null;
  rationale: string | null;
  parent_id?: number | null;
};

export type BuildEvent =
  | { stage: "topic"; topic: string; classes: number }
  | { stage: "structuring"; topic: string; classes: number }
  | { stage: "class"; index: number; total: number; title: string }
  | { stage: "writing"; total: number }
  | { stage: "written"; index: number; total: number; title: string; ok: boolean }
  | { stage: "done"; path: GrowthPath }
  | { stage: "error"; message: string };

export type ClassTeachResponse = {
  student_reply: string;
  new_segment: BackendSegment;
  asked: boolean;
  question: TargetedQuestion | null;
  explained?: boolean;
};

export type AudioClassTeachResponse = Omit<ClassTeachResponse, "new_segment"> & {
  new_segment: BackendSegment | null;
  analysis: ChunkAnalysis;
  degraded: boolean;
  all_goals_covered: boolean;
  answer_correct?: boolean | null;
  conversation_over?: boolean;
  turns_used?: number;
};

export type FusionSegment = {
  segment_id: number;
  text: string;
  disturbance: number;
  transfer_delta: number | null;
  quadrant: "blind_spot" | "aware_gap" | "productive_struggle" | "mastery" | "unknown";
};

export type FusionResult = {
  session_id: string;
  per_segment: FusionSegment[];
  quadrant_counts: Record<string, number>;
  calibration_rho: number | null;
};

export type RunResult = {
  session_id: string;
  delta_overall: number;
  survival_rate: number;
  per_question: Array<{ question_id: number; taught_mean: number; cold_mean: number; delta: number }>;
  calibration_rho: number | null;
};

export type AnalysisStatus = {
  session_id: string;
  status: "pending" | "running" | "complete" | "failed";
  error: string | null;
  updated_at: number;
  run?: RunResult | null;
  fusion?: FusionResult | null;
};

export type SessionSnapshot = {
  session_id: string;
  transcript: BackendSegment[];
  analyses: ChunkAnalysis[];
  questions: Array<{ question: TargetedQuestion; answer: string | null; answered_at: number | null }>;
  run: RunResult | null;
  fusion: FusionResult | null;
};

export type BackendHealth = {
  ok: boolean;
  llm_configured: boolean;
  store_backend: string;
};

export function classSessionId(pathId: string, classId: string) {
  return `${pathId}:${classId}`;
}
