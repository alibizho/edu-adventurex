from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

class Segment(BaseModel):
    id: int
    idx: int
    text: str
    t_start: Optional[float] = None
    t_end: Optional[float] = None

class QuestionKind(str, Enum):
    transfer = "transfer"
    recall = "recall"

class Question(BaseModel):
    id: int
    text: str
    kind: QuestionKind = QuestionKind.transfer
    answer_key: Optional[str] = None
    cold_pass_rate: Optional[float] = None
    survived: Optional[bool] = None

class Arm(str, Enum):
    taught = "taught"
    cold = "cold"

class Score(BaseModel):
    question_id: int
    arm: Arm
    persona_seed: str
    correct: bool
    not_covered: bool = False
    cited_segment_ids: list[int] = Field(default_factory=list)

class QuestionDelta(BaseModel):
    question_id: int
    taught_mean: float
    cold_mean: float
    delta: float

class RunResult(BaseModel):
    session_id: str
    delta_overall: float
    survival_rate: float
    per_question: list[QuestionDelta]
    calibration_rho: Optional[float] = None

class TeachTurnRequest(BaseModel):
    session_id: str
    transcript: list[Segment]
    latest_utterance: str

class TeachTurnResponse(BaseModel):
    student_reply: str
    new_segment: Segment

class Anomaly(BaseModel):
    type: str
    source: str = ""
    score: float = 0.0
    evidence: Optional[str] = None

class WordScore(BaseModel):
    word: str
    hesitation_zscore: float = 0.0
    is_anomaly: bool = False
    pace_zscore: float = 0.0
    is_bottleneck: bool = False
    attention_entropy: float = 0.0
    is_scattered: bool = False

class StudentQuestion(BaseModel):
    question_text: str
    target_concept: str
    anomaly_type: str

class CurriculumUpdate(BaseModel):
    added_concepts: list[str] = Field(default_factory=list)

class SpeechProsody(BaseModel):
    speech_ms: int = 0
    total_ms: int = 0
    pause_count: int = 0
    longest_pause_ms: int = 0
    mean_level: float = 0.0
    peak_level: float = 0.0

class ChunkAnalysis(BaseModel):
    chunk_id: int
    text: str
    confidence: float
    anomalies: list[Anomaly] = Field(default_factory=list)
    localized_target: Optional[str] = None
    detail: list[WordScore] = Field(default_factory=list)
    student_question: Optional[StudentQuestion] = None
    curriculum_update: Optional[CurriculumUpdate] = None
    prosody: Optional[SpeechProsody] = None
    gpu_confidence: Optional[float] = None

class SegmentFusion(BaseModel):
    segment_id: int
    text: str
    disturbance: float
    transfer_delta: Optional[float] = None
    quadrant: str

class FusionResult(BaseModel):
    session_id: str
    per_segment: list[SegmentFusion]
    quadrant_counts: dict[str, int]
    calibration_rho: Optional[float] = None

class TargetedQuestion(BaseModel):
    id: int
    chunk_id: int
    text: str
    anomaly_type: Optional[str] = None
    rationale: Optional[str] = None
    answer_key: Optional[str] = None
    parent_id: Optional[int] = None

class QAEntry(BaseModel):
    question: TargetedQuestion
    answer: Optional[str] = None
    answered_at: Optional[float] = None

class ChunkQuestionResponse(BaseModel):
    asked: bool
    analysis: ChunkAnalysis
    question: Optional[TargetedQuestion] = None

class ClassObjective(BaseModel):
    id: str
    text: str

class ClassUnit(BaseModel):
    class_id: str
    title: str
    objective: str
    objectives: list[ClassObjective] = Field(default_factory=list)
    difficulty: str = "beginner"
    prerequisites: list[str] = Field(default_factory=list)
    teacher_notes: str = ""
    notes_generated: bool = False

    def checklist(self) -> list[ClassObjective]:
        return self.objectives or [ClassObjective(id="o1", text=self.objective)]

class GrowthPath(BaseModel):
    path_id: str
    original_input: str
    confirmed_topic: str
    total_classes: int
    recommended_order: list[str] = Field(default_factory=list)
    classes: list[ClassUnit] = Field(default_factory=list)
    source_material_summary: Optional[str] = None

class ScopeSuggestion(BaseModel):
    topic: str
    rationale: str
    suggested_classes: int

class TopicScope(BaseModel):
    is_broad: bool
    suggestions: list[ScopeSuggestion] = Field(default_factory=list)
    confirmed_topic: str
    suggested_classes: int

class ClassProgressRecord(BaseModel):
    status: Literal["not_started", "in_progress", "complete"] = "not_started"
    reset_count: int = 0
    readiness: int = 0
    turn_count: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    completion_mode: Optional[Literal["self-teaching", "guided-explanation"]] = None
    analysis_status: Literal["not_started", "pending", "running", "complete", "failed"] = "not_started"
    analysis_error: Optional[str] = None

    covered_objectives: list[str] = Field(default_factory=list)
    objective_evidence: dict[str, str] = Field(default_factory=dict)
    last_checked_segment: int = -1
    last_goal_probe_turn: int = -1
    passed_on_mastery: bool = False
    explanations_given: int = 0

    struggle_scores: dict[str, float] = Field(default_factory=dict)
    focus_target: str = ""

class PathMemory(BaseModel):
    path_id: str
    class_progress: dict[str, ClassProgressRecord] = Field(default_factory=dict)
    covered_concepts: list[str] = Field(default_factory=list)
    asked_questions: list[str] = Field(default_factory=list)
    understood: list[str] = Field(default_factory=list)
    struggled: list[str] = Field(default_factory=list)
    expanded_concepts: list[str] = Field(default_factory=list)

class ScopeRequest(BaseModel):
    original_input: str
    material_text: Optional[str] = None
    preferred_classes: Optional[int] = None

class BuildPlanRequest(BaseModel):
    original_input: str
    confirmed_topic: str
    num_classes: Optional[int] = None
    material_text: Optional[str] = None

class TeachTurnBody(BaseModel):
    latest_utterance: str

_EXPLAINED = "Whether student_reply is an explanation handed over after the teacher gave up."

class ClassTeachResponse(BaseModel):
    student_reply: str
    new_segment: Segment
    asked: bool = False
    question: Optional[TargetedQuestion] = None
    explained: bool = Field(default=False, description=_EXPLAINED)

class AudioClassTeachResponse(BaseModel):
    student_reply: str = ""
    new_segment: Optional[Segment] = None
    analysis: ChunkAnalysis
    asked: bool = False
    question: Optional[TargetedQuestion] = None
    degraded: bool = False
    explained: bool = Field(default=False, description=_EXPLAINED)
    all_goals_covered: bool = False

    answer_correct: Optional[bool] = None
    conversation_over: bool = False
    turns_used: int = 0

class EndClassRequest(BaseModel):
    completion_mode: Literal["self-teaching", "guided-explanation"] = "self-teaching"

class MaterialFileSummary(BaseModel):
    name: str
    media_type: str
    size: int
    extracted_characters: int

class MaterialExtractionResponse(BaseModel):
    material_text: str
    files: list[MaterialFileSummary]
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False

class AnalysisJob(BaseModel):
    session_id: str
    status: Literal["pending", "running", "complete", "failed"]
    error: Optional[str] = None
    updated_at: float

class AnalysisStatusResponse(AnalysisJob):
    run: Optional[RunResult] = None
    fusion: Optional[FusionResult] = None

class SessionSnapshot(BaseModel):
    session_id: str
    transcript: list[Segment] = Field(default_factory=list)
    analyses: list[ChunkAnalysis] = Field(default_factory=list)
    questions: list[QAEntry] = Field(default_factory=list)
    run: Optional[RunResult] = None
    fusion: Optional[FusionResult] = None

class IngestRequest(BaseModel):
    session_id: str
    chunks: list[ChunkAnalysis]

class NextQuestionsRequest(BaseModel):
    session_id: str
    n: int = 3

class AnswerRequest(BaseModel):
    session_id: str
    question_id: int
    answer: str
