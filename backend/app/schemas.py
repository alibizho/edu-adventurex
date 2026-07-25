"""Shared data model. Everything downstream keys off `Segment.id` — do not change that
contract without a heads-up (see README)."""
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Segment(BaseModel):
    id: int                       # stable primary key — attribution + speech both cite this
    idx: int                      # order within the session
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
    answer_key: Optional[str] = None         # ground-truth answer the verifier grades against
    cold_pass_rate: Optional[float] = None   # set by the filter
    survived: Optional[bool] = None          # survived the filter?


class Arm(str, Enum):
    taught = "taught"   # persona heard the transcript
    cold = "cold"       # persona heard nothing (control)


class Score(BaseModel):
    question_id: int
    arm: Arm
    persona_seed: str
    correct: bool
    not_covered: bool = False    # persona said NOT COVERED — a gap, not a wrong answer
    cited_segment_ids: list[int] = Field(default_factory=list)


class QuestionDelta(BaseModel):
    question_id: int
    taught_mean: float
    cold_mean: float
    delta: float            # taught_mean - cold_mean; < 0 == teaching made it worse


class RunResult(BaseModel):
    session_id: str
    delta_overall: float
    survival_rate: float
    per_question: list[QuestionDelta]
    calibration_rho: Optional[float] = None   # filled once Instrument B lands


# ---- teaching-loop I/O ----

class TeachTurnRequest(BaseModel):
    session_id: str
    transcript: list[Segment]
    latest_utterance: str


class TeachTurnResponse(BaseModel):
    student_reply: str
    new_segment: Segment


# ---- confusion engine output (the contract the ml-service fills) ----

class Anomaly(BaseModel):
    type: str                       # e.g. "factual_error", "recall_failure", "logic_error", "hedging"
    source: str = ""                # which detector flagged it (free text)
    score: float = 0.0              # detector confidence for THIS anomaly, [0,1]
    evidence: Optional[str] = None  # ground-truth / context that triggered the flag


class WordScore(BaseModel):
    """Per-word Space-A detail from the ml-service (mock leaves this empty)."""
    word: str
    hesitation_zscore: float = 0.0
    is_anomaly: bool = False
    pace_zscore: float = 0.0
    is_bottleneck: bool = False
    attention_entropy: float = 0.0
    is_scattered: bool = False


class StudentQuestion(BaseModel):
    """Question produced on the GPU from the strongest detected anomaly."""
    question_text: str
    target_concept: str
    anomaly_type: str


class CurriculumUpdate(BaseModel):
    """New, valid concepts introduced beyond the current curriculum context."""
    added_concepts: list[str] = Field(default_factory=list)


class SpeechProsody(BaseModel):
    """How the utterance sounded, measured in the browser from the same RMS blocks the recorder's
    voice-activity detection already runs on (see useContinuousRecorder.ts).

    It exists because the ml-service's hesitation score is z-scored *within* an utterance, so it
    cannot see speech that is unsure the whole way through — hedge from start to finish and no
    single word is an outlier. Pauses and dead air are absolute and need no model."""
    speech_ms: int = 0
    total_ms: int = 0
    pause_count: int = 0            # internal stalls over ~350ms that didn't end the utterance
    longest_pause_ms: int = 0
    mean_level: float = 0.0
    peak_level: float = 0.0


class ChunkAnalysis(BaseModel):
    chunk_id: int                   # aligns with the segment/clause spine when both exist
    text: str
    confidence: float               # [0,1]; HIGH = speaker sounded clear, LOW = uncertain/confused
    anomalies: list[Anomaly] = Field(default_factory=list)
    localized_target: Optional[str] = None   # the specific word that broke, if any (ml-service)
    detail: list[WordScore] = Field(default_factory=list)   # per-word Space-A detail (ml-service)
    student_question: Optional[StudentQuestion] = None
    curriculum_update: Optional[CurriculumUpdate] = None
    prosody: Optional[SpeechProsody] = None
    # What the ml-service alone reported, kept after prosody fusion so the two can be compared
    # while ABSOLUTE_DISSONANCE is being calibrated.
    gpu_confidence: Optional[float] = None


# ---- fusion: confidence x competence (report §6) ----

class SegmentFusion(BaseModel):
    """One segment crossed by both instruments."""
    segment_id: int
    text: str
    disturbance: float                      # 1 - confidence, from the confusion engine [0,1]
    transfer_delta: Optional[float] = None  # mean delta over questions citing this segment; None if none
    quadrant: str                           # blind_spot | aware_gap | productive_struggle | mastery | unknown


class FusionResult(BaseModel):
    session_id: str
    per_segment: list[SegmentFusion]
    quadrant_counts: dict[str, int]
    # Pearson(disturbance, -delta) across segments. Positive == well-calibrated: the kid sounded
    # unsure exactly where the teaching failed to transmit (report §6.2). None if too few points.
    calibration_rho: Optional[float] = None


# ---- targeted-question agent + memory ----

class TargetedQuestion(BaseModel):
    id: int
    chunk_id: int                   # the low-confidence chunk this question probes
    text: str
    anomaly_type: Optional[str] = None
    rationale: Optional[str] = None  # why the agent asked this (for debugging / UI)
    # Ground truth for pipeline.grading.grade_answer. Without it an answer can only be checked for
    # existence, which is how "uhh I dunno" used to count as understanding.
    answer_key: Optional[str] = None
    # The question this one follows up on, so a back-and-forth about one concept stays linked.
    parent_id: Optional[int] = None


class QAEntry(BaseModel):
    question: TargetedQuestion
    answer: Optional[str] = None
    answered_at: Optional[float] = None


class ChunkQuestionResponse(BaseModel):
    """Real-time per-chunk result: the confusion analysis of the just-spoken chunk plus a question
    generated only when the chunk was confused (else `asked=False`, `question=None`)."""
    asked: bool
    analysis: ChunkAnalysis
    question: Optional[TargetedQuestion] = None


# ---- learning plan: growth path + classes + cross-class memory ----

class ClassObjective(BaseModel):
    """One checkable thing the learner must be able to explain. The class is passed by covering
    these, not by talking for long enough."""
    id: str                             # "o1", stable within the class
    text: str                           # "Explain why force and acceleration are proportional"


class ClassUnit(BaseModel):
    """One class = one topic the learner must understand by teaching it. No subtopics — a class is
    the unit of teaching. Teacher's notes are written up-front at /plan/build, against the whole
    outline, so no two classes cover the same ground."""
    class_id: str
    title: str                          # "Newton's Laws of Motion" — the class IS the topic
    objective: str                      # one-sentence learning goal
    objectives: list[ClassObjective] = Field(default_factory=list)  # the checkable breakdown
    difficulty: str = "beginner"        # beginner | intermediate | advanced
    prerequisites: list[str] = Field(default_factory=list)  # class_ids that should come first
    teacher_notes: str = ""             # Markdown primer, written at build (may embed ```mermaid)
    notes_generated: bool = False

    def checklist(self) -> list[ClassObjective]:
        """Objectives to grade against. Plans built before objectives existed fall back to the
        single one-sentence goal, so old paths keep working instead of showing an empty class."""
        return self.objectives or [ClassObjective(id="o1", text=self.objective)]


class GrowthPath(BaseModel):
    """The learner's teaching plan: a confirmed topic broken into ~5 ordered classes."""
    path_id: str
    original_input: str                 # "I want to learn physics"
    confirmed_topic: str                # "Classical Mechanics: Forces and Motion"
    total_classes: int
    recommended_order: list[str] = Field(default_factory=list)  # class_ids in sequence
    classes: list[ClassUnit] = Field(default_factory=list)
    source_material_summary: Optional[str] = None   # short summary of any pasted material


class ScopeSuggestion(BaseModel):
    topic: str
    rationale: str
    suggested_classes: int


class TopicScope(BaseModel):
    """Result of scoping the learner's request. If too broad, offer 3 narrower alternatives."""
    is_broad: bool
    suggestions: list[ScopeSuggestion] = Field(default_factory=list)
    confirmed_topic: str
    suggested_classes: int


class ClassProgressRecord(BaseModel):
    status: Literal["not_started", "in_progress", "complete"] = "not_started"
    # How many times the learner has thrown this class away and started it over. Not a statistic:
    # a teaching turn or coverage check already in flight when that happened compares this against
    # the value it read, and drops its write rather than resurrecting a deleted lesson.
    reset_count: int = 0
    readiness: int = 0                  # % of this class's objectives actually covered
    turn_count: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    completion_mode: Optional[Literal["self-teaching", "guided-explanation"]] = None
    analysis_status: Literal["not_started", "pending", "running", "complete", "failed"] = "not_started"
    analysis_error: Optional[str] = None

    # --- objective mastery ---
    covered_objectives: list[str] = Field(default_factory=list)   # ClassObjective ids
    # id -> the sentence that earned it. Keeps the checkmark accountable: the summary can show
    # what the learner actually said instead of asking them to trust a tick.
    objective_evidence: dict[str, str] = Field(default_factory=dict)
    # Highest segment id already judged, so a background check never re-grades old speech.
    last_checked_segment: int = -1
    last_goal_probe_turn: int = -1      # throttles "you haven't covered X yet" nudges
    passed_on_mastery: bool = False     # ended having covered everything, vs just stopped
    # Times the student stopped asking and told the learner the answer (they said "I don't know",
    # or ran out of tries). Any of these makes the class `guided-explanation` rather than
    # self-teaching, which is the difference between working it out and being told.
    explanations_given: int = 0

    # --- struggle ledger ("Feynman ledger") ---
    # concept -> cumulative struggle score, accumulated from anomaly scores on the word the
    # ml-service localized. Objectives above are what the CLASS wants taught; this is what the
    # LEARNER actually keeps tripping over, which is not the same list and is the one worth
    # chasing. Per class on purpose: a weakness in class 3 should not steer class 5.
    struggle_scores: dict[str, float] = Field(default_factory=dict)
    # Current argmax of struggle_scores, passed to the ml-service so the AI student keeps probing
    # the same weak spot instead of drifting. "" when nothing is outstanding.
    focus_target: str = ""


class PathMemory(BaseModel):
    """Durable cross-class memory so the AI doesn't re-teach or re-ask across classes. Keyed by
    path_id (spans every class in the plan)."""
    path_id: str
    class_progress: dict[str, ClassProgressRecord] = Field(default_factory=dict)
    covered_concepts: list[str] = Field(default_factory=list)  # already taught → don't re-teach
    asked_questions: list[str] = Field(default_factory=list)   # already asked → no near-duplicates
    understood: list[str] = Field(default_factory=list)        # learner got it → skip
    struggled: list[str] = Field(default_factory=list)         # didn't get it → OK to re-probe
    expanded_concepts: list[str] = Field(default_factory=list)  # valid beyond-scope concepts found live


# ---- learning-plan API I/O ----

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


class ClassTeachResponse(BaseModel):
    student_reply: str
    new_segment: Segment
    asked: bool = False
    question: Optional[TargetedQuestion] = None


class AudioClassTeachResponse(BaseModel):
    """One browser-recorded teaching turn, including the GPU analysis when available."""
    student_reply: str = ""
    new_segment: Optional[Segment] = None
    analysis: ChunkAnalysis
    asked: bool = False
    question: Optional[TargetedQuestion] = None
    degraded: bool = False
    # Nothing left to teach in this class. Lets the UI say "the class is following" instead of
    # leaving a silent turn ambiguous between "you're doing well" and "it's broken".
    all_goals_covered: bool = False

    # --- one-to-one conversation (set only when answering a specific student's question) ---
    # Whether the answer actually conveyed the question's answer key. None when nothing was being
    # answered, or when there was no key to grade against.
    answer_correct: Optional[bool] = None
    # True when the student is done with this question — either satisfied, or out of follow-ups.
    conversation_over: bool = False
    # How many times the learner has now answered this thread; drives the turn cap in the UI.
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


# ---- API I/O ----

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
