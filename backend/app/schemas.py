"""Shared data model. Everything downstream keys off `Segment.id` — do not change that
contract without a heads-up (see README). Mirrors TECHNICAL_REPORT.md §7."""
from enum import Enum
from typing import Optional

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


# ---- confusion engine output (the contract the future ML fills; see Divay.MD §6) ----

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


class ChunkAnalysis(BaseModel):
    chunk_id: int                   # aligns with the segment/clause spine when both exist
    text: str
    confidence: float               # [0,1]; HIGH = speaker sounded clear, LOW = uncertain/confused
    anomalies: list[Anomaly] = Field(default_factory=list)
    localized_target: Optional[str] = None   # the specific word that broke, if any (ml-service)
    detail: list[WordScore] = Field(default_factory=list)   # per-word Space-A detail (ml-service)


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
