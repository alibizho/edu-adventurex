"""Response contract. Mirrors backend/app/schemas.py (Anomaly / WordScore / StudentQuestion /
CurriculumUpdate / ChunkAnalysis) so the output drops straight into the backend confusion engine.
`detail` carries the rich word-level data.

The two services deploy separately (backend on CPU, this on a rented GPU box) and share no package,
so backend/tests/test_ml_service_contract.py asserts these models field-for-field against
app/schemas.py. Change one side, change the other.
"""
from typing import Optional

from pydantic import BaseModel, Field

# Anomaly type strings the backend expects (snake_case), keyed by which space raised them.
RECALL_FAILURE = "recall_failure"   # Space A: audio/text mismatch (hesitation)
LOGIC_ERROR = "logic_error"         # Space B: text/text self-contradiction
FACTUAL_ERROR = "factual_error"     # Space C: text/knowledge mismatch
HEDGING = "hedging"                 # lexical uncertainty markers (raised by the backend's mock)
COGNITIVE_LOAD = "cognitive_load"   # slow articulation / scattered alignment (processing effort)
# Cross-modal + curriculum-grounded types. These need the request's curriculum context to fire.
FLUENCY_ISSUE = "fluency_issue"     # text is correct but the audio says it cost real effort
OFF_TOPIC = "off_topic"             # drifted away from the class being taught
BEYOND = "beyond"                   # correct AND past the curriculum -> grows it, not an error


class Anomaly(BaseModel):
    type: str
    source: str = ""
    score: float = 0.0
    evidence: Optional[str] = None


class WordScore(BaseModel):
    word: str
    hesitation_zscore: float = 0.0    # Space A audio/text dissonance (z)
    is_anomaly: bool = False
    pace_zscore: float = 0.0          # articulation speed (z); high = took abnormally long to say
    is_bottleneck: bool = False       # pace_zscore over threshold -> cognitive bottleneck
    attention_entropy: float = 0.0    # how scattered the audio->text alignment was for this word
    is_scattered: bool = False        # entropy well above the utterance mean -> unstable alignment


class StudentQuestion(BaseModel):
    """Question produced on the GPU from the strongest detected anomaly."""
    question_text: str
    target_concept: str
    anomaly_type: str


class CurriculumUpdate(BaseModel):
    """New, valid concepts introduced beyond the current curriculum context."""
    added_concepts: list[str] = Field(default_factory=list)


class ChunkAnalysis(BaseModel):
    """One utterance = one chunk on the shared segment spine."""
    chunk_id: int
    text: str                                   # the transcript
    confidence: float                           # [0,1]; HIGH = sounded clear, LOW = confused
    anomalies: list[Anomaly] = Field(default_factory=list)
    localized_target: Optional[str] = None      # the specific word that broke, if any
    detail: list[WordScore] = Field(default_factory=list)   # per-word Space-A detail
    student_question: Optional[StudentQuestion] = None      # the AI student's interruption
    curriculum_update: Optional[CurriculumUpdate] = None    # concepts to fold into the class


class AnalyzeRequest(BaseModel):
    session_id: str = "default"
    chunk_id: int = 0
    history: list[str] = Field(default_factory=list)   # prior transcripts (stateless context)
    enable_space_c: Optional[bool] = None              # override the server default per call
    # Curriculum grounding. Without these, Space C fact-checks against the judge's own knowledge
    # and no student question is written; with them, both are anchored to what's being taught.
    overall_topic: str = ""
    curriculum_context: str = ""
    key_concepts: list[str] = Field(default_factory=list)
