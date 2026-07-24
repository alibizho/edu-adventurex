"""Response contract. Mirrors backend/app/schemas.py (Anomaly / ChunkAnalysis) so the output
drops straight into the backend confusion engine. `detail` carries the rich word-level data."""
from typing import Optional

from pydantic import BaseModel, Field

# Anomaly type strings the backend expects (snake_case), keyed by which space raised them.
RECALL_FAILURE = "recall_failure"   # Space A: audio/text mismatch (hesitation)
LOGIC_ERROR = "logic_error"         # Space B: text/text self-contradiction
FACTUAL_ERROR = "factual_error"     # Space C: text/knowledge mismatch
HEDGING = "hedging"                 # lexical uncertainty markers
COGNITIVE_LOAD = "cognitive_load"   # slow articulation / scattered alignment (processing effort)


class Anomaly(BaseModel):
    type: str
    source: str = ""
    score: float = 0.0
    evidence: Optional[str] = None


class WordScore(BaseModel):
    word: str
    hesitation_zscore: float          # Space A audio/text dissonance (z)
    is_anomaly: bool
    pace_zscore: float = 0.0          # articulation speed (z); high = took abnormally long to say
    is_bottleneck: bool = False       # pace_zscore over threshold -> cognitive bottleneck
    attention_entropy: float = 0.0    # how scattered the audio->text alignment was for this word
    is_scattered: bool = False        # entropy well above the utterance mean -> unstable alignment


class ChunkAnalysis(BaseModel):
    """One utterance = one chunk on the shared segment spine."""
    chunk_id: int
    text: str                                   # the transcript
    confidence: float                           # [0,1]; HIGH = sounded clear, LOW = confused
    anomalies: list[Anomaly] = Field(default_factory=list)
    localized_target: Optional[str] = None      # the specific word that broke, if any
    detail: list[WordScore] = Field(default_factory=list)   # per-word Space-A detail


class AnalyzeRequest(BaseModel):
    session_id: str = "default"
    chunk_id: int = 0
    history: list[str] = Field(default_factory=list)   # prior transcripts (stateless context)
    enable_space_c: Optional[bool] = None              # override the server default per call
