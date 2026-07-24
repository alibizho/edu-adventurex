"""Instrument B — speech-disturbance model (report §5). Owned by the ML-speech workstream.

Contract only, for now: given audio aligned to segment ids, return a per-segment disturbance
score D_s in [0,1] (speaker-normalized) plus the feature breakdown. The fusion (report §6)
crosses D_s with the transfer delta Δ_s. Everything keys off the shared Segment.id.
"""
from pydantic import BaseModel, Field


class DisturbanceFeatures(BaseModel):
    filled_pause_rate: float = 0.0
    silent_pause_ratio: float = 0.0     # mid-clause weighted
    repair_count: int = 0
    pitch_uncertainty: float = 0.0      # uptalk + F0 reset
    rate_dip: float = 0.0
    hedge_density: float = 0.0
    dominant_marker: str = "none"


class SegmentDisturbance(BaseModel):
    segment_id: int
    score: float = Field(ge=0.0, le=1.0)   # D_s, speaker-normalized
    features: DisturbanceFeatures = DisturbanceFeatures()


def score_segment(audio_path: str, segment_id: int) -> SegmentDisturbance:
    """STUB. Real impl (report §5.2–5.6): openSMILE/prosody features -> per-speaker z-norm ->
    LightGBM/fusion -> D_s. Returns a neutral score until the model lands."""
    return SegmentDisturbance(segment_id=segment_id, score=0.0)
