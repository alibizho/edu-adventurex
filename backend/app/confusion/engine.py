"""Confusion engine — the Instrument-C model. Owned by the ML
workstream; NOT built yet. `analyze` is the contract the future model fills.

Until the real model lands, `analyze` is a light heuristic MOCK so the targeted-question backend
runs end-to-end: chunks with hedging / uncertainty markers get a low confidence score and a
plausible anomaly; clear chunks score high. Real impl: frozen encoders + retrieval
+ NLI -> per-chunk confidence + typed anomalies.

Contract: confidence in [0,1], HIGH = the speaker sounded clear/confident on the chunk, LOW =
uncertain/confused. The backend targets the LOWEST-confidence chunks.
"""
import re

from ..config import settings
from ..schemas import Anomaly, ChunkAnalysis, SpeechProsody

# Uncertainty markers → lower confidence. Weight is how much each drags confidence down.
_HEDGES = [
    "maybe", "i think", "i guess", "kind of", "sort of", "probably", "possibly",
    "not sure", "um", "uh", "er", "i don't know", "dunno", "or something", "whatever",
    "i mean", "you know", "like,",
]
_HEDGE_ANOMALY = {"type": "hedging", "source": "mock/lexical"}

# Word-boundary hesitation pattern for has_confusion_markers (the real-time gate backstop). Built
# from _HEDGES so common substrings don't fire ("er" in "router", "um" in "column"); "like" is
# matched only as the discourse marker "like," to avoid the ordinary preposition.
_HEDGE_TOKENS = [h for h in _HEDGES if h != "like,"]
_HEDGE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in _HEDGE_TOKENS) + r")\b" + r"|\blike\b\s*,",
    re.IGNORECASE,
)


def _mock_confidence(text: str) -> tuple[float, list[Anomaly]]:
    # Word-boundary matched via the shared _HEDGE_RE so substrings don't fire ("er" in "router",
    # "um" in "column") — same guard has_confusion_markers uses. Dedupe so a repeated marker
    # counts once, matching the original "distinct markers" semantics.
    hits = list(dict.fromkeys(m.strip().lower() for m in _HEDGE_RE.findall(text)))
    if text.strip().endswith("?"):
        hits.append("?")  # a declarative that trails off into a question reads as uncertain
    confidence = max(0.1, 1.0 - 0.22 * len(hits))
    anomalies: list[Anomaly] = []
    if hits:
        anomalies.append(
            Anomaly(type="hedging", source="mock/lexical", score=round(1.0 - confidence, 2),
                    evidence=f"uncertainty markers: {', '.join(h for h in hits if h != '?')[:80] or 'trailing question'}")
        )
    return round(confidence, 2), anomalies


def analyze(chunks: list[str]) -> list[ChunkAnalysis]:
    """STUB for the real confusion model. Heuristic mock — see module docstring."""
    out: list[ChunkAnalysis] = []
    for i, text in enumerate(chunks):
        confidence, anomalies = _mock_confidence(text)
        out.append(ChunkAnalysis(chunk_id=i, text=text, confidence=confidence, anomalies=anomalies))
    return out


def select_low_confidence(
    analyses: list[ChunkAnalysis],
    k: int = 3,
    threshold: float | None = None,
    exclude_ids: set[int] | None = None,
) -> list[ChunkAnalysis]:
    """Return the k lowest-confidence chunks, skipping `exclude_ids` (already covered) and any chunk
    above `threshold` when one is given. Lowest confidence first."""
    exclude_ids = exclude_ids or set()
    pool = [a for a in analyses if a.chunk_id not in exclude_ids]
    if threshold is not None:
        pool = [a for a in pool if a.confidence < threshold]
    pool.sort(key=lambda a: a.confidence)
    return pool[:k]


def has_confusion_markers(text: str) -> bool:
    """Lexical backstop for hesitation/uncertainty (e.g. 'um', 'uh', a trailing '?'). Used by the
    real-time question gate when the ml-service's Space A alignment brain misses obvious hedging.
    Word-boundary matched so 'er' in 'router' or 'um' in 'column' don't fire."""
    if _HEDGE_RE.search(text):
        return True
    return text.strip().endswith("?")


def prosody_confidence(prosody: SpeechProsody, word_count: int) -> float:
    """How clear the delivery sounded, from browser-measured timing alone. [0,1], HIGH = fluent.

    Deliberately model-free. The ml-service scores hesitation relative to the rest of the same
    utterance, which cannot see speech that is unsure throughout; dead air and stalls are absolute
    and mean the same thing regardless of what surrounds them.
    """
    if prosody.total_ms <= 0:
        return 1.0

    conf = 1.0
    speech_ratio = prosody.speech_ms / prosody.total_ms
    # Long gaps between words: thinking mid-sentence, not a considered pause between them.
    if speech_ratio < 0.65:
        conf -= min(0.4, (0.65 - speech_ratio) * 1.6)

    # Repeated stalls. One is a breath; several in one utterance is someone assembling an
    # explanation they don't have ready.
    if prosody.pause_count >= 2:
        conf -= min(0.3, (prosody.pause_count - 1) * 0.12)

    # A single long freeze is its own signal even when everything else flows.
    if prosody.longest_pause_ms >= 1200:
        conf -= min(0.25, (prosody.longest_pause_ms - 1200) / 4000 + 0.1)

    # Trailing off mid-explanation — the words are there but the voice gives up on them.
    if prosody.peak_level > 0 and prosody.mean_level / prosody.peak_level < 0.18:
        conf -= 0.1

    # Speaking rate, only once there are enough words for it to mean anything.
    voiced_seconds = prosody.speech_ms / 1000
    if word_count >= 6 and voiced_seconds > 1:
        words_per_second = word_count / voiced_seconds
        if words_per_second < 1.6:
            conf -= min(0.2, (1.6 - words_per_second) * 0.25)

    return round(max(0.05, min(1.0, conf)), 2)


def fuse_prosody(analysis: ChunkAnalysis, prosody: SpeechProsody | None) -> ChunkAnalysis:
    """Combine the GPU's confidence with the browser's, pessimistically.

    `min` rather than a weighted blend on purpose: one signal is uncalibrated (the cosine
    dissonance scale depends on the trained brain) and the other is a direct measurement. Taking
    whichever is more worried is explainable, cannot be tuned into nonsense, and fails toward
    asking the learner a question — which is the cheap mistake. The ml-service's own value is kept
    on `gpu_confidence` so the two can be compared while ABSOLUTE_DISSONANCE is calibrated.
    """
    if prosody is None:
        return analysis
    analysis.prosody = prosody
    analysis.gpu_confidence = analysis.confidence
    analysis.confidence = min(
        analysis.confidence, prosody_confidence(prosody, len(analysis.text.split()))
    )
    return analysis


def is_confused(analysis: ChunkAnalysis) -> bool:
    """The shared confusion gate for both the real-time endpoint (/questions/from_chunk) and the
    plan teaching turn. A chunk is 'confused' when anomaly-gating is on AND it has an anomaly, OR
    its confidence is below the threshold, OR it carries a lexical hesitation marker. Keep both
    call sites on this one definition so they can't drift."""
    return (
        (settings.question_gate_on_anomalies and bool(analysis.anomalies))
        or analysis.confidence < settings.question_confidence_threshold
        or has_confusion_markers(analysis.text)
    )
