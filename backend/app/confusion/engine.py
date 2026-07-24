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

from ..schemas import Anomaly, ChunkAnalysis

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
