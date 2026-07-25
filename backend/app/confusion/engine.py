import re

from ..config import settings
from ..schemas import Anomaly, ChunkAnalysis, SpeechProsody

_HEDGES = [
    "maybe", "i think", "i guess", "kind of", "sort of", "probably", "possibly",
    "not sure", "um", "uh", "er", "i don't know", "dunno", "or something", "whatever",
    "i mean", "you know", "like,",
]
_HEDGE_ANOMALY = {"type": "hedging", "source": "mock/lexical"}

_HEDGE_TOKENS = [h for h in _HEDGES if h != "like,"]
_HEDGE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in _HEDGE_TOKENS) + r")\b" + r"|\blike\b\s*,",
    re.IGNORECASE,
)

def _mock_confidence(text: str) -> tuple[float, list[Anomaly]]:
    hits = list(dict.fromkeys(m.strip().lower() for m in _HEDGE_RE.findall(text)))
    if text.strip().endswith("?"):
        hits.append("?")
    confidence = max(0.1, 1.0 - 0.22 * len(hits))
    anomalies: list[Anomaly] = []
    if hits:
        anomalies.append(
            Anomaly(type="hedging", source="mock/lexical", score=round(1.0 - confidence, 2),
                    evidence=f"uncertainty markers: {', '.join(h for h in hits if h != '?')[:80] or 'trailing question'}")
        )
    return round(confidence, 2), anomalies

def analyze(chunks: list[str]) -> list[ChunkAnalysis]:
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
    exclude_ids = exclude_ids or set()
    pool = [a for a in analyses if a.chunk_id not in exclude_ids]
    if threshold is not None:
        pool = [a for a in pool if a.confidence < threshold]
    pool.sort(key=lambda a: a.confidence)
    return pool[:k]

def has_confusion_markers(text: str) -> bool:
    if _HEDGE_RE.search(text):
        return True
    return text.strip().endswith("?")

def prosody_confidence(prosody: SpeechProsody, word_count: int) -> float:
    if prosody.total_ms <= 0:
        return 1.0

    conf = 1.0
    speech_ratio = prosody.speech_ms / prosody.total_ms
    if speech_ratio < 0.65:
        conf -= min(0.4, (0.65 - speech_ratio) * 1.6)

    if prosody.pause_count >= 2:
        conf -= min(0.3, (prosody.pause_count - 1) * 0.12)

    if prosody.longest_pause_ms >= 1200:
        conf -= min(0.25, (prosody.longest_pause_ms - 1200) / 4000 + 0.1)

    if prosody.peak_level > 0 and prosody.mean_level / prosody.peak_level < 0.18:
        conf -= 0.1

    voiced_seconds = prosody.speech_ms / 1000
    if word_count >= 6 and voiced_seconds > 1:
        words_per_second = word_count / voiced_seconds
        if words_per_second < 1.6:
            conf -= min(0.2, (1.6 - words_per_second) * 0.25)

    return round(max(0.05, min(1.0, conf)), 2)

def fuse_prosody(analysis: ChunkAnalysis, prosody: SpeechProsody | None) -> ChunkAnalysis:
    if prosody is None:
        return analysis
    analysis.prosody = prosody
    analysis.gpu_confidence = analysis.confidence
    analysis.confidence = min(
        analysis.confidence, prosody_confidence(prosody, len(analysis.text.split()))
    )
    return analysis

def is_confused(analysis: ChunkAnalysis) -> bool:
    return (
        (settings.question_gate_on_anomalies and bool(analysis.anomalies))
        or analysis.confidence < settings.question_confidence_threshold
        or has_confusion_markers(analysis.text)
    )
