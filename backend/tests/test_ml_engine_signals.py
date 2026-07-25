"""Space-A signal logic from ml-service/engine.py, exercised without a GPU.

`ConfusionEngine._build` is a pure function of numpy arrays once Space A/B/C have run, so the two
defects that made every utterance score 1.0 can be pinned here on a CPU laptop — no torch, no
models, no box.

Both were silent failures, which is why they survived: one reported an unmeasurable signal as a
good one, the other could only ever see relative outliers.
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy", reason="pip install -r requirements-dev.txt")

# engine.py pulls in torch/torchaudio at module scope for the real pipeline, and alignment.py
# subclasses nn.Module. None of that is reachable from _build, so stub just enough to import:
# torch for config.py's device pick, and an nn whose layers are inert placeholders.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"

    nn_stub = types.ModuleType("torch.nn")

    class _Module:
        def __init__(self, *args, **kwargs): ...

    for layer in ("Module", "Linear", "MultiheadAttention", "LayerNorm"):
        setattr(nn_stub, layer, type(layer, (_Module,), {}))

    functional_stub = types.ModuleType("torch.nn.functional")
    nn_stub.functional = functional_stub
    torch_stub.nn = nn_stub

    sys.modules.update({
        "torch": torch_stub,
        "torch.nn": nn_stub,
        "torch.nn.functional": functional_stub,
        "torchaudio": types.ModuleType("torchaudio"),
    })

import config as C          # noqa: E402  (ml-service is on sys.path via conftest)
from engine import ConfusionEngine  # noqa: E402
from schemas import COGNITIVE_LOAD, RECALL_FAILURE  # noqa: E402

CLEAR = {"verdict": "ALIGNED", "span": None, "correction": None}


def _words(n: int) -> list[dict]:
    return [{"word": f"word{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(n)]


def _space_a(n: int, *, raw: float, pace_ok: bool, z: np.ndarray | None = None) -> dict:
    """A Space-A result with everything flat unless a test says otherwise."""
    return {
        "z": np.zeros(n) if z is None else z,
        "raw": np.full(n, raw),
        "pace_z": np.zeros(n),
        "entropy": np.zeros(n),
        "pace_ok": pace_ok,
    }


def _build(sa: dict, words: list[dict], red: list[int]):
    return ConfusionEngine._build(
        object.__new__(ConfusionEngine),   # _build touches no instance state
        chunk_id=1, transcript=" ".join(w["word"] for w in words), words=words, sa=sa, red=red,
        logic=CLEAR, fact=CLEAR, history=[], overall_topic="", curriculum_context="",
    )


def test_unmeasurable_pace_is_not_reported_as_even_delivery(monkeypatch):
    """When Whisper's word alignment fails, `_transcribe` splits a segment's duration by word
    length — so seconds-per-character is identical everywhere and pace_z is a row of zeros. That
    must read as "not measured", not as a calm, evenly-paced explanation."""
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    words = _words(6)

    result = _build(_space_a(6, raw=0.1, pace_ok=False), words, red=[])

    assert [a.type for a in result.anomalies] == [], "no pace data means no pace verdict"
    assert all(w.is_bottleneck is False for w in result.detail)


def test_measurable_pace_still_flags_a_bottleneck(monkeypatch):
    """The counterpart: with real timings, a slow word must still surface. Otherwise the fix above
    would have quietly disabled the signal it was meant to protect."""
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    words = _words(6)
    sa = _space_a(6, raw=0.1, pace_ok=True)
    sa["pace_z"] = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])   # word 2 took far longer to say

    result = _build(sa, words, red=[])

    assert COGNITIVE_LOAD in [a.type for a in result.anomalies]
    assert result.detail[2].is_bottleneck is True
    assert result.confidence < 1.0


def test_uniformly_shaky_speech_is_caught_without_any_outlier(monkeypatch):
    """The reported bug. Hedge through a whole sentence and no word stands out, so every z-score
    stays near zero — the relative signal is blind by construction. Absolute dissonance is what
    sees it."""
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    words = _words(8)
    shaky = C.ABSOLUTE_DISSONANCE + 0.2

    result = _build(_space_a(8, raw=shaky, pace_ok=True), words, red=[1, 2, 3])

    assert RECALL_FAILURE in [a.type for a in result.anomalies]
    assert result.confidence < 1.0, "uniform hesitation must not score as a clear explanation"


def test_a_genuinely_clear_explanation_still_scores_high(monkeypatch):
    """Guard against over-correcting: low dissonance, no outliers, real timings -> stays clean."""
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    words = _words(8)

    result = _build(_space_a(8, raw=0.05, pace_ok=True), words, red=[])

    assert result.anomalies == []
    assert result.confidence == 1.0


@pytest.mark.parametrize("raw", [0.0, 0.3, 0.9])
def test_confidence_stays_in_range(monkeypatch, raw):
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    result = _build(_space_a(5, raw=raw, pace_ok=True), _words(5), red=[])
    assert 0.05 <= result.confidence <= 1.0
