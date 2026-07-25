from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy", reason="pip install -r requirements-dev.txt")

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

import config as C
from engine import ConfusionEngine
from schemas import COGNITIVE_LOAD, RECALL_FAILURE

CLEAR = {"verdict": "ALIGNED", "span": None, "correction": None}

def _words(n: int) -> list[dict]:
    return [{"word": f"word{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(n)]

def _space_a(n: int, *, raw: float, pace_ok: bool, z: np.ndarray | None = None) -> dict:
    return {
        "z": np.zeros(n) if z is None else z,
        "raw": np.full(n, raw),
        "pace_z": np.zeros(n),
        "entropy": np.zeros(n),
        "pace_ok": pace_ok,
    }

def _build(sa: dict, words: list[dict], red: list[int]):
    return ConfusionEngine._build(
        object.__new__(ConfusionEngine),
        chunk_id=1, transcript=" ".join(w["word"] for w in words), words=words, sa=sa, red=red,
        logic=CLEAR, fact=CLEAR, history=[], overall_topic="", curriculum_context="",
    )

def test_unmeasurable_pace_is_not_reported_as_even_delivery(monkeypatch):
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    words = _words(6)

    result = _build(_space_a(6, raw=0.1, pace_ok=False), words, red=[])

    assert [a.type for a in result.anomalies] == [], "no pace data means no pace verdict"
    assert all(w.is_bottleneck is False for w in result.detail)

def test_measurable_pace_still_flags_a_bottleneck(monkeypatch):
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    words = _words(6)
    sa = _space_a(6, raw=0.1, pace_ok=True)
    sa["pace_z"] = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])

    result = _build(sa, words, red=[])

    assert COGNITIVE_LOAD in [a.type for a in result.anomalies]
    assert result.detail[2].is_bottleneck is True
    assert result.confidence < 1.0

def test_uniformly_shaky_speech_is_caught_without_any_outlier(monkeypatch):
    monkeypatch.setattr(ConfusionEngine, "_generate_student_question", lambda *a, **k: None)
    words = _words(8)
    shaky = C.ABSOLUTE_DISSONANCE + 0.2

    result = _build(_space_a(8, raw=shaky, pace_ok=True), words, red=[1, 2, 3])

    assert RECALL_FAILURE in [a.type for a in result.anomalies]
    assert result.confidence < 1.0, "uniform hesitation must not score as a clear explanation"

def test_a_genuinely_clear_explanation_still_scores_high(monkeypatch):
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
