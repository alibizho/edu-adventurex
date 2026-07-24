"""Offline backend tests: pure logic (confusion engine, fusion) + the HTTP surface via
FastAPI's TestClient, backed by the in-memory store. Fully hermetic — no LLM, no ml-service,
no Postgres. Run:  .venv/bin/python -m pytest tests/test_backend.py -q

Config is forced hermetic (memory store, unreachable ml-service) BEFORE the app is imported,
so the confusion client degrades to a neutral analysis instead of hitting a real GPU box.
"""
import asyncio
import os

os.environ["STORE_BACKEND"] = "memory"
os.environ["ML_SERVICE_URL"] = "http://127.0.0.1:9"  # unreachable -> neutral degrade

from fastapi.testclient import TestClient  # noqa: E402

from app.confusion import engine  # noqa: E402
from app.fusion import DISTURBANCE_HIGH, fuse  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import (  # noqa: E402
    Arm,
    ChunkAnalysis,
    QuestionDelta,
    Score,
    Segment,
    TargetedQuestion,
)
from app.store import store  # noqa: E402

client = TestClient(app)


def run(coro):
    """Drive a store coroutine to completion (memory store does no real I/O)."""
    return asyncio.new_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------- #
# confusion engine (the mock Instrument-C heuristic)
# --------------------------------------------------------------------------- #

def test_analyze_flags_hedging_low_confidence():
    (clear, hedged) = engine.analyze(
        ["Two plus two equals four.",
         "It's maybe kind of like, um, energy or something?"]
    )
    assert clear.confidence > 0.9
    assert clear.anomalies == []
    assert hedged.confidence < clear.confidence
    assert any(a.type == "hedging" for a in hedged.anomalies)
    # chunk_id tracks position in the input list
    assert [clear.chunk_id, hedged.chunk_id] == [0, 1]


def test_analyze_no_falsepositive_on_substrings():
    """Regression: _mock_confidence must word-boundary match, not substring match, so clear speech
    ('er' in 'router', 'um' in 'column') isn't flagged as hedging. analyze must agree with
    has_confusion_markers (both use _HEDGE_RE)."""
    for text in ["The router forwards packets.", "Numbers are always positive.",
                 "Select the column here."]:
        (a,) = engine.analyze([text])
        assert a.confidence > 0.9, f"clear sentence flagged: {text!r} -> {a.confidence}"
        assert a.anomalies == []
        assert engine.has_confusion_markers(text) is False


def test_analyze_trailing_question_reads_uncertain():
    (a,) = engine.analyze(["So the mitochondria is the powerhouse?"])
    assert a.confidence < 1.0


def test_has_confusion_markers_word_boundary():
    assert engine.has_confusion_markers("um, I think so")          # hedge token
    assert engine.has_confusion_markers("Is that right?")           # trailing question
    assert not engine.has_confusion_markers("The router forwards packets.")  # 'er' in router
    assert not engine.has_confusion_markers("Select the column here.")       # 'um' in column
    assert not engine.has_confusion_markers("A clear declarative sentence.")


def test_select_low_confidence_orders_and_excludes():
    analyses = [
        ChunkAnalysis(chunk_id=0, text="a", confidence=0.9),
        ChunkAnalysis(chunk_id=1, text="b", confidence=0.2),
        ChunkAnalysis(chunk_id=2, text="c", confidence=0.5),
        ChunkAnalysis(chunk_id=3, text="d", confidence=0.1),
    ]
    picked = engine.select_low_confidence(analyses, k=2)
    assert [p.chunk_id for p in picked] == [3, 1]        # two lowest, ascending

    excl = engine.select_low_confidence(analyses, k=2, exclude_ids={3})
    assert [p.chunk_id for p in excl] == [1, 2]          # 3 skipped

    thr = engine.select_low_confidence(analyses, k=10, threshold=0.3)
    assert {p.chunk_id for p in thr} == {1, 3}           # only below threshold


# --------------------------------------------------------------------------- #
# fusion (confidence x competence quadrant map)
# --------------------------------------------------------------------------- #

def _fusion_inputs():
    """Four segments, one per quadrant, plus scores/deltas that attribute a delta to each."""
    analyses = [
        ChunkAnalysis(chunk_id=0, text="aware_gap",           confidence=0.2),  # dist 0.8
        ChunkAnalysis(chunk_id=1, text="productive_struggle", confidence=0.2),  # dist 0.8
        ChunkAnalysis(chunk_id=2, text="blind_spot",          confidence=0.95), # dist 0.05
        ChunkAnalysis(chunk_id=3, text="mastery",             confidence=0.95), # dist 0.05
    ]
    per_question = [
        QuestionDelta(question_id=0, taught_mean=0.0, cold_mean=0.5, delta=-0.5),  # failed
        QuestionDelta(question_id=1, taught_mean=0.9, cold_mean=0.1, delta=0.8),   # passed
        QuestionDelta(question_id=2, taught_mean=0.1, cold_mean=0.4, delta=-0.3),  # failed
        QuestionDelta(question_id=3, taught_mean=0.9, cold_mean=0.2, delta=0.7),   # passed
    ]
    scores = [
        Score(question_id=q, arm=Arm.taught, persona_seed="p", correct=True, cited_segment_ids=[q])
        for q in range(4)
    ]
    return analyses, scores, per_question


def test_fuse_assigns_all_four_quadrants():
    analyses, scores, per_question = _fusion_inputs()
    result = fuse("s", analyses, scores, per_question)
    quad = {s.segment_id: s.quadrant for s in result.per_segment}
    assert quad == {0: "aware_gap", 1: "productive_struggle", 2: "blind_spot", 3: "mastery"}
    assert result.quadrant_counts == {
        "aware_gap": 1, "productive_struggle": 1, "blind_spot": 1, "mastery": 1,
    }
    # disturbance is 1 - confidence
    dist = {s.segment_id: s.disturbance for s in result.per_segment}
    assert dist[0] == 0.8 and dist[2] == 0.05


def test_fuse_disturbance_threshold_boundary():
    # exactly at DISTURBANCE_HIGH is NOT "high" (strict >)
    a = ChunkAnalysis(chunk_id=0, text="edge", confidence=round(1 - DISTURBANCE_HIGH, 3))
    s = Score(question_id=0, arm=Arm.taught, persona_seed="p", correct=True, cited_segment_ids=[0])
    d = QuestionDelta(question_id=0, taught_mean=0.0, cold_mean=0.5, delta=-0.5)
    res = fuse("s", [a], [s], [d])
    assert res.per_segment[0].quadrant == "blind_spot"  # low disturbance + failed


def test_fuse_segment_without_delta_is_unknown():
    a = ChunkAnalysis(chunk_id=7, text="no scores", confidence=0.3)
    res = fuse("s", [a], [], [])
    assert res.per_segment[0].quadrant == "unknown"
    assert res.per_segment[0].transfer_delta is None
    assert res.calibration_rho is None  # <2 crossed points


def test_fuse_calibration_rho_positive_when_calibrated():
    # A CALIBRATED speaker: sounded unsure (high disturbance) exactly where transfer failed
    # (delta <= 0), and confident where it landed. disturbance and -delta then move together.
    analyses = [
        ChunkAnalysis(chunk_id=0, text="unsure+failed", confidence=0.1),  # dist 0.9
        ChunkAnalysis(chunk_id=1, text="unsure+failed", confidence=0.3),  # dist 0.7
        ChunkAnalysis(chunk_id=2, text="sure+passed",   confidence=0.8),  # dist 0.2
        ChunkAnalysis(chunk_id=3, text="sure+passed",   confidence=0.9),  # dist 0.1
    ]
    per_question = [
        QuestionDelta(question_id=0, taught_mean=0.0, cold_mean=0.6, delta=-0.6),
        QuestionDelta(question_id=1, taught_mean=0.1, cold_mean=0.5, delta=-0.4),
        QuestionDelta(question_id=2, taught_mean=0.9, cold_mean=0.4, delta=0.5),
        QuestionDelta(question_id=3, taught_mean=0.9, cold_mean=0.2, delta=0.7),
    ]
    scores = [
        Score(question_id=q, arm=Arm.taught, persona_seed="p", correct=True, cited_segment_ids=[q])
        for q in range(4)
    ]
    res = fuse("s", analyses, scores, per_question)
    assert res.calibration_rho is not None
    assert res.calibration_rho > 0.9   # strongly calibrated


# --------------------------------------------------------------------------- #
# HTTP surface (in-memory store)
# --------------------------------------------------------------------------- #

def test_root_and_health():
    assert client.get("/").json()["service"] == "wut"
    assert client.get("/health").json() == {"ok": True}


def test_confusion_health_reports_unreachable():
    body = client.get("/confusion/health").json()
    assert body["reachable"] is False   # ml-service pointed at a dead port


def test_ingest_then_fusion_roundtrip():
    sid = "http-ingest"
    chunks = [
        {"chunk_id": 0, "text": "clear", "confidence": 0.95, "anomalies": []},
        {"chunk_id": 1, "text": "shaky", "confidence": 0.2, "anomalies": []},
    ]
    r = client.post("/confusion/ingest", json={"session_id": sid, "chunks": chunks})
    assert r.status_code == 200
    assert r.json() == {"session_id": sid, "n_chunks": 2}

    f = client.get(f"/fusion/{sid}")
    assert f.status_code == 200
    body = f.json()
    assert body["session_id"] == sid
    dist = {s["segment_id"]: s["disturbance"] for s in body["per_segment"]}
    assert dist == {0: 0.05, 1: 0.8}
    # no measurement run yet -> every segment is 'unknown'
    assert all(s["quadrant"] == "unknown" for s in body["per_segment"])


def test_fusion_unknown_session_404():
    assert client.get("/fusion/does-not-exist").status_code == 404


def test_mock_requires_transcript_then_analyzes():
    sid = "http-mock"
    # no transcript yet
    assert client.post("/confusion/mock", params={"session_id": sid}).status_code == 404
    # seed a transcript directly through the store, then mock over it
    run(store.append_segment(sid, Segment(id=0, idx=0, text="It is maybe kind of unclear, um.")))
    run(store.append_segment(sid, Segment(id=1, idx=1, text="Energy is conserved.")))
    r = client.post("/confusion/mock", params={"session_id": sid})
    assert r.status_code == 200
    assert r.json()["n_chunks"] == 2
    # the hedged segment should now read low-confidence in fusion
    body = client.get(f"/fusion/{sid}").json()
    dist = {s["segment_id"]: s["disturbance"] for s in body["per_segment"]}
    assert dist[0] > dist[1]


def test_answer_unknown_question_404():
    r = client.post(
        "/questions/answer",
        json={"session_id": "nope", "question_id": 999, "answer": "x"},
    )
    assert r.status_code == 404


def test_question_ledger_answer_and_history():
    sid = "http-ledger"
    # seed a question through the store (generation itself needs the LLM)
    q = TargetedQuestion(id=0, chunk_id=1, text="Why does that hold?", anomaly_type="hedging")
    run(store.record_questions(sid, [q]))

    hist = client.get(f"/questions/history/{sid}").json()
    assert len(hist) == 1 and hist[0]["answer"] is None

    r = client.post(
        "/questions/answer",
        json={"session_id": sid, "question_id": 0, "answer": "Because energy is conserved."},
    )
    assert r.json()["recorded"] is True

    hist = client.get(f"/questions/history/{sid}").json()
    assert hist[0]["answer"] == "Because energy is conserved."
    assert hist[0]["answered_at"] is not None


def test_from_chunk_degrades_and_stays_silent():
    """ml-service unreachable -> neutral analysis (confidence 1.0, empty text) -> not confused."""
    sid = "http-chunk"
    r = client.post(
        "/questions/from_chunk",
        data={"session_id": sid, "chunk_id": 0},
        files={"audio": ("chunk.wav", b"RIFFfakewavdata", "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["asked"] is False
    assert body["question"] is None
    assert body["analysis"]["confidence"] == 1.0
