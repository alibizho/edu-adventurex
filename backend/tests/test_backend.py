"""Offline backend tests: pure logic (confusion engine, fusion) + the HTTP surface via
FastAPI's TestClient, backed by the in-memory store. Fully hermetic — no LLM, no ml-service,
no Postgres. Run:  .venv/bin/python -m pytest tests/test_backend.py -q

Config is forced hermetic (memory store, unreachable ml-service) BEFORE the app is imported,
so the confusion client degrades to a neutral analysis instead of hitting a real GPU box.
"""
import asyncio
import os
from unittest.mock import patch

os.environ["STORE_BACKEND"] = "memory"
os.environ["ML_SERVICE_URL"] = "http://127.0.0.1:9"  # unreachable -> neutral degrade

from fastapi.testclient import TestClient  # noqa: E402
from httpx import Request, Response  # noqa: E402
from openai import AuthenticationError  # noqa: E402

from app.confusion import engine  # noqa: E402
from app.curriculum.build import CurriculumGenerationError  # noqa: E402
from app.fusion import DISTURBANCE_HIGH, fuse  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import (  # noqa: E402
    Arm,
    ChunkAnalysis,
    PathMemory,
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


def test_invalid_curriculum_shape_returns_actionable_502():
    async def invalid_curriculum(_request):
        raise CurriculumGenerationError("invalid model output")

    with patch("app.api.plan_routes.build_plan", invalid_curriculum):
        response = client.post(
            "/plan/build",
            json={
                "original_input": "quantum mechanics",
                "confirmed_topic": "quantum mechanics",
                "num_classes": 5,
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "THE LANGUAGE MODEL COULD NOT PRODUCE A VALID COURSE STRUCTURE. PLEASE TRY AGAIN."
    }


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
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["store_backend"] == "memory"
    assert isinstance(health["llm_configured"], bool)


def test_llm_authentication_error_is_not_reported_as_dead_backend():
    async def reject_scope(_request):
        request = Request("POST", "https://api.deepseek.com/chat/completions")
        response = Response(401, request=request)
        raise AuthenticationError(
            "invalid key",
            response=response,
            body={"error": {"message": "invalid key"}},
        )

    with patch("app.api.plan_routes.scope_topic", reject_scope):
        response = client.post(
            "/plan/scope",
            json={"original_input": "quantum physics", "material_text": None},
        )

    assert response.status_code == 401
    assert response.json() == {
        "detail": (
            "THE LANGUAGE MODEL API KEY IS INVALID. UPDATE backend/.env AND RESTART THE BACKEND."
        )
    }


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


def test_memory_store_getters_return_copies_not_the_live_list():
    """The memory store must not hand back its internal list.

    The db store builds a fresh list per call, so a caller that holds a transcript across an
    append_segment sees the new segment under one backend and not the other. That divergence made
    the same teaching turn behave differently depending on STORE_BACKEND: class_audio_turn put the
    learner's utterance into the student prompt twice, and the goal probe's chunk_id pointed at a
    different segment. Both stores now return a copy; this pins that down.
    """
    sid = "store-copy"
    run(store.append_segment(sid, Segment(id=0, idx=0, text="first")))

    held = run(store.get_transcript(sid))
    assert [s.id for s in held] == [0]

    run(store.append_segment(sid, Segment(id=1, idx=1, text="second")))
    assert [s.id for s in held] == [0], "an already-fetched transcript must not grow behind the caller"
    assert [s.id for s in run(store.get_transcript(sid))] == [0, 1]

    # Same contract for the other two list getters.
    held_analyses = run(store.get_analyses(sid))
    run(store.append_analysis(sid, ChunkAnalysis(chunk_id=0, text="first", confidence=0.9)))
    assert held_analyses == []

    held_history = run(store.get_history(sid))
    run(store.record_questions(sid, [TargetedQuestion(id=0, chunk_id=0, text="Why?")]))
    assert held_history == []


def test_memory_store_history_entries_stay_shared_for_answers():
    """The copy above is shallow on purpose: record_answer mutates the QAEntry objects, so an
    already-fetched history must still observe an answer landing."""
    sid = "store-copy-answer"
    run(store.record_questions(sid, [TargetedQuestion(id=0, chunk_id=0, text="Why does that hold?")]))

    held = run(store.get_history(sid))
    assert held[0].answer is None

    assert run(store.record_answer(sid, 0, "Because energy is conserved.")) is True
    assert held[0].answer == "Because energy is conserved."


# ---- struggle ledger ("Feynman ledger") ----

def _ledger_analysis(text, confidence, target=None, anomalies=()):
    from app.schemas import Anomaly as A
    return ChunkAnalysis(
        chunk_id=0, text=text, confidence=confidence, localized_target=target,
        anomalies=[A(type=t, source="test", score=s) for t, s in anomalies],
    )


def test_struggle_ledger_accumulates_and_picks_the_worst_concept():
    """Repeated stumbles must outrank a single one — that's what makes the focus stable enough to
    hold a conversation together instead of chasing whatever broke most recently."""
    from app.curriculum.teaching import _update_struggle_ledger
    from app.schemas import ClassProgressRecord

    progress = ClassProgressRecord()
    _update_struggle_ledger(progress, _ledger_analysis(
        "the mitochondria makes energy", 0.4, "mitochondria", [("recall_failure", 0.6)]))
    assert progress.focus_target == "mitochondria"

    # A different concept scores higher on one turn...
    _update_struggle_ledger(progress, _ledger_analysis(
        "the calvin cycle fixes carbon", 0.3, "calvin cycle", [("factual_error", 0.8)]))
    assert progress.focus_target == "calvin cycle"

    # ...but two more mitochondria stumbles accumulate past it.
    for _ in range(2):
        _update_struggle_ledger(progress, _ledger_analysis(
            "mitochondria again", 0.4, "mitochondria", [("fluency_issue", 0.5)]))
    assert progress.struggle_scores["mitochondria"] == 1.6
    assert progress.focus_target == "mitochondria"


def test_struggle_ledger_decays_on_a_clear_confident_explanation():
    """A clean, confident sentence naming the concept is evidence of mastery: halve it, and drop it
    once it falls below the threshold."""
    from app.curriculum.teaching import _update_struggle_ledger
    from app.schemas import ClassProgressRecord

    progress = ClassProgressRecord()
    _update_struggle_ledger(progress, _ledger_analysis(
        "uhh the ribosome", 0.3, "ribosome", [("recall_failure", 0.6)]))
    assert progress.struggle_scores["ribosome"] == 0.6

    clean = _ledger_analysis("the ribosome assembles proteins from mRNA", 0.95)
    _update_struggle_ledger(progress, clean)
    assert progress.struggle_scores["ribosome"] == 0.3      # halved, still outstanding
    assert progress.focus_target == "ribosome"

    _update_struggle_ledger(progress, clean)                # 0.15 -> below MASTERED_BELOW
    assert "ribosome" not in progress.struggle_scores
    assert progress.focus_target == ""


def test_struggle_ledger_decay_matches_multi_word_targets():
    """localized_target is often a multi-word judge span. A set(text.split()) membership test can
    never match one, so the decay branch would look right and silently never fire."""
    from app.curriculum.teaching import _update_struggle_ledger
    from app.schemas import ClassProgressRecord

    progress = ClassProgressRecord()
    _update_struggle_ledger(progress, _ledger_analysis(
        "something about it", 0.3, "the Calvin cycle", [("factual_error", 0.8)]))
    assert progress.struggle_scores["the Calvin cycle"] == 0.8

    _update_struggle_ledger(progress, _ledger_analysis(
        "The Calvin cycle turns CO2 into sugar using ATP.", 0.93))
    assert progress.struggle_scores["the Calvin cycle"] == 0.4


def test_struggle_ledger_ignores_cognitive_load_and_stays_bounded():
    """cognitive_load is effort, not misunderstanding, and fires on most long sentences."""
    from app.curriculum.teaching import LEDGER_LIMIT, _update_struggle_ledger
    from app.schemas import ClassProgressRecord

    progress = ClassProgressRecord()
    _update_struggle_ledger(progress, _ledger_analysis(
        "a long careful sentence", 0.5, "osmosis", [("cognitive_load", 0.9)]))
    assert progress.struggle_scores == {} and progress.focus_target == ""

    for i in range(LEDGER_LIMIT + 6):
        _update_struggle_ledger(progress, _ledger_analysis(
            f"concept {i}", 0.3, f"concept-{i}", [("recall_failure", 0.1 * (i + 1))]))
    assert len(progress.struggle_scores) == LEDGER_LIMIT
    # The cap must keep the WORST concepts, not the most recent ones.
    assert progress.focus_target == f"concept-{LEDGER_LIMIT + 5}"


# ---- multi-turn conversation with one student ----

def test_thread_turns_counts_the_whole_follow_up_chain():
    """The cap has to apply to the back-and-forth about one concept, not to each rephrasing of it,
    so the count follows parent_id links rather than looking at a single question."""
    from app.store.base import count_thread_answers
    from app.schemas import QAEntry as QE

    q0 = TargetedQuestion(id=0, chunk_id=0, text="what is a router?")
    q1 = TargetedQuestion(id=1, chunk_id=1, text="say it another way?", parent_id=0)
    q2 = TargetedQuestion(id=2, chunk_id=2, text="and the table?", parent_id=1)
    unrelated = TargetedQuestion(id=3, chunk_id=3, text="different topic")
    ledger = [
        QE(question=q0, answer="a box"), QE(question=q1, answer="a thing"),
        QE(question=q2), QE(question=unrelated, answer="x"),
    ]
    # Two answered in the chain; the unrelated question doesn't count toward it.
    assert count_thread_answers(ledger, 2) == 2
    assert count_thread_answers(ledger, 0) == 2      # same thread from either end
    assert count_thread_answers(ledger, 3) == 1


def test_conversation_ends_on_a_correct_answer_and_decays_the_ledger():
    from unittest.mock import patch as _patch
    from app.curriculum import teaching as t
    from app.schemas import ClassProgressRecord, ClassUnit, QAEntry as QE

    cls = ClassUnit(class_id="c1", title="Networks", objective="routing")
    progress = ClassProgressRecord(struggle_scores={"routing table": 0.8}, focus_target="routing table")
    entry = QE(question=TargetedQuestion(
        id=0, chunk_id=0, text="what is the routing table for?",
        answer_key="It maps destinations to the next hop."))
    analysis = ChunkAnalysis(chunk_id=1, text="it maps destinations to the next hop", confidence=0.9)

    with _patch.object(t, "grade_answer", return_value=(True, False)):
        reply, follow_up, correct, over = run(t._conversation_turn(
            "s", cls, entry, analysis.text, analysis, PathMemory(path_id="p"), progress, []))

    assert correct is True and over is True and follow_up is None
    assert "MAKES SENSE" in reply
    assert progress.struggle_scores["routing table"] == 0.4   # halved by the correct answer


def test_wrong_answer_earns_a_follow_up_on_the_same_thread():
    from unittest.mock import patch as _patch
    from app.curriculum import teaching as t
    from app.schemas import ClassProgressRecord, ClassUnit, QAEntry as QE

    cls = ClassUnit(class_id="c1", title="Networks", objective="routing")
    entry = QE(question=TargetedQuestion(
        id=0, chunk_id=0, text="what is the routing table for?", answer_key="Maps destination to next hop."))
    analysis = ChunkAnalysis(chunk_id=1, text="uhh it stores websites", confidence=0.4)

    async def fake_follow_up(*_a, **kw):
        return [TargetedQuestion(id=1, chunk_id=1, text="what does a row in it hold?",
                                 parent_id=kw.get("parent_id"))]

    with _patch.object(t, "grade_answer", return_value=(False, False)), \
         _patch.object(t, "generate_targeted_questions", fake_follow_up):
        reply, follow_up, correct, over = run(t._conversation_turn(
            "s2", cls, entry, analysis.text, analysis, PathMemory(path_id="p"),
            ClassProgressRecord(), []))

    assert correct is False and over is False
    assert follow_up is not None and follow_up.parent_id == 0   # linked to the parent question
    assert reply == follow_up.text


def test_question_without_an_answer_key_releases_instead_of_trapping():
    """GPU-authored questions carry no key. Holding the learner against a standard we cannot check
    would trap them, so one attempt and the class moves on."""
    from unittest.mock import patch as _patch
    from app.curriculum import teaching as t
    from app.schemas import ClassProgressRecord, ClassUnit, QAEntry as QE

    cls = ClassUnit(class_id="c1", title="Networks", objective="routing")
    entry = QE(question=TargetedQuestion(id=0, chunk_id=0, text="wait, what is a hop?"))
    analysis = ChunkAnalysis(chunk_id=1, text="not sure", confidence=0.4)

    with _patch.object(t, "grade_answer", return_value=(False, False)):
        _reply, follow_up, correct, over = run(t._conversation_turn(
            "s3", cls, entry, analysis.text, analysis, PathMemory(path_id="p"),
            ClassProgressRecord(), []))

    assert over is True and follow_up is None
    assert correct is None      # ungraded, not "wrong"


# ---- "I don't know" (the branch between asking again and teaching) ----

def test_give_up_is_detected_without_punishing_a_short_answer():
    """This decides whether the learner gets another question or gets the answer. Too permissive
    hands over something they could have reached; too strict strands them, which is the bug it
    exists to fix — so it matches phrases, never length."""
    from app.curriculum.teaching import _is_give_up

    for surrender in ["I don't know", "i dont know", "IDK", "dunno", "no idea", "I'm not sure",
                      "uhh no clue", "i give up", "you tell me", "just tell me the answer",
                      "i forgot", "i can't explain it", "skip", "pass",
                      # Nothing but filler is a shrug however many words it took.
                      "um, yeah, like...", "   ", ""]:
        assert _is_give_up(surrender), f"{surrender!r} is the learner asking for help"

    for attempt in ["Friction.", "Force equals mass times acceleration.",
                    "Because the two forces cancel out.", "Gravity pulls it down",
                    # Hedged, but an attempt: they should be pressed, not handed the answer.
                    "um, i think it's maybe about acceleration?"]:
        assert not _is_give_up(attempt), f"{attempt!r} is an attempt and must be graded"


# ---- filler-word rejection (why students used to ask about "bro") ----

def test_filler_words_never_become_the_concept_being_chased():
    """Regression on real data. Production analyses localized onto "Yeah", "You", "the", "bro" —
    the top targets by frequency — and the student then asked "Wait, I thought bro was different?".
    Multi-word judge spans must still survive: those are deliberate."""
    from app.curriculum.teaching import is_concept_like

    # Every one of these was a real localized_target in the database.
    for junk in ["Yeah", "I", "You", "What", "Thank", "No", "Oh", "is", "the", "bro",
                 "usually", "it's", "like", "one", "A", "Good", "",
                 # Drawn-out variants the bare stopword set used to miss — a pause on "umm" is the
                 # exact failure that produced "I thought umm was different?".
                 "umm", "uhh", "hmmm", "sooo", "aaah",
                 # Hedges not in the stopword list.
                 "basically", "essentially", "obviously", "totally",
                 # Multi-word spans the old single-token gate let through because " " in clean.
                 "um basically", "yeah so um", "like essentially"]:
        assert not is_concept_like(junk), f"{junk!r} should not be chased as a concept"

    for concept in ["normalization", "convolution", "overfitting", "mitochondria",
                    "the water cycle", "routing table", "gradient descent",
                    # Doubled-letter words must NOT be collapsed into false fillers.
                    "bee", "book", "address"]:
        assert is_concept_like(concept), f"{concept!r} is a real concept and must survive"


def test_ledger_ignores_filler_targets():
    """The ledger is what feeds focus_target back to the GPU, so junk in here compounds: every
    later question would chase the filler word."""
    from app.curriculum.teaching import _update_struggle_ledger
    from app.schemas import ClassProgressRecord

    progress = ClassProgressRecord()
    _update_struggle_ledger(progress, _ledger_analysis(
        "Yeah so um", 0.05, "Yeah", [("recall_failure", 0.9)]))
    assert progress.struggle_scores == {} and progress.focus_target == ""

    _update_struggle_ledger(progress, _ledger_analysis(
        "normalization scales the inputs", 0.3, "normalization", [("recall_failure", 0.7)]))
    assert progress.focus_target == "normalization"


def test_gpu_question_about_a_filler_word_is_dropped():
    """The GPU writes its question from target_concept. When that's a filler word the question is
    about nothing, so it must be refused and the LLM generator asked instead."""
    from app.curriculum.teaching import _question_from_analysis
    from app.schemas import StudentQuestion

    junk = ChunkAnalysis(chunk_id=0, text="yeah so", confidence=0.2, student_question=StudentQuestion(
        question_text='How does the word "You" relate to the grid-like structure of CNNs?',
        target_concept="You", anomaly_type="off_topic"))
    assert run(_question_from_analysis("s", junk, PathMemory(path_id="p"))) is None

    real = ChunkAnalysis(chunk_id=0, text="convolution slides a kernel", confidence=0.2,
                         student_question=StudentQuestion(
                             question_text="What does the kernel actually do as it slides?",
                             target_concept="convolution", anomaly_type="recall_failure"))
    assert run(_question_from_analysis("s", real, PathMemory(path_id="p"))) is not None
