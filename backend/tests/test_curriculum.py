import asyncio
import os

import pytest

os.environ["STORE_BACKEND"] = "memory"
os.environ["ML_SERVICE_URL"] = "http://127.0.0.1:9"

from app.curriculum import teaching
from app.schemas import (
    ChunkAnalysis,
    ClassObjective,
    ClassUnit,
    GrowthPath,
    PathMemory,
    Segment,
    TargetedQuestion,
    TeachTurnResponse,
)
from app.store import store

def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)

def _path(path_id: str, *classes: ClassUnit, topic: str = "Mechanics") -> GrowthPath:
    return GrowthPath(
        path_id=path_id,
        original_input="I want to learn physics",
        confirmed_topic=topic,
        total_classes=len(classes),
        recommended_order=[c.class_id for c in classes],
        classes=list(classes),
    )

def test_store_path_roundtrip_and_save_class():
    path = _path(
        "gp-rt",
        ClassUnit(class_id="c1", title="Forces", objective="Understand forces."),
        ClassUnit(class_id="c2", title="Energy", objective="Understand energy."),
    )
    run(store.save_path(path))

    got = run(store.get_path("gp-rt"))
    assert got is not None
    assert got.confirmed_topic == "Mechanics"
    assert [c.class_id for c in got.classes] == ["c1", "c2"]
    assert got.classes[0].notes_generated is False

    c1 = got.classes[0]
    c1.teacher_notes = "# Forces\nA force is a push or pull."
    c1.notes_generated = True
    run(store.save_class("gp-rt", c1))

    again = run(store.get_path("gp-rt"))
    assert again.classes[0].teacher_notes.startswith("# Forces")
    assert again.classes[0].notes_generated is True
    assert again.classes[1].notes_generated is False

def test_store_memory_roundtrip():
    fresh = run(store.get_memory("gp-mem"))
    assert fresh.path_id == "gp-mem" and fresh.covered_concepts == []

    fresh.covered_concepts.append("Forces")
    fresh.asked_questions.append("Why does F=ma hold?")
    run(store.update_memory("gp-mem", fresh))

    again = run(store.get_memory("gp-mem"))
    assert again.covered_concepts == ["Forces"]
    assert again.asked_questions == ["Why does F=ma hold?"]

def _stub_llm(monkeypatch):
    seen_histories: list[list[str]] = []

    async def fake_student_turn(transcript, utterance):
        nid = max((s.id for s in transcript), default=-1) + 1
        return TeachTurnResponse(
            student_reply="wait, what?",
            new_segment=Segment(id=nid, idx=len(transcript), text=utterance),
        )

    async def fake_generate(chunks, history, start_id=0, topic=None, transcript="",
                            focus_target="", parent_id=None, objectives=None):
        seen_histories.append([e.question.text for e in history])
        return [
            TargetedQuestion(
                id=start_id, chunk_id=chunks[0].chunk_id, text=f"probe:{chunks[0].text[:16]}",
                answer_key=f"key:{chunks[0].text[:16]}", parent_id=parent_id,
            )
        ]

    monkeypatch.setattr(teaching, "student_turn", fake_student_turn)
    monkeypatch.setattr(teaching, "generate_targeted_questions", fake_generate)
    return seen_histories

def test_confident_utterance_asks_nothing(monkeypatch):
    _stub_llm(monkeypatch)
    c1 = ClassUnit(class_id="c1", title="Forces", objective="Understand forces.")
    run(store.save_path(_path("gp-clear", c1)))

    r = run(teaching.class_teach_turn(
        "gp-clear", "c1", c1, "A force is a push or a pull that changes an object's motion."
    ))
    assert r.asked is False and r.question is None
    assert run(store.get_memory("gp-clear")).asked_questions == []

def test_hedged_utterance_asks_and_memory_spans_classes(monkeypatch):
    seen = _stub_llm(monkeypatch)
    c1 = ClassUnit(class_id="c1", title="Forces", objective="Understand forces.")
    c2 = ClassUnit(class_id="c2", title="Energy", objective="Understand energy.")
    run(store.save_path(_path("gp-hedge", c1, c2)))

    r1 = run(teaching.class_teach_turn(
        "gp-hedge", "c1", c1, "um, i think a force is maybe like, kind of a push?"
    ))
    assert r1.asked is True and r1.question is not None
    q1 = r1.question.text
    assert q1 in run(store.get_memory("gp-hedge")).asked_questions

    r2 = run(teaching.class_teach_turn(
        "gp-hedge", "c2", c2, "uh, energy is sort of, i dunno, the ability to do stuff?"
    ))
    assert r2.asked is True
    assert q1 in seen[-1], "cross-class asked question should be injected into question history"

def _stub_conversation(monkeypatch, *, correct: bool, follow_up_key: str | None = None):
    async def fake_grade(answer, question):
        return correct, False

    async def fake_explain(question, topic="", transcript="", objectives=None):
        return "HERE IS THE ANSWER"

    async def fake_follow_up(chunks, history, start_id=0, topic=None, transcript="",
                             focus_target="", parent_id=None, objectives=None):
        return [TargetedQuestion(id=start_id, chunk_id=0, text="but why does that happen?",
                                 answer_key=follow_up_key, parent_id=parent_id)]

    monkeypatch.setattr(teaching, "grade_answer", fake_grade)
    monkeypatch.setattr(teaching, "explain_answer", fake_explain)
    monkeypatch.setattr(teaching, "generate_targeted_questions", fake_follow_up)

def _answer_once(path_id: str, cls: ClassUnit, question_id: int, answer: str):
    return run(teaching.class_audio_turn(
        path_id, cls.class_id, cls,
        ChunkAnalysis(chunk_id=0, text=answer, confidence=0.8),
        degraded=False, silent=False, answering_question_id=question_id,
    ))

def test_a_keyless_question_gets_a_follow_up_before_it_is_answered_for_you(monkeypatch):
    _stub_conversation(monkeypatch, correct=False)
    c1 = ClassUnit(class_id="c1", title="Waves", objective="Understand waves.")
    run(store.save_path(_path("gp-keyless", c1, topic="Physics")))
    sid = teaching.class_session_id("gp-keyless", "c1")
    run(store.record_questions(sid, [TargetedQuestion(id=0, chunk_id=0, text="what is a wave?")]))

    first = _answer_once("gp-keyless", c1, 0, "uh, it is a thing that moves along")
    assert first.conversation_over is False, "the first answer must not end the conversation"
    assert first.question is not None and first.asked is True
    assert first.student_reply == "but why does that happen?"

    second = _answer_once("gp-keyless", c1, first.question.id, "still not sure honestly")
    assert second.conversation_over is True
    assert second.student_reply == "HERE IS THE ANSWER"
    assert second.question is None

def test_saying_you_dont_know_to_a_student_gets_the_answer_immediately(monkeypatch):
    _stub_conversation(monkeypatch, correct=False, follow_up_key="a travelling disturbance")
    c1 = ClassUnit(class_id="c1", title="Waves", objective="Understand waves.")
    run(store.save_path(_path("gp-idk", c1, topic="Physics")))
    sid = teaching.class_session_id("gp-idk", "c1")
    run(store.record_questions(sid, [
        TargetedQuestion(id=0, chunk_id=0, text="what does a wave carry?",
                         answer_key="energy, not matter"),
    ]))

    result = _answer_once("gp-idk", c1, 0, "honestly I don't know")

    assert result.student_reply == "HERE IS THE ANSWER", "the student must answer, not re-ask"
    assert result.question is None and result.asked is False
    assert result.conversation_over is True
    progress = run(store.get_memory("gp-idk")).class_progress["c1"]
    assert progress.explanations_given == 1

def test_a_correct_answer_still_ends_the_conversation(monkeypatch):
    _stub_conversation(monkeypatch, correct=True)
    c1 = ClassUnit(class_id="c1", title="Waves", objective="Understand waves.")
    run(store.save_path(_path("gp-correct", c1, topic="Physics")))
    sid = teaching.class_session_id("gp-correct", "c1")
    run(store.record_questions(sid, [
        TargetedQuestion(id=0, chunk_id=0, text="what is a wave?", answer_key="a travelling disturbance"),
    ]))

    result = _answer_once("gp-correct", c1, 0, "a travelling disturbance that carries energy")
    assert result.conversation_over is True and result.answer_correct is True
    assert result.question is None

def test_a_graded_question_presses_twice_before_answering_itself(monkeypatch):
    _stub_conversation(monkeypatch, correct=False, follow_up_key="because it carries energy")
    c1 = ClassUnit(class_id="c1", title="Waves", objective="Understand waves.")
    run(store.save_path(_path("gp-graded", c1, topic="Physics")))
    sid = teaching.class_session_id("gp-graded", "c1")
    run(store.record_questions(sid, [
        TargetedQuestion(id=0, chunk_id=0, text="what is a wave?", answer_key="a travelling disturbance"),
    ]))

    question_id = 0
    for attempt in range(teaching.MAX_CONVERSATION_TURNS - 1):
        result = _answer_once("gp-graded", c1, question_id, f"attempt {attempt}")
        assert result.conversation_over is False, f"attempt {attempt} should still press"
        question_id = result.question.id

    last = _answer_once("gp-graded", c1, question_id, "i give up on phrasing this")
    assert last.conversation_over is True and last.student_reply == "HERE IS THE ANSWER"

def test_admitting_you_dont_know_is_answered_not_re_asked(monkeypatch):
    _stub_conversation(monkeypatch, correct=False)
    c1 = ClassUnit(class_id="c1", title="Waves", objective="Understand waves.",
                   objectives=[ClassObjective(id="o1", text="Explain what a wave carries.")])
    run(store.save_path(_path("gp-stuck", c1, topic="Physics")))

    result = run(teaching.class_audio_turn(
        "gp-stuck", "c1", c1,
        ChunkAnalysis(chunk_id=0, text="honestly I don't know how to explain this one",
                      confidence=0.9),
        degraded=False, silent=True,
    ))

    assert result.student_reply == "HERE IS THE ANSWER"
    assert result.explained is True, "the UI cannot tell an answer from chatter without this"
    assert result.asked is False and result.question is None, "a stuck teacher must not be re-probed"

def test_typing_that_you_dont_know_is_answered_too(monkeypatch):
    _stub_conversation(monkeypatch, correct=False)

    async def fake_student_turn(transcript, utterance):
        nid = max((s.id for s in transcript), default=-1) + 1
        return TeachTurnResponse(
            student_reply="haha okay", new_segment=Segment(id=nid, idx=len(transcript), text=utterance),
        )

    monkeypatch.setattr(teaching, "student_turn", fake_student_turn)
    c1 = ClassUnit(class_id="c1", title="Waves", objective="Understand waves.",
                   objectives=[ClassObjective(id="o1", text="Explain what a wave carries.")])
    run(store.save_path(_path("gp-typed", c1, topic="Physics")))

    result = run(teaching.class_teach_turn("gp-typed", "c1", c1, "i have no idea honestly"))

    assert result.explained is True
    assert result.student_reply == "HERE IS THE ANSWER", "not the student's conversational reply"
    assert result.asked is False

@pytest.mark.parametrize("utterance", [
    "i don't know",
    "I don’t know",
    "hmm, I'm not sure",
    "I don't understand this at all",
    "yeah I'm totally lost here",
    "no idea, sorry",
    "um, uh, like",
])
def test_give_up_phrasings(utterance):
    assert teaching._is_give_up(utterance), f"{utterance!r} should read as not knowing"

@pytest.mark.parametrize("utterance", [
    "a wave carries energy without carrying matter along with it",
    "friction",
    "the alliance system pulled every great power into the war",
    "uh, energy is sort of, i dunno, the ability to do stuff",
    "waves carry energy, though i'm not sure i can explain the maths",
    "the treaty was signed in 1919 and no idea why they picked that date",
])
def test_a_real_explanation_is_not_mistaken_for_giving_up(utterance):
    assert not teaching._is_give_up(utterance)

def test_end_class_folds_memory():
    c1 = ClassUnit(class_id="c1", title="Waves", objective="Understand waves.")
    run(store.save_path(_path("gp-end", c1, topic="Physics")))
    sid = teaching.class_session_id("gp-end", "c1")
    run(store.record_questions(sid, [
        TargetedQuestion(id=0, chunk_id=0, text="Q-answered"),
        TargetedQuestion(id=1, chunk_id=1, text="Q-unanswered"),
    ]))
    run(store.record_answer(sid, 0, "an answer"))

    mem = run(teaching.end_class("gp-end", "c1", c1))
    assert "Waves" in mem.covered_concepts
    assert "Q-answered" in mem.understood
    assert "Q-unanswered" in mem.struggled

def test_reset_class_erases_the_session_and_takes_back_its_memory():
    c1 = ClassUnit(
        class_id="c1", title="Waves", objective="Understand waves.",
        teacher_notes="- Waves carry energy, not matter.",
    )
    run(store.save_path(_path("gp-reset", c1, topic="Physics")))
    sid = teaching.class_session_id("gp-reset", "c1")
    run(store.append_segment(sid, Segment(id=0, idx=0, text="a wave is, uh, a thing that waves")))
    run(store.record_questions(sid, [TargetedQuestion(id=0, chunk_id=0, text="Q-asked")]))
    run(store.record_answer(sid, 0, "an answer"))
    run(teaching.end_class("gp-reset", "c1", c1))

    memory = run(teaching.reset_class("gp-reset", "c1", c1))

    assert run(store.get_transcript(sid)) == []
    assert run(store.get_history(sid)) == []
    assert "Waves" not in memory.covered_concepts
    assert "Q-asked" not in memory.understood
    assert memory.asked_questions == []
    progress = memory.class_progress["c1"]
    assert progress.status == "not_started" and progress.turn_count == 0
    assert progress.reset_count == 1, "in-flight writers compare this before resurrecting a class"

def test_teaching_turn_in_flight_during_a_reset_is_dropped():
    c1 = ClassUnit(class_id="c1", title="Optics", objective="Understand optics.")
    run(store.save_path(_path("gp-inflight", c1, topic="Physics")))
    stale = PathMemory.model_validate(run(store.get_memory("gp-inflight")).model_dump())
    teaching._advance_progress(stale, "c1", c1)
    stale.asked_questions.append("Q-from-the-deleted-lesson")

    run(teaching.reset_class("gp-inflight", "c1", c1))
    run(teaching._save_memory("gp-inflight", "c1", c1, stale))

    after = run(store.get_memory("gp-inflight"))
    assert after.class_progress["c1"].turn_count == 0
    assert after.asked_questions == []

def test_end_class_is_idempotent():
    c1 = ClassUnit(class_id="c1", title="Optics", objective="Understand optics.")
    run(store.save_path(_path("gp-idempotent", c1, topic="Physics")))
    first = run(teaching.end_class("gp-idempotent", "c1", c1))
    completed_at = first.class_progress["c1"].completed_at
    second = run(teaching.end_class("gp-idempotent", "c1", c1))
    assert second.class_progress["c1"].completed_at == completed_at
    assert second.covered_concepts.count("Optics") == 1
