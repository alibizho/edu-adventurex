"""Full end-to-end workflow test for the /plan surface, LLM stubbed (hermetic — no network, no
Postgres). Drives scope -> build -> notes -> teach/turn (confident + hedged) -> end across TWO
classes and asserts the whole flow hangs together:

  - scope returns a structured TopicScope
  - build returns an ordered GrowthPath, saved (GET finds it)
  - notes fills teacher_notes (Markdown) and persists
  - a confident teach turn asks nothing; a hedged one fires a question
  - the fired question is recorded and remembered across classes (no near-duplicate)
  - end folds the class into PathMemory (covered / struggled)

Run:  .venv/bin/python -m pytest tests/test_workflow.py -q
"""
import asyncio
import os

os.environ["STORE_BACKEND"] = "memory"
os.environ["ML_SERVICE_URL"] = "http://127.0.0.1:9"  # unreachable -> neutral degrade

from fastapi.testclient import TestClient  # noqa: E402

import app.curriculum.build as build  # noqa: E402
import app.curriculum.teaching as teaching  # noqa: E402
import app.api.plan_routes as plan_routes  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import (  # noqa: E402
    ClassUnit,
    ChunkAnalysis,
    CurriculumUpdate,
    GrowthPath,
    PathMemory,
    ScopeSuggestion,
    StudentQuestion,
    TeachTurnResponse,
    TargetedQuestion,
    TopicScope,
)

client = TestClient(app)


def _stub_llm():
    """Stub the LLM-touching functions with deterministic fakes. scope_topic / generate_class_notes
    are imported by name into plan_routes, so they must be patched on plan_routes; structure_curriculum
    is called inside build.py (so patch build.structure_curriculum and let the real build_plan run);
    student_turn / generate_targeted_questions are called inside teaching.py (patch teaching.*).
    The question generator records every history it's handed so we can assert cross-class dedup."""

    def _install(monkeypatch):
        seen_histories: list[list[str]] = []

        async def fake_scope(req):
            return TopicScope(
                is_broad=False,
                suggestions=[],
                confirmed_topic=req.original_input.title(),
                suggested_classes=2,
            )

        async def fake_structure(confirmed_topic, num_classes, material):
            return (
                [
                    ClassUnit(class_id="c1", title="Forces", objective="Define force."),
                    ClassUnit(class_id="c2", title="Energy", objective="Define energy.", prerequisites=["c1"]),
                ],
                ["c1", "c2"],
            )

        async def fake_notes(path, cls, memory):
            covered = ", ".join(memory.covered_concepts) or "none"
            return f"# {cls.title}\nCovered so far: {covered}\nA primer."

        async def fake_student_turn(transcript, utterance):
            from app.schemas import Segment as Segment_
            nid = max((s.id for s in transcript), default=-1) + 1
            return TeachTurnResponse(
                student_reply="hmm, can you say more?",
                new_segment=Segment_(id=nid, idx=len(transcript), text=utterance),
            )

        async def fake_generate(chunks, history, start_id=0, topic=None):
            seen_histories.append([e.question.text for e in history])
            return [TargetedQuestion(id=start_id, chunk_id=chunks[0].chunk_id, text=f"probe:{chunks[0].text[:8]}")]

        monkeypatch.setattr(plan_routes, "scope_topic", fake_scope)
        monkeypatch.setattr(build, "structure_curriculum", fake_structure)
        monkeypatch.setattr(plan_routes, "generate_class_notes", fake_notes)
        monkeypatch.setattr(teaching, "student_turn", fake_student_turn)
        monkeypatch.setattr(teaching, "generate_targeted_questions", fake_generate)
        return seen_histories

    return _install


def test_full_workflow_two_classes(monkeypatch):
    seen = _stub_llm()(monkeypatch)

    # 1. scope
    r = client.post("/plan/scope", json={"original_input": "mechanics"})
    assert r.status_code == 200
    scope = r.json()
    assert scope["is_broad"] is False and scope["suggested_classes"] == 2

    # 2. build
    r = client.post("/plan/build", json={
        "original_input": "mechanics", "confirmed_topic": "Mechanics", "num_classes": 2,
    })
    assert r.status_code == 200
    path = r.json()
    pid = path["path_id"]
    assert [c["class_id"] for c in path["classes"]] == ["c1", "c2"]
    assert all(c["notes_generated"] is False for c in path["classes"])

    # 3. GET finds the saved plan
    assert client.get(f"/plan/{pid}").json()["confirmed_topic"] == "Mechanics"
    assert any(item["path_id"] == pid for item in client.get("/plan").json())
    assert client.get(f"/plan/{pid}/memory").json()["path_id"] == pid

    # 4. notes for c1 (no earlier classes -> "none" in the primer)
    r = client.post(f"/plan/{pid}/class/c1/notes")
    assert r.status_code == 200
    assert r.json()["notes_generated"] is True
    assert "Covered so far: none" in r.json()["teacher_notes"]

    # 5a. confident turn -> no question
    r = client.post(f"/plan/{pid}/class/c1/teach/turn",
                    json={"latest_utterance": "A force is a push or a pull."})
    assert r.json()["asked"] is False and r.json()["question"] is None

    # 5b. hedged turn -> a question fires and is remembered
    r = client.post(f"/plan/{pid}/class/c1/teach/turn",
                    json={"latest_utterance": "um, i think a force is maybe kind of a push?"})
    hedged = r.json()
    assert hedged["asked"] is True and hedged["question"] is not None
    q1 = hedged["question"]["text"]

    # 6. end c1 -> memory now has the class covered + the unanswered probe as struggled
    r = client.post(f"/plan/{pid}/class/c1/end")
    mem = r.json()
    assert "Forces" in mem["covered_concepts"]
    assert q1 in mem["asked_questions"]
    assert q1 in mem["struggled"] and q1 not in mem["understood"]

    # 7. notes for c2 -> primer sees c1 as covered (cross-class context flows into notes)
    r = client.post(f"/plan/{pid}/class/c2/notes")
    assert "Forces" in r.json()["teacher_notes"]

    # 8. hedged turn in c2 -> question fires, and c1's question was handed in as history
    r = client.post(f"/plan/{pid}/class/c2/teach/turn",
                    json={"latest_utterance": "uh, energy is sort of the ability to do stuff?"})
    assert r.json()["asked"] is True
    assert q1 in seen[-1], "c1's question must appear in c2's question-generation history (dedup)"

    # 9. GPU unavailable is explicit and does not create a fake audio teaching turn.
    r = client.post(
        f"/plan/{pid}/class/c2/teach/audio-turn",
        data={"chunk_id": 1, "history": "[]"},
        files={"audio": ("chunk.wav", b"RIFFfakewavdata", "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["degraded"] is True
    assert r.json()["new_segment"] is None


def test_workflow_unknown_path_404(monkeypatch):
    _stub_llm()(monkeypatch)
    assert client.get("/plan/does-not-exist").status_code == 404
    assert client.post("/plan/does-not-exist/class/c1/notes").status_code == 404
    assert client.post("/plan/does-not-exist/class/c1/end").status_code == 404


def test_scope_broad_returns_suggestions(monkeypatch):
    """A broad topic returns exactly 3 narrower suggestions for the learner to pick from."""
    async def fake_scope(req):
        return TopicScope(
            is_broad=True,
            suggestions=[
                ScopeSuggestion(topic=f"Option {i}", rationale="r", suggested_classes=5)
                for i in range(1, 4)
            ],
            confirmed_topic="Option 1",
            suggested_classes=5,
        )
    monkeypatch.setattr(plan_routes, "scope_topic", fake_scope)
    r = client.post("/plan/scope", json={"original_input": "physics"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_broad"] is True
    assert len(body["suggestions"]) == 3
    assert all("topic" in s and "rationale" in s and "suggested_classes" in s for s in body["suggestions"])


def test_audio_turn_uses_gpu_question_and_persists_curriculum_update(monkeypatch):
    _stub_llm()(monkeypatch)
    captured: dict[str, object] = {}

    async def fake_analyze_audio(audio_bytes, **kwargs):
        captured.update(kwargs)
        return (
            ChunkAnalysis(
                chunk_id=kwargs["chunk_id"],
                text="Measurement can also lead us to decoherence.",
                confidence=0.68,
                student_question=StudentQuestion(
                    question_text="Why is decoherence not exactly the same as collapse?",
                    target_concept="decoherence",
                    anomaly_type="beyond",
                ),
                curriculum_update=CurriculumUpdate(
                    added_concepts=["environmental decoherence"]
                ),
            ),
            False,
        )

    monkeypatch.setattr(
        plan_routes.client,
        "analyze_audio_with_status",
        fake_analyze_audio,
    )

    built = client.post(
        "/plan/build",
        json={
            "original_input": "mechanics",
            "confirmed_topic": "Mechanics",
            "num_classes": 2,
        },
    ).json()
    pid = built["path_id"]
    notes = client.post(f"/plan/{pid}/class/c1/notes")
    assert notes.status_code == 200

    response = client.post(
        f"/plan/{pid}/class/c1/teach/audio-turn",
        data={"chunk_id": 4, "history": '["Earlier explanation"]'},
        files={"audio": ("chunk.wav", b"RIFFfakewavdata", "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert captured["overall_topic"] == "Mechanics"
    assert "Define force." in str(captured["curriculum_context"])
    assert "A primer." in str(captured["curriculum_context"])
    assert "Forces" in captured["key_concepts"]
    assert body["asked"] is True
    assert body["question"]["text"] == "Why is decoherence not exactly the same as collapse?"
    assert body["student_reply"] == body["question"]["text"]
    assert body["analysis"]["curriculum_update"]["added_concepts"] == [
        "environmental decoherence"
    ]

    memory = client.get(f"/plan/{pid}/memory").json()
    assert memory["expanded_concepts"] == ["environmental decoherence"]
    session = client.get(f"/sessions/{pid}:c1").json()
    stored = session["analyses"][0]
    assert stored["student_question"]["target_concept"] == "decoherence"
    assert stored["curriculum_update"]["added_concepts"] == [
        "environmental decoherence"
    ]
