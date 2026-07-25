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
    ClassObjective,
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


async def _never_probe(cls, objective, transcript):
    """Default for tests that aren't exercising goal nudges."""
    return ""


def _covers(*objective_ids: str):
    """A judge_coverage stub that always credits exactly these objective ids."""
    async def fake(cls, open_objectives, transcript):
        open_ids = {o.id for o in open_objectives}
        return {oid: f"evidence for {oid}" for oid in objective_ids if oid in open_ids}
    return fake


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
                    ClassUnit(
                        class_id="c1",
                        title="Forces",
                        objective="Define force.",
                        objectives=[
                            ClassObjective(id="o1", text="Explain what a force does to motion."),
                            ClassObjective(id="o2", text="Explain why forces come in pairs."),
                        ],
                    ),
                    # No objectives: a class from a plan built before they existed.
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


def _silent_class(monkeypatch, text: str, confidence: float, *, probe: bool = False):
    """Build a plan and stub the GPU so one silent audio chunk can be posted into it. Returns
    (path_id, response_body, replies) where `replies` counts reply-LLM calls.

    `probe=False` stubs the goal probe out, so a test can look at the confusion path alone."""
    _stub_llm()(monkeypatch)
    replies: list[str] = []

    async def counting_student_turn(transcript, utterance):
        from app.schemas import Segment as Segment_
        replies.append(utterance)
        nid = max((s.id for s in transcript), default=-1) + 1
        return TeachTurnResponse(
            student_reply="hmm, can you say more?",
            new_segment=Segment_(id=nid, idx=len(transcript), text=utterance),
        )

    async def fake_analyze_audio(audio_bytes, **kwargs):
        return ChunkAnalysis(chunk_id=kwargs["chunk_id"], text=text, confidence=confidence), False

    async def fake_probe(cls, objective, transcript):
        return f"steer:{objective.id}" if probe else ""

    monkeypatch.setattr(teaching, "student_turn", counting_student_turn)
    monkeypatch.setattr(teaching.mastery, "goal_probe", fake_probe)
    monkeypatch.setattr(plan_routes.client, "analyze_audio_with_status", fake_analyze_audio)

    pid = client.post("/plan/build", json={
        "original_input": "mechanics", "confirmed_topic": "Mechanics", "num_classes": 2,
    }).json()["path_id"]
    client.post(f"/plan/{pid}/class/c1/notes")
    response = client.post(
        f"/plan/{pid}/class/c1/teach/audio-turn",
        data={"chunk_id": 0, "history": "[]", "silent": "true"},
        files={"audio": ("chunk.wav", b"RIFFfakewavdata", "audio/wav")},
    )
    assert response.status_code == 200, response.text
    return pid, response.json(), replies


def test_silent_audio_turn_records_the_segment_without_paying_for_a_reply(monkeypatch):
    """The live classroom ships a chunk on every pause. A clear one must still land on the
    transcript — the end-of-class measurement reads it — but must not cost a reply LLM call."""
    pid, body, replies = _silent_class(monkeypatch, "A force is a push or a pull.", 0.95)

    assert body["asked"] is False and body["question"] is None
    assert body["student_reply"] == ""
    assert replies == [], "silent mode must not call the student reply LLM"

    segments = client.get(f"/sessions/{pid}:c1").json()["transcript"]
    assert [s["text"] for s in segments] == ["A force is a push or a pull."]
    assert segments[0]["id"] == body["new_segment"]["id"]


def test_a_clear_chunk_steers_toward_a_goal_instead_of_saying_nothing(monkeypatch):
    """The reported bug: teaching well produced no question, no reply and no reaction, so you
    couldn't tell whether you were doing fine or the app was broken. A clear chunk with goals
    still open must now come back with a student steering toward one — and still without paying
    for a generic reply."""
    _pid, body, replies = _silent_class(
        monkeypatch, "A force is a push or a pull.", 0.95, probe=True
    )

    assert body["asked"] is True
    assert body["question"]["anomaly_type"] == "uncovered_goal"
    assert body["question"]["text"] == "steer:o1", "steers to the first uncovered objective"
    assert body["student_reply"] == body["question"]["text"]
    assert body["all_goals_covered"] is False
    assert replies == [], "steering replaces the reply; it must not also cost a reply call"


def test_silent_audio_turn_still_raises_a_hand_when_confused(monkeypatch):
    """Silence is only the default. A confused chunk must still produce the question that puts a
    ? over a student's head — and that question, not a generic reply, is what the class says."""
    _pid, body, replies = _silent_class(monkeypatch, "um, i think force is maybe a push?", 0.2)

    assert body["asked"] is True
    assert body["question"]["text"].startswith("probe:")
    assert body["student_reply"] == body["question"]["text"]
    assert replies == [], "the question replaces the reply; it must not also cost an LLM call"


# --------------------------------------------------------------------------- #
# objective mastery
# --------------------------------------------------------------------------- #

def _teach_chunks(pid: str, class_id: str, texts: list[str], monkeypatch):
    """Post N silent audio chunks, stubbing the GPU to return each text as a clear transcript."""
    queue = list(texts)

    async def fake_analyze(audio_bytes, **kwargs):
        return ChunkAnalysis(chunk_id=kwargs["chunk_id"], text=queue.pop(0), confidence=0.95), False

    monkeypatch.setattr(plan_routes.client, "analyze_audio_with_status", fake_analyze)
    for i in range(len(texts)):
        r = client.post(
            f"/plan/{pid}/class/{class_id}/teach/audio-turn",
            data={"chunk_id": i, "history": "[]", "silent": "true"},
            files={"audio": ("chunk.wav", b"RIFFfakewavdata", "audio/wav")},
        )
        assert r.status_code == 200, r.text


def _build_with_notes(class_id: str = "c1") -> str:
    pid = client.post("/plan/build", json={
        "original_input": "mechanics", "confirmed_topic": "Mechanics", "num_classes": 2,
    }).json()["path_id"]
    client.post(f"/plan/{pid}/class/{class_id}/notes")
    return pid


def test_covering_an_objective_moves_readiness_off_the_turn_counter(monkeypatch):
    """Readiness must reflect what was explained, not how long they spoke. Two of c1's objectives
    exist; covering exactly one puts readiness at 50% no matter how many turns it took."""
    _stub_llm()(monkeypatch)

    async def fake_judge(cls, open_objectives, transcript):
        # Credit only the first objective, and only once it has actually been mentioned.
        if "motion" in transcript:
            return {"o1": "a force changes how something moves"}
        return {}

    monkeypatch.setattr(teaching.mastery, "judge_coverage", fake_judge)
    monkeypatch.setattr(teaching.mastery, "goal_probe", _never_probe)

    pid = _build_with_notes()
    _teach_chunks(pid, "c1", ["a force changes motion", "and it has a direction", "that is it"], monkeypatch)

    progress = client.get(f"/plan/{pid}/memory").json()["class_progress"]["c1"]
    assert progress["covered_objectives"] == ["o1"]
    assert progress["objective_evidence"]["o1"] == "a force changes how something moves"
    assert progress["readiness"] == 50, "1 of 2 objectives — not 25 + turns * 15"


def test_ending_without_covering_everything_is_not_a_pass(monkeypatch):
    """'Complete' means they stopped; passed_on_mastery means they got it. Stamping 100 on every
    finished class is what made readiness meaningless in the first place."""
    _stub_llm()(monkeypatch)
    monkeypatch.setattr(teaching.mastery, "judge_coverage", _covers("o1"))
    monkeypatch.setattr(teaching.mastery, "goal_probe", _never_probe)

    pid = _build_with_notes()
    _teach_chunks(pid, "c1", ["one", "two", "three"], monkeypatch)
    memory = client.post(f"/plan/{pid}/class/c1/end").json()

    progress = memory["class_progress"]["c1"]
    assert progress["status"] == "complete"
    assert progress["passed_on_mastery"] is False
    assert progress["readiness"] == 50
    # The open objective is what a later class should come back to.
    assert "Explain why forces come in pairs." in memory["struggled"]
    assert "Explain what a force does to motion." in memory["understood"]


def test_covering_everything_passes_the_class(monkeypatch):
    _stub_llm()(monkeypatch)
    monkeypatch.setattr(teaching.mastery, "judge_coverage", _covers("o1", "o2"))
    monkeypatch.setattr(teaching.mastery, "goal_probe", _never_probe)

    pid = _build_with_notes()
    _teach_chunks(pid, "c1", ["one", "two", "three"], monkeypatch)

    progress = client.get(f"/plan/{pid}/memory").json()["class_progress"]["c1"]
    assert progress["readiness"] == 100
    memory = client.post(f"/plan/{pid}/class/c1/end").json()
    assert memory["class_progress"]["c1"]["passed_on_mastery"] is True


def test_class_without_objectives_still_teaches_and_ends(monkeypatch):
    """c2 has no objectives — a plan built before they existed. It must fall back to the
    one-sentence goal rather than showing an empty checklist or dividing by zero."""
    _stub_llm()(monkeypatch)
    monkeypatch.setattr(teaching.mastery, "judge_coverage", _covers("o1"))
    monkeypatch.setattr(teaching.mastery, "goal_probe", _never_probe)

    pid = _build_with_notes("c2")
    _teach_chunks(pid, "c2", ["one", "two", "three"], monkeypatch)

    progress = client.get(f"/plan/{pid}/memory").json()["class_progress"]["c2"]
    assert progress["readiness"] == 100, "the single fallback objective was covered"
    memory = client.post(f"/plan/{pid}/class/c2/end").json()
    assert memory["class_progress"]["c2"]["passed_on_mastery"] is True
    assert "Define energy." in memory["understood"]


def test_a_student_asks_about_an_objective_that_is_still_open(monkeypatch):
    """The guidance half: nobody is confused, but a goal is untouched, so someone raises a hand.

    Steering is throttled by `goal_probe_cooldown` turns, NOT by the coverage-check batch size.
    Tying the two together is what made probes effectively never fire — they needed 3 chunks to
    accumulate *and* 4 turns to elapse. Here: some turns steer, not all of them, and never twice
    about the same thing."""
    _stub_llm()(monkeypatch)
    monkeypatch.setattr(teaching.mastery, "judge_coverage", _covers())     # nothing gets covered
    seen: list[str] = []

    async def fake_probe(cls, objective, transcript):
        seen.append(objective.text)
        return f"but why {objective.text.lower()}? ({len(seen)})"

    monkeypatch.setattr(teaching.mastery, "goal_probe", fake_probe)

    pid = _build_with_notes()
    _teach_chunks(pid, "c1", ["one", "two", "three", "four", "five"], monkeypatch)

    assert set(seen) == {"Explain what a force does to motion."}, "always the first OPEN objective"
    questions = client.get(f"/sessions/{pid}:c1").json()["questions"]
    probes = [q["question"] for q in questions if q["question"]["anomaly_type"] == "uncovered_goal"]
    assert 0 < len(probes) < 5, "the room steers periodically, not on every single utterance"
    assert len({p["text"] for p in probes}) == len(probes), "no repeated nudges"


def test_notes_are_generated_once_and_reused(monkeypatch):
    """The expensive call in the plan surface. Re-entering a class must not rewrite its primer."""
    _stub_llm()(monkeypatch)
    calls: list[str] = []

    async def counting_notes(path, cls, memory):
        calls.append(cls.class_id)
        return f"# {cls.title}\nprimer {len(calls)}"

    monkeypatch.setattr(plan_routes, "generate_class_notes", counting_notes)

    pid = client.post("/plan/build", json={
        "original_input": "mechanics", "confirmed_topic": "Mechanics", "num_classes": 2,
    }).json()["path_id"]

    first = client.post(f"/plan/{pid}/class/c1/notes").json()
    second = client.post(f"/plan/{pid}/class/c1/notes").json()
    assert calls == ["c1"], "the second call must reuse the saved notes"
    assert second["teacher_notes"] == first["teacher_notes"]

    forced = client.post(f"/plan/{pid}/class/c1/notes?regenerate=true").json()
    assert calls == ["c1", "c1"]
    assert forced["teacher_notes"] != first["teacher_notes"]
