"""Teaching a class = the existing pipeline, reused, plus cross-class memory.

The learner teaches a class to the in-character AI student (`student_turn`). We run each utterance
through the confusion mock (`engine`) and, ONLY when it sounds unconfident, generate one targeted
question (`generate_targeted_questions`). Cross-class memory (`PathMemory`) is injected as extra
"already asked" history so the AI doesn't repeat near-duplicate questions in later classes — it
only asks a fresh angle when the learner keeps sounding unsure.

Each class runs on its own store session: session_id = "<path_id>:<class_id>".
"""
from __future__ import annotations

import time

from ..agents.student import next_segment, student_turn
from ..agents.targeted import generate_targeted_questions
from ..config import settings
from ..confusion import engine
from . import mastery
from ..schemas import (
    AudioClassTeachResponse,
    ChunkAnalysis,
    ClassObjective,
    ClassProgressRecord,
    ClassTeachResponse,
    ClassUnit,
    PathMemory,
    QAEntry,
    Segment,
    TargetedQuestion,
)
from ..store import store


def class_session_id(path_id: str, class_id: str) -> str:
    return f"{path_id}:{class_id}"


def _memory_as_history(memory: PathMemory) -> list[QAEntry]:
    """Render cross-class asked questions as synthetic (prompt-only) history entries so the
    'do NOT repeat any of these' rule in generate_targeted_questions spans every class."""
    return [
        QAEntry(question=TargetedQuestion(id=-1, chunk_id=-1, text=q))
        for q in memory.asked_questions
    ]


def _readiness(cls: ClassUnit, progress: ClassProgressRecord) -> int:
    """Share of this class's objectives actually covered. Note what this is NOT: talking for longer
    no longer moves it. A learner who says nothing of substance for ten turns stays at 0."""
    checklist = cls.checklist()
    if not checklist:
        return 0
    covered = sum(1 for o in checklist if o.id in progress.covered_objectives)
    return round(100 * covered / len(checklist))


def _advance_progress(memory: PathMemory, class_id: str, cls: ClassUnit) -> ClassProgressRecord:
    now = time.time()
    progress = memory.class_progress.get(class_id) or ClassProgressRecord()
    if progress.started_at is None:
        progress.started_at = now
    progress.status = "in_progress"
    progress.turn_count += 1
    progress.readiness = _readiness(cls, progress)
    memory.class_progress[class_id] = progress
    return progress


def _apply_curriculum_update(memory: PathMemory, analysis: ChunkAnalysis) -> None:
    update = analysis.curriculum_update
    if update is None:
        return
    for concept in update.added_concepts:
        clean = concept.strip()
        if clean and clean not in memory.expanded_concepts:
            memory.expanded_concepts.append(clean)


async def _question_from_analysis(
    session_id: str,
    analysis: ChunkAnalysis,
    memory: PathMemory,
) -> TargetedQuestion | None:
    generated = analysis.student_question
    if generated is None or not generated.question_text.strip():
        return None
    text = generated.question_text.strip()
    if any(existing.casefold() == text.casefold() for existing in memory.asked_questions):
        return None
    return TargetedQuestion(
        id=await store.next_question_id(session_id),
        chunk_id=analysis.chunk_id,
        text=text,
        anomaly_type=generated.anomaly_type,
        rationale=f"GPU confusion signal on {generated.target_concept}",
    )


async def class_teach_turn(
    path_id: str, class_id: str, cls: ClassUnit, latest_utterance: str
) -> ClassTeachResponse:
    session_id = class_session_id(path_id, class_id)

    transcript = await store.get_transcript(session_id)
    resp = await student_turn(transcript, latest_utterance)
    await store.append_segment(session_id, resp.new_segment)

    memory = await store.get_memory(path_id)
    _advance_progress(memory, class_id, cls)

    # Confusion gate over just-spoken utterance (mock heuristic — the SAME gate as
    # /questions/from_chunk, via engine.is_confused, so the two paths can't drift).
    analysis = engine.analyze([latest_utterance])[0]
    analysis.chunk_id = resp.new_segment.id
    if not engine.is_confused(analysis):
        await store.update_memory(path_id, memory)
        return ClassTeachResponse(
            student_reply=resp.student_reply, new_segment=resp.new_segment, asked=False
        )

    history = await store.get_history(session_id) + _memory_as_history(memory)
    topic = f"{cls.title} — {cls.objective}"
    questions = await generate_targeted_questions(
        [analysis], history, start_id=await store.next_question_id(session_id), topic=topic
    )
    if not questions:
        await store.update_memory(path_id, memory)
        return ClassTeachResponse(
            student_reply=resp.student_reply, new_segment=resp.new_segment, asked=False
        )

    await store.record_questions(session_id, questions)
    memory.asked_questions.append(questions[0].text)
    await store.update_memory(path_id, memory)
    return ClassTeachResponse(
        student_reply=resp.student_reply,
        new_segment=resp.new_segment,
        asked=True,
        question=questions[0],
    )


def _probe_is_due(progress: ClassProgressRecord) -> bool:
    """Throttle for goal nudges. Deliberately separate from `objective_check_every`: judging what
    was covered is the expensive batched call, whereas steering only needs the turn counter, and
    tying them together is why a probe used to need 3 chunks AND 4 turns and so never fired."""
    return progress.turn_count - progress.last_goal_probe_turn >= settings.goal_probe_cooldown


async def _goal_probe_question(
    session_id: str,
    cls: ClassUnit,
    objective: ClassObjective,
    transcript: list[Segment],
    memory: PathMemory,
) -> TargetedQuestion | None:
    """A student asking about something the learner hasn't covered yet."""
    text = await mastery.goal_probe(cls, objective, mastery.transcript_excerpt(transcript))
    if not text or text in memory.asked_questions:
        return None
    return TargetedQuestion(
        id=await store.next_question_id(session_id),
        chunk_id=transcript[-1].id if transcript else 0,
        text=text,
        anomaly_type="uncovered_goal",
        rationale=f"objective not yet covered: {objective.text}",
    )


async def class_audio_turn(
    path_id: str,
    class_id: str,
    cls: ClassUnit,
    analysis: ChunkAnalysis,
    degraded: bool,
    silent: bool = False,
) -> AudioClassTeachResponse:
    """Use the ml-service transcript and confidence in one atomic teaching turn.

    `silent` is the live-classroom mode: the kid talks continuously and every pause ships a chunk,
    so the utterance is still transcribed, stored and analyzed — but the class only speaks when a
    question actually fires. Without it each chunk costs a reply LLM call and the room ends up
    several sentences behind the teacher. Set it False for the one-to-one flow, where the student
    is expected to answer whatever was just said to them.
    """
    if degraded or not analysis.text.strip():
        return AudioClassTeachResponse(analysis=analysis, degraded=True)

    session_id = class_session_id(path_id, class_id)
    transcript = await store.get_transcript(session_id)
    # Build the segment before deciding on a reply: the question generator below reads the stored
    # transcript, and it must see the same spine whether or not the student ends up speaking.
    segment = next_segment(transcript, analysis.text.strip())
    analysis.chunk_id = segment.id
    await store.append_segment(session_id, segment)
    await store.append_analysis(session_id, analysis)

    memory = await store.get_memory(path_id)
    _advance_progress(memory, class_id, cls)
    _apply_curriculum_update(memory, analysis)
    # Prefer the question the GPU already generated from the strongest anomaly; only fall back to
    # the LLM question generator when it didn't produce one (or it duplicated an earlier probe).
    question = await _question_from_analysis(session_id, analysis, memory)
    if question is None and engine.is_confused(analysis):
        history = await store.get_history(session_id) + _memory_as_history(memory)
        topic = f"{cls.title} - {cls.objective}"
        questions = await generate_targeted_questions(
            [analysis],
            history,
            start_id=await store.next_question_id(session_id),
            topic=topic,
        )
        if questions:
            question = questions[0]

    # Nothing wrong with what they said — but silence is not feedback. If goals are still open,
    # a student steers toward the next one; that turns dead air into direction, and is the only
    # signal the learner gets that the room is following at all.
    progress = memory.class_progress[class_id]
    open_objectives = [o for o in cls.checklist() if o.id not in progress.covered_objectives]
    if question is None and open_objectives and _probe_is_due(progress):
        question = await _goal_probe_question(session_id, cls, open_objectives[0], transcript, memory)
        if question is not None:
            progress.last_goal_probe_turn = progress.turn_count

    if question is not None:
        await store.record_questions(session_id, [question])
        if question.text not in memory.asked_questions:
            memory.asked_questions.append(question.text)
        student_reply = question.text
    elif silent:
        student_reply = ""          # the class listened; nobody had anything to ask.
    else:
        student_reply = (await student_turn(transcript, analysis.text.strip())).student_reply

    await store.update_memory(path_id, memory)
    return AudioClassTeachResponse(
        student_reply=student_reply,
        new_segment=segment,
        analysis=analysis,
        asked=question is not None,
        question=question,
        degraded=False,
        # No question and nothing left to cover: say so, so the UI can show that the class is
        # following rather than leaving the learner guessing whether anything is working.
        all_goals_covered=not open_objectives,
    )


async def end_class(
    path_id: str,
    class_id: str,
    cls: ClassUnit,
    completion_mode: str = "self-teaching",
) -> PathMemory:
    """'End class' action: fold the class into cross-class memory. The class title becomes a
    covered concept; answered probes count as understood, unanswered ones as still-struggled
    (so a later class may re-probe them from a new angle)."""
    session_id = class_session_id(path_id, class_id)
    memory = await store.get_memory(path_id)

    progress = memory.class_progress.get(class_id) or ClassProgressRecord()
    if progress.status == "complete":
        return memory
    now = time.time()
    progress.status = "complete"
    # Readiness stays as earned. Stamping 100 on every completed class was what made the whole
    # metric meaningless — "complete" says the learner stopped, `passed_on_mastery` says they got it.
    progress.readiness = _readiness(cls, progress)
    progress.passed_on_mastery = all(
        o.id in progress.covered_objectives for o in cls.checklist()
    )
    progress.started_at = progress.started_at or now
    progress.completed_at = now
    progress.completion_mode = (
        "guided-explanation" if completion_mode == "guided-explanation" else "self-teaching"
    )
    memory.class_progress[class_id] = progress

    if cls.title not in memory.covered_concepts:
        memory.covered_concepts.append(cls.title)

    for entry in await store.get_history(session_id):
        label = entry.question.text
        if entry.answer:
            # answered wins: it's understood, and no longer counts as struggled
            if label not in memory.understood:
                memory.understood.append(label)
            if label in memory.struggled:
                memory.struggled.remove(label)
        elif label not in memory.struggled and label not in memory.understood:
            memory.struggled.append(label)

    # Objectives the learner demonstrably explained are understood across the whole path; the ones
    # left open are what a later class should come back to.
    for objective in cls.checklist():
        target = memory.understood if objective.id in progress.covered_objectives else memory.struggled
        if objective.text not in target:
            target.append(objective.text)

    await store.update_memory(path_id, memory)
    return memory


# ---- objective mastery (background) ----


async def run_objective_check(path_id: str, class_id: str, cls: ClassUnit) -> None:
    """Judge which objectives the learner has now covered, and let a student chase one that's open.

    Runs as a FastAPI background task after the teaching response has already been sent, batched
    over several utterances (`settings.objective_check_every`). Two LLM calls at most, and only
    when there is new speech to judge — the teaching loop never waits on this.

    Failures are swallowed: a missed check just means the checkmarks land one batch later, which
    is not worth breaking a live class over.
    """
    try:
        session_id = class_session_id(path_id, class_id)
        transcript = await store.get_transcript(session_id)
        if not transcript:
            return

        memory = await store.get_memory(path_id)
        progress = memory.class_progress.get(class_id) or ClassProgressRecord()
        fresh = [s for s in transcript if s.id > progress.last_checked_segment]
        if not mastery.should_check(len(fresh)):
            return

        checklist = cls.checklist()
        open_objectives = [o for o in checklist if o.id not in progress.covered_objectives]
        progress.last_checked_segment = max(s.id for s in transcript)

        if open_objectives:
            covered = await mastery.judge_coverage(
                cls, open_objectives, mastery.transcript_excerpt(fresh)
            )
            for objective_id, evidence in covered.items():
                if objective_id not in progress.covered_objectives:
                    progress.covered_objectives.append(objective_id)
                progress.objective_evidence[objective_id] = evidence
            open_objectives = [o for o in open_objectives if o.id not in covered]

        progress.readiness = _readiness(cls, progress)
        # Steering lives on the teaching turn itself (see class_audio_turn) so the room reacts
        # while the learner is still there, not one batch later.
        memory.class_progress[class_id] = progress
        await store.update_memory(path_id, memory)
    except Exception as exc:  # noqa: BLE001 — a background nicety must never kill a live class
        print(f"[mastery] objective check failed for {path_id}:{class_id} ({exc!r})")
