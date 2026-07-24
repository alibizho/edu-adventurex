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

from ..agents.student import student_turn
from ..agents.targeted import generate_targeted_questions
from ..confusion import engine
from ..schemas import (
    AudioClassTeachResponse,
    ChunkAnalysis,
    ClassProgressRecord,
    ClassTeachResponse,
    ClassUnit,
    PathMemory,
    QAEntry,
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


def _advance_progress(memory: PathMemory, class_id: str) -> ClassProgressRecord:
    now = time.time()
    progress = memory.class_progress.get(class_id) or ClassProgressRecord()
    if progress.started_at is None:
        progress.started_at = now
    progress.status = "in_progress"
    progress.turn_count += 1
    progress.readiness = min(95, 25 + progress.turn_count * 15)
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
    _advance_progress(memory, class_id)

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


async def class_audio_turn(
    path_id: str,
    class_id: str,
    cls: ClassUnit,
    analysis: ChunkAnalysis,
    degraded: bool,
) -> AudioClassTeachResponse:
    """Use the ml-service transcript and confidence in one atomic teaching turn."""
    if degraded or not analysis.text.strip():
        return AudioClassTeachResponse(analysis=analysis, degraded=True)

    session_id = class_session_id(path_id, class_id)
    transcript = await store.get_transcript(session_id)
    resp = await student_turn(transcript, analysis.text.strip())
    analysis.chunk_id = resp.new_segment.id
    await store.append_segment(session_id, resp.new_segment)
    await store.append_analysis(session_id, analysis)

    memory = await store.get_memory(path_id)
    _advance_progress(memory, class_id)
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

    if question is not None:
        await store.record_questions(session_id, [question])
        if question.text not in memory.asked_questions:
            memory.asked_questions.append(question.text)

    await store.update_memory(path_id, memory)
    return AudioClassTeachResponse(
        student_reply=question.text if question else resp.student_reply,
        new_segment=resp.new_segment,
        analysis=analysis,
        asked=question is not None,
        question=question,
        degraded=False,
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
    progress.readiness = 100
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

    await store.update_memory(path_id, memory)
    return memory
