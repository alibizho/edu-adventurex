"""Teaching a class = the existing pipeline, reused, plus cross-class memory.

The learner teaches a class to the in-character AI student (`student_turn`). We run each utterance
through the confusion mock (`engine`) and, ONLY when it sounds unconfident, generate one targeted
question (`generate_targeted_questions`). Cross-class memory (`PathMemory`) is injected as extra
"already asked" history so the AI doesn't repeat near-duplicate questions in later classes — it
only asks a fresh angle when the learner keeps sounding unsure.

Each class runs on its own store session: session_id = "<path_id>:<class_id>".
"""
from __future__ import annotations

from ..agents.student import student_turn
from ..agents.targeted import generate_targeted_questions
from ..confusion import engine
from ..schemas import ClassTeachResponse, ClassUnit, PathMemory, QAEntry, TargetedQuestion
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


async def class_teach_turn(
    path_id: str, class_id: str, cls: ClassUnit, latest_utterance: str
) -> ClassTeachResponse:
    session_id = class_session_id(path_id, class_id)

    transcript = await store.get_transcript(session_id)
    resp = await student_turn(transcript, latest_utterance)
    await store.append_segment(session_id, resp.new_segment)

    # Confusion gate over just-spoken utterance (mock heuristic — the SAME gate as
    # /questions/from_chunk, via engine.is_confused, so the two paths can't drift).
    analysis = engine.analyze([latest_utterance])[0]
    analysis.chunk_id = resp.new_segment.id
    if not engine.is_confused(analysis):
        return ClassTeachResponse(
            student_reply=resp.student_reply, new_segment=resp.new_segment, asked=False
        )

    memory = await store.get_memory(path_id)
    history = await store.get_history(session_id) + _memory_as_history(memory)
    topic = f"{cls.title} — {cls.objective}"
    questions = await generate_targeted_questions(
        [analysis], history, start_id=await store.next_question_id(session_id), topic=topic
    )
    if not questions:
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


async def end_class(path_id: str, class_id: str, cls: ClassUnit) -> PathMemory:
    """'End class' action: fold the class into cross-class memory. The class title becomes a
    covered concept; answered probes count as understood, unanswered ones as still-struggled
    (so a later class may re-probe them from a new angle)."""
    session_id = class_session_id(path_id, class_id)
    memory = await store.get_memory(path_id)

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
