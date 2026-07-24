"""Targeted-question agent (newTR.md §6.3). Given the lowest-confidence chunks from the confusion
engine plus the Q&A history, it writes specialized questions probing exactly those weak spots, and
never repeats a question already asked. One LLM call per round — kept as a plain async function so
it can become a LangGraph node later if the loop grows an adaptive branch.
"""
from ..llm import generator_chat
from ..schemas import ChunkAnalysis, QAEntry, TargetedQuestion
from .parsing import extract_json_array
from .prompts import TARGETED_QUESTION_SYSTEM


def _render_chunks(chunks: list[ChunkAnalysis]) -> str:
    lines = []
    for c in chunks:
        anoms = ", ".join(a.type for a in c.anomalies) or "none"
        lines.append(f'chunk {c.chunk_id} (confidence {c.confidence:.2f}, anomalies: {anoms}): "{c.text}"')
    return "\n".join(lines)


def _render_history(history: list[QAEntry]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for e in history:
        ans = e.answer if e.answer else "(unanswered)"
        lines.append(f'- asked (chunk {e.question.chunk_id}): "{e.question.text}"  | answer: {ans}')
    return "\n".join(lines)


async def generate_targeted_questions(
    chunks: list[ChunkAnalysis],
    history: list[QAEntry],
    start_id: int = 0,
    topic: str | None = None,
) -> list[TargetedQuestion]:
    """Return one specialized, non-repeating question per low-confidence chunk. `start_id` is where
    to begin numbering the returned questions (the store assigns the real ids on record). `topic`
    is the lesson topic the teacher declared; when given it grounds the questions in that topic."""
    if not chunks:
        return []

    topic_line = f"LESSON TOPIC: {topic}\n\n" if topic else ""
    user = (
        f"{topic_line}"
        f"LOW-CONFIDENCE CHUNKS:\n{_render_chunks(chunks)}\n\n"
        f"HISTORY (do NOT repeat any of these):\n{_render_history(history)}"
    )
    raw = await generator_chat(TARGETED_QUESTION_SYSTEM, user, temperature=0.6)

    valid_ids = {c.chunk_id for c in chunks}
    questions: list[TargetedQuestion] = []
    for it in extract_json_array(raw):
        if not (isinstance(it, dict) and str(it.get("text", "")).strip()):
            continue
        try:
            chunk_id = int(it.get("chunk_id"))
        except (TypeError, ValueError):
            continue
        if chunk_id not in valid_ids:
            continue
        questions.append(
            TargetedQuestion(
                id=start_id + len(questions),
                chunk_id=chunk_id,
                text=str(it["text"]).strip(),
                anomaly_type=(str(it.get("anomaly_type", "")).strip() or None),
                rationale=(str(it.get("rationale", "")).strip() or None),
            )
        )
    return questions
