"""Targeted-question agent. Given the lowest-confidence chunks from the confusion
engine plus the Q&A history, it writes specialized questions probing exactly those weak spots, and
never repeats a question already asked. One LLM call per round — kept as a plain async function so
it can become a LangGraph node later if the loop grows an adaptive branch.
"""
from ..llm import generator_chat
from ..schemas import ChunkAnalysis, QAEntry, TargetedQuestion
from .parsing import extract_json_array
from .prompts import STUDENT_EXPLAIN_SYSTEM, TARGETED_QUESTION_SYSTEM


def _render_goals(objectives: list[str] | None) -> str:
    """The class checklist, so a question lands on what this class is FOR. Without it the generator
    only knows the one-sentence class objective, and a thin chunk has nothing to be anchored to."""
    goals = [o.strip() for o in (objectives or []) if o and o.strip()]
    if not goals:
        return ""
    listed = "\n".join(f"- {g}" for g in goals)
    return f"CLASS GOALS (what this class must get the learner to explain):\n{listed}\n\n"


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
    transcript: str = "",
    focus_target: str = "",
    parent_id: int | None = None,
    objectives: list[str] | None = None,
) -> list[TargetedQuestion]:
    """Return one specialized, non-repeating question per low-confidence chunk. `start_id` is where
    to begin numbering the returned questions (the store assigns the real ids on record). `topic`
    is the lesson topic the teacher declared; when given it grounds the questions in that topic.

    `transcript`, `objectives` and `focus_target` are what keep a series of questions feeling like
    one conversation about a subject: without them the generator sees a single orphaned fragment
    plus a list of questions it is forbidden to repeat, and the only way to satisfy that is to
    change the subject — or to seize on whichever word the fragment flagged, which is how a
    question about nothing gets asked.
    """
    if not chunks:
        return []

    topic_line = f"LESSON TOPIC: {topic}\n\n" if topic else ""
    recent_line = f"RECENTLY SAID (the thread so far):\n{transcript}\n\n" if transcript else ""
    focus_line = (
        f"CURRENT WEAK SPOT (keep pursuing this): {focus_target}\n\n" if focus_target else ""
    )
    user = (
        f"{topic_line}"
        f"{_render_goals(objectives)}"
        f"{recent_line}"
        f"{focus_line}"
        f"LOW-CONFIDENCE CHUNKS:\n{_render_chunks(chunks)}\n\n"
        f"HISTORY (do NOT repeat any of these verbatim):\n{_render_history(history)}"
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
                answer_key=(str(it.get("answer_key", "")).strip() or None),
                parent_id=parent_id,
            )
        )
    return questions


# Said when the generator is unreachable. Better than silence, and it keeps the promise the
# conversation just made — that the learner is about to be told, not asked again.
EXPLAIN_FALLBACK = "LET'S COME BACK TO THAT ONE TOGETHER — I COULDN'T PIN IT DOWN EITHER."


async def explain_answer(
    question: TargetedQuestion,
    topic: str = "",
    transcript: str = "",
    objectives: list[str] | None = None,
) -> str:
    """Answer our own question, out loud, because the learner is stuck.

    The deliberate exception to the never-explain rule (see agents/prompts.py). It fires only once
    the learner has said they don't know or has spent their tries: at that point another question
    is not a probe, it is a wall. `question.answer_key` is the ground truth when the generator
    wrote the question; GPU-relayed questions have none, and the model answers from the topic.
    """
    answer_line = (
        f"ANSWER (ground truth):\n{question.answer_key.strip()}\n\n"
        if (question.answer_key or "").strip()
        else "ANSWER: not recorded — answer from established knowledge of the topic.\n\n"
    )
    user = (
        f"LESSON TOPIC: {topic}\n\n" if topic else ""
    ) + (
        f"{_render_goals(objectives)}"
        f"QUESTION YOU ASKED:\n{question.text}\n\n"
        f"{answer_line}"
        + (f"WHAT THEY HAVE BEEN SAYING:\n{transcript}\n" if transcript else "")
    )
    said = (await generator_chat(STUDENT_EXPLAIN_SYSTEM, user, temperature=0.4)).strip()
    return said or EXPLAIN_FALLBACK
