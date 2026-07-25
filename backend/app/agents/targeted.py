from ..llm import generator_chat
from ..schemas import ChunkAnalysis, QAEntry, TargetedQuestion
from .parsing import extract_json_array
from .prompts import STUDENT_EXPLAIN_SYSTEM, TARGETED_QUESTION_SYSTEM

def _render_goals(objectives: list[str] | None) -> str:
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

EXPLAIN_FALLBACK = "LET'S COME BACK TO THAT ONE TOGETHER — I COULDN'T PIN IT DOWN EITHER."

async def explain_answer(
    question: TargetedQuestion,
    topic: str = "",
    transcript: str = "",
    objectives: list[str] | None = None,
) -> str:
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
