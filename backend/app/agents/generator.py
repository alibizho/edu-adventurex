"""Transfer-question generation from a transcript (report §4.2), each with a ground-truth key."""
import re

from ..llm import generator_chat
from ..schemas import Question, QuestionKind, Segment
from .parsing import extract_json_array
from .prompts import GENERATOR_SYSTEM


def _parse_numbered_list(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if m:
            items.append(m.group(1).strip())
    return items


async def generate_questions(transcript: list[Segment], source: str = "") -> list[Question]:
    """Return 15-20 candidate transfer questions, each with an answer key grounded in `source`
    (ground truth). Filtering happens in pipeline/filter.py."""
    lesson = "\n".join(f"[{s.id}] {s.text}" for s in transcript)
    user = f"LESSON:\n{lesson}"
    if source.strip():
        user += f"\n\nSOURCE (ground truth):\n{source}"
    raw = await generator_chat(GENERATOR_SYSTEM, user, temperature=0.5)

    items = extract_json_array(raw)
    questions: list[Question] = []
    for i, it in enumerate(items):
        if isinstance(it, dict) and str(it.get("text", "")).strip():
            key = str(it.get("answer_key", "")).strip() or None
            questions.append(
                Question(id=i, text=str(it["text"]).strip(), answer_key=key, kind=QuestionKind.transfer)
            )
    if questions:
        return questions

    # Fallback: numbered list without keys (verifier will mark these incorrect — visible downstream).
    return [
        Question(id=i, text=t, kind=QuestionKind.transfer)
        for i, t in enumerate(_parse_numbered_list(raw))
    ]
