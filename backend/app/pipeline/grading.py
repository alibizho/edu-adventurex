from typing import Protocol

from ..agents.prompts import VERIFIER_SYSTEM
from ..llm import verifier_chat

NOT_COVERED = "NOT COVERED"

class Gradable(Protocol):
    text: str
    answer_key: str | None

def is_not_covered(answer: str) -> bool:
    return NOT_COVERED.lower() in answer.strip().lower()[:40]

async def grade_answer(persona_answer: str, question: Gradable) -> tuple[bool, bool]:
    if is_not_covered(persona_answer):
        return False, True

    key = (question.answer_key or "").strip()
    if not key:
        return False, False

    user = (
        f"QUESTION:\n{question.text}\n\n"
        f"ANSWER KEY:\n{key}\n\n"
        f"TEST-TAKER ANSWER:\n{persona_answer}"
    )
    verdict = await verifier_chat(VERIFIER_SYSTEM, user, temperature=0.0)
    return verdict.strip().upper().startswith("CORRECT"), False
