"""Correctness check. The verifier NEVER scores the child — it compares a persona's answer to
the question's ANSWER KEY (ground truth). Style, fluency and length are ignored; only whether
the answer conveys the key's essential claim matters.

`NOT COVERED` is graded incorrect but reported separately (a gap, not a wrong answer).
"""
from ..agents.prompts import VERIFIER_SYSTEM
from ..llm import verifier_chat
from ..schemas import Question

NOT_COVERED = "NOT COVERED"


def is_not_covered(answer: str) -> bool:
    return NOT_COVERED.lower() in answer.strip().lower()[:40]


async def grade_answer(persona_answer: str, question: Question) -> tuple[bool, bool]:
    """Return (correct, not_covered). Verifier compares the answer to `question.answer_key` only."""
    if is_not_covered(persona_answer):
        return False, True

    key = (question.answer_key or "").strip()
    if not key:
        # No key to grade against — cannot credit. Surfaces as a data problem in the harness.
        return False, False

    user = (
        f"QUESTION:\n{question.text}\n\n"
        f"ANSWER KEY:\n{key}\n\n"
        f"TEST-TAKER ANSWER:\n{persona_answer}"
    )
    verdict = await verifier_chat(VERIFIER_SYSTEM, user, temperature=0.0)
    return verdict.strip().upper().startswith("CORRECT"), False
