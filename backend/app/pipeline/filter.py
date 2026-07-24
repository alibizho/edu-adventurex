"""Cold-student filter (report §4.3) — the load-bearing step.

Every candidate question is put to a cold student (no transcript), `settings.filter_cold_samples`
samples. Any question the cold student answers correctly in >= 2/3 samples is measuring priors,
not teaching, and is dropped. If fewer than 25% survive, regenerate harder.
"""
import asyncio

from ..agents.prompts import COLD_SYSTEM
from ..config import settings
from ..llm import student_chat
from ..schemas import Question
from .grading import grade_answer

PASS_THRESHOLD = 2 / 3          # drop if cold pass-rate >= this


async def _cold_pass_rate(question: Question, sem: asyncio.Semaphore) -> float:
    k = settings.filter_cold_samples

    async def one_sample() -> bool:
        async with sem:
            ans = await student_chat(COLD_SYSTEM, question.text, temperature=0.8)
        async with sem:
            correct, _ = await grade_answer(ans, question)
        return correct

    results = await asyncio.gather(*(one_sample() for _ in range(k)))
    return sum(results) / k


async def filter_questions(questions: list[Question]) -> tuple[list[Question], float]:
    """Return (surviving questions, survival_rate). Grades cold answers against each question's key."""
    sem = asyncio.Semaphore(settings.max_concurrency)
    rates = await asyncio.gather(*(_cold_pass_rate(q, sem) for q in questions))
    survivors: list[Question] = []
    for q, rate in zip(questions, rates):
        q.cold_pass_rate = rate
        q.survived = rate < PASS_THRESHOLD
        if q.survived:
            survivors.append(q)

    survival_rate = len(survivors) / len(questions) if questions else 0.0
    return survivors, survival_rate
