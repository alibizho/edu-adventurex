"""Ensemble scoring + delta math (report §4.4–4.5).

Taught arm (transcript in context) and cold arm (no transcript, control) both answer the
surviving questions closed-book. Delta = mean(taught) - mean(cold), per question and overall.
Negative delta == teaching made the learner worse. Correctness comes from the verifier
(grade_answer), which checks each answer against the question's ground-truth key.
"""
import asyncio

from ..agents.personas import cold_personas, taught_personas, taught_system
from ..agents.prompts import COLD_SYSTEM
from ..config import settings
from ..llm import student_chat
from ..schemas import Arm, Question, QuestionDelta, RunResult, Score, Segment
from .attribution import parse_citations
from .grading import grade_answer


async def _answer(
    system: str, question: Question, persona: str, arm: Arm, sem: asyncio.Semaphore,
) -> Score:
    async with sem:
        ans = await student_chat(system, question.text, temperature=0.7)
    async with sem:
        correct, not_covered = await grade_answer(ans, question)
    return Score(
        question_id=question.id,
        arm=arm,
        persona_seed=persona,
        correct=correct,
        not_covered=not_covered,
        cited_segment_ids=parse_citations(ans),
    )


async def score_ensemble(
    transcript: list[Segment],
    questions: list[Question],
    survival_rate: float,
    session_id: str,
) -> tuple[RunResult, list[Score]]:
    sem = asyncio.Semaphore(settings.max_concurrency)
    tasks: list = []

    for q in questions:
        for persona in taught_personas():
            tasks.append(_answer(taught_system(persona, transcript), q, persona, Arm.taught, sem))
        for persona in cold_personas():
            tasks.append(_answer(COLD_SYSTEM, q, persona, Arm.cold, sem))

    scores: list[Score] = await asyncio.gather(*tasks)

    per_question = _deltas(questions, scores)
    delta_overall = sum(d.delta for d in per_question) / len(per_question) if per_question else 0.0
    result = RunResult(
        session_id=session_id,
        delta_overall=delta_overall,
        survival_rate=survival_rate,
        per_question=per_question,
    )
    return result, scores


def _deltas(questions: list[Question], scores: list[Score]) -> list[QuestionDelta]:
    out: list[QuestionDelta] = []
    for q in questions:
        taught = [s.correct for s in scores if s.question_id == q.id and s.arm == Arm.taught]
        cold = [s.correct for s in scores if s.question_id == q.id and s.arm == Arm.cold]
        t = sum(taught) / len(taught) if taught else 0.0
        z = sum(cold) / len(cold) if cold else 0.0
        out.append(QuestionDelta(question_id=q.id, taught_mean=t, cold_mean=z, delta=t - z))
    return out
