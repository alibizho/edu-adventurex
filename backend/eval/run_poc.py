"""POC harness — does the taught-vs-cold transfer delta separate good teaching from bad?

Runs Instrument A end-to-end on a GOOD and a BAD transcript of the same topic, in two modes:
  - FIXED:     a hand-curated gold question set (clean, repeatable delta signal), filter skipped.
  - GENERATED: generate -> filter -> score (full pipeline, question-generation variance included).

Success = delta(GOOD) clearly above delta(BAD), and BAD surfaces at least one negative-delta
question (teaching made the learner worse).

Run from backend/:  PYTHONPATH=. .venv/bin/python -m eval.run_poc
"""
import asyncio
import random

from app.agents.generator import generate_questions
from app.agents.personas import cold_personas, taught_personas, taught_system
from app.config import settings
from app.llm import student_chat
from app.pipeline.filter import filter_questions
from app.pipeline.grading import grade_answer
from app.pipeline.scoring import score_ensemble
from app.schemas import Question, RunResult, Segment

from eval.fixtures import BAD_TRANSCRIPT, GOLD_QUESTIONS, GOOD_TRANSCRIPT, SOURCE, TOPIC

NA = -1.0  # survival_rate sentinel for the fixed set (filter skipped)


def _segments(lines: list[str]) -> list[Segment]:
    return [Segment(id=i, idx=i, text=t) for i, t in enumerate(lines)]


def _gold_questions() -> list[Question]:
    return [Question(id=i, text=q["text"], answer_key=q["answer_key"]) for i, q in enumerate(GOLD_QUESTIONS)]


def _bootstrap_ci(deltas: list[float], B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over questions (the near-independent unit; personas are correlated)."""
    if not deltas:
        return 0.0, 0.0
    rnd = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return means[int(0.025 * B)], means[int(0.975 * B)]


async def _run(label: str, transcript_lines: list[str], mode: str) -> RunResult:
    transcript = _segments(transcript_lines)
    if mode == "generated":
        questions = await generate_questions(transcript, SOURCE)
        missing = sum(1 for q in questions if not q.answer_key)
        if missing:
            print(f"  [warn] {label}/generated: {missing}/{len(questions)} generated questions have no answer key")
        survivors, survival_rate = await filter_questions(questions)
    else:  # fixed
        survivors, survival_rate = _gold_questions(), NA
    result, _scores = await score_ensemble(transcript, survivors, survival_rate, f"{label}-{mode}")
    return result


def _print_mode_table(mode: str, results: dict) -> None:
    print(f"\n=== {mode.upper()} question set ===")
    print(f"{'transcript':<12}{'delta':>9}{'95% CI':>20}{'neg-Δ':>9}{'survival':>10}")
    for label in ("GOOD", "BAD"):
        r = results[(label, mode)]
        deltas = [d.delta for d in r.per_question]
        lo, hi = _bootstrap_ci(deltas)
        neg = sum(1 for d in deltas if d < 0)
        sr = " n/a" if r.survival_rate < 0 else f"{r.survival_rate:.2f}"
        print(f"{label:<12}{r.delta_overall:>+9.3f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}{f'{neg}/{len(deltas)}':>9}{sr:>10}")


def _print_perquestion(results: dict) -> None:
    print("\n--- per-question deltas (FIXED set) ---")
    for label in ("GOOD", "BAD"):
        r = results[(label, "fixed")]
        print(f"\n[{label}]")
        for d in r.per_question:
            flag = "  <-- negative" if d.delta < 0 else ""
            print(f"  q{d.question_id}: taught={d.taught_mean:.2f} cold={d.cold_mean:.2f} delta={d.delta:+.2f}{flag}")


async def _debug_samples(n: int = 2) -> None:
    """Eyeball a few grader verdicts on the good transcript — confirms CORRECT/INCORRECT is sane."""
    transcript = _segments(GOOD_TRANSCRIPT)
    print("\n--- grader sanity check (good transcript, first taught persona) ---")
    for q in _gold_questions()[:n]:
        ans = await student_chat(taught_system(taught_personas()[0], transcript), q.text, temperature=0.7)
        correct, nc = await grade_answer(ans, q)
        print(f"\nQ: {q.text}")
        print(f"key:    {q.answer_key}")
        print(f"answer: {ans.strip()[:200]!r}")
        print(f"verdict: correct={correct} not_covered={nc}")


async def main() -> None:
    verifier = settings.verifier_model or settings.generator_model
    print(f"TOPIC: {TOPIC}")
    print(f"personas: {len(taught_personas())} taught / {len(cold_personas())} cold  |  "
          f"student={settings.student_model}  verifier={verifier}")

    results: dict = {}
    for label, lines in (("GOOD", GOOD_TRANSCRIPT), ("BAD", BAD_TRANSCRIPT)):
        for mode in ("fixed", "generated"):
            results[(label, mode)] = await _run(label, lines, mode)

    for mode in ("fixed", "generated"):
        _print_mode_table(mode, results)
    _print_perquestion(results)
    await _debug_samples()

    # Go / no-go on the fixed set (the clean signal).
    good = results[("GOOD", "fixed")].delta_overall
    bad = results[("BAD", "fixed")].delta_overall
    bad_neg = sum(1 for d in results[("BAD", "fixed")].per_question if d.delta < 0)
    print("\n=== GO / NO-GO (fixed set) ===")
    print(f"delta(GOOD)={good:+.3f}  delta(BAD)={bad:+.3f}  separation={good - bad:+.3f}  BAD negative-Δ Qs={bad_neg}")
    passed = good > bad and bad_neg >= 1
    print("RESULT:", "PASS — the delta separates good from bad teaching, and bad shows negative delta."
          if passed else "CHECK — weak separation; tune the verifier prompt or the gold keys, not the plumbing.")


if __name__ == "__main__":
    asyncio.run(main())
