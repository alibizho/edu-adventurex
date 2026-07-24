"""POC harness for the targeted-question agent — does it (1) target the lowest-confidence chunks,
(2) tailor questions to the anomaly, and (3) never repeat itself across rounds?

Round 1: request questions on the low-confidence chunks.
Answer some of them (leaving one chunk still open).
Round 2: request again — the answered chunks are skipped, and the still-open chunk gets a NEW,
non-repeating question.

Run from backend/:  PYTHONPATH=. .venv/bin/python -m eval.run_targeted
"""
import asyncio

from app.agents.targeted import generate_targeted_questions
from app.confusion.engine import select_low_confidence
from app.store import memory

from eval.confusion_fixtures import DEMO_CHUNKS

SESSION = "demo"
THRESHOLD = 0.5   # only genuinely low-confidence chunks are eligible
K = 3
ANSWER_CHUNKS = {2, 4}   # answer these in round 1; leave the rest open for round 2


async def _round(label: str):
    analyses = memory.get_analyses(SESSION)
    covered = memory.covered_chunk_ids(SESSION)
    chunks = select_low_confidence(analyses, k=K, threshold=THRESHOLD, exclude_ids=covered)
    print(f"\n=== {label} ===")
    print(f"covered chunks (skipped): {sorted(covered) or '—'}")
    print(f"selected low-confidence chunks: {[(c.chunk_id, c.confidence) for c in chunks]}")

    questions = await generate_targeted_questions(
        chunks, memory.get_history(SESSION), start_id=memory.next_question_id(SESSION)
    )
    memory.record_questions(SESSION, questions)
    for q in questions:
        print(f"  q{q.id} [chunk {q.chunk_id} / {q.anomaly_type}]: {q.text}")
    return questions


async def main():
    memory.set_analyses(SESSION, DEMO_CHUNKS)
    print(f"session={SESSION}  chunks={len(DEMO_CHUNKS)}  threshold={THRESHOLD}")

    r1 = await _round("ROUND 1")

    # Answer the questions that target the two most-confused chunks; leave the rest open.
    print("\n--- user answers round-1 questions on chunks", sorted(ANSWER_CHUNKS), "---")
    for q in r1:
        if q.chunk_id in ANSWER_CHUNKS:
            memory.record_answer(SESSION, q.id, "(user's answer here)")
            print(f"  answered q{q.id} (chunk {q.chunk_id})")

    r2 = await _round("ROUND 2")

    # Checks.
    r1_texts = {q.text.strip().lower() for q in r1}
    repeats = [q for q in r2 if q.text.strip().lower() in r1_texts]
    r2_chunks = {q.chunk_id for q in r2}
    leaked = r2_chunks & ANSWER_CHUNKS

    print("\n=== CHECKS ===")
    print(f"round-2 repeats of round-1 questions: {len(repeats)} (want 0)")
    print(f"round-2 touched an already-answered chunk: {sorted(leaked) or 'none'} (want none)")
    still_open = {c.chunk_id for c in DEMO_CHUNKS if c.confidence < THRESHOLD} - ANSWER_CHUNKS
    print(f"still-open low-confidence chunks re-asked in round 2: "
          f"{sorted(still_open & r2_chunks) or 'none'} (want {sorted(still_open)})")
    ok = not repeats and not leaked
    print("RESULT:", "PASS — targets the weak chunks, remembers answers, never repeats."
          if ok else "CHECK — a repeat or a covered-chunk leak slipped through.")

    print("\n--- ledger ---")
    for e in memory.get_history(SESSION):
        ans = e.answer or "(unanswered)"
        print(f"  q{e.question.id} chunk {e.question.chunk_id}: {e.question.text}  -> {ans}")


if __name__ == "__main__":
    asyncio.run(main())
