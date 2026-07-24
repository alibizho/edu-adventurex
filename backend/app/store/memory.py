"""In-memory Store impl — module-level dicts keyed by session_id. The dev default
(STORE_BACKEND=memory): fast, no extra deps, but state is lost on restart and it's not
concurrency-safe. The Postgres impl in db.py is the durable alternative. Both satisfy the Store
protocol in base.py, so swapping is config-only.

The dicts are module-level (process-wide) so the MemoryStore singleton shares them, matching the
behaviour routes relied on before the Store abstraction.
"""
import time

from ..schemas import ChunkAnalysis, QAEntry, RunResult, Score, Segment, TargetedQuestion

# Process-wide state, keyed by session_id.
_transcripts: dict[str, list[Segment]] = {}
_runs: dict[str, RunResult] = {}
_scores: dict[str, list[Score]] = {}
_chunk_analyses: dict[str, list[ChunkAnalysis]] = {}
_qa_ledger: dict[str, list[QAEntry]] = {}
_topics: dict[str, str] = {}


class MemoryStore:
    """Store impl backed by module-level dicts. Methods are async to match the Store protocol but
    do no I/O."""

    async def init(self) -> None:
        pass

    async def dispose(self) -> None:
        pass

    # ---- transcript ----

    async def get_transcript(self, session_id: str) -> list[Segment]:
        return _transcripts.setdefault(session_id, [])

    async def append_segment(self, session_id: str, segment: Segment) -> None:
        _transcripts.setdefault(session_id, []).append(segment)

    # ---- confusion analyses ----

    async def get_analyses(self, session_id: str) -> list[ChunkAnalysis]:
        return _chunk_analyses.setdefault(session_id, [])

    async def append_analysis(self, session_id: str, analysis: ChunkAnalysis) -> None:
        """Upsert by chunk_id: replace an existing entry with the same chunk_id, else append."""
        lst = _chunk_analyses.setdefault(session_id, [])
        for i, a in enumerate(lst):
            if a.chunk_id == analysis.chunk_id:
                lst[i] = analysis
                return
        lst.append(analysis)

    async def set_analyses(self, session_id: str, analyses: list[ChunkAnalysis]) -> None:
        _chunk_analyses[session_id] = analyses

    # ---- measurement run + raw scores ----

    async def set_run(self, session_id: str, result: RunResult, scores: list[Score]) -> None:
        _runs[session_id] = result
        _scores[session_id] = scores

    async def get_run(self, session_id: str) -> RunResult | None:
        return _runs.get(session_id)

    async def get_scores(self, session_id: str) -> list[Score]:
        return _scores.get(session_id, [])

    # ---- targeted-question ledger ----

    async def get_history(self, session_id: str) -> list[QAEntry]:
        return _qa_ledger.setdefault(session_id, [])

    async def next_question_id(self, session_id: str) -> int:
        """Monotonic id for the next targeted question in this session."""
        ledger = _qa_ledger.setdefault(session_id, [])
        return (max((e.question.id for e in ledger), default=-1)) + 1

    async def record_questions(self, session_id: str, questions: list[TargetedQuestion]) -> None:
        """Append newly-asked questions to the ledger (unanswered)."""
        ledger = _qa_ledger.setdefault(session_id, [])
        ledger.extend(QAEntry(question=q) for q in questions)

    async def record_answer(self, session_id: str, question_id: int, answer: str) -> bool:
        """Attach an answer to a previously-asked question. Returns False if the id is unknown."""
        for entry in _qa_ledger.setdefault(session_id, []):
            if entry.question.id == question_id:
                entry.answer = answer
                entry.answered_at = time.time()
                return True
        return False

    async def covered_chunk_ids(self, session_id: str) -> set[int]:
        """Chunks that already have an ANSWERED question — excluded from further selection."""
        return {
            e.question.chunk_id
            for e in _qa_ledger.setdefault(session_id, [])
            if e.answer is not None
        }

    # ---- session topic ----

    async def set_topic(self, session_id: str, topic: str) -> None:
        _topics[session_id] = topic

    async def get_topic(self, session_id: str) -> str:
        return _topics.get(session_id, "")
