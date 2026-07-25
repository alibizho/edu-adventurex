"""In-memory Store impl — module-level dicts keyed by session_id. The dev default
(STORE_BACKEND=memory): fast, no extra deps, but state is lost on restart and it's not
concurrency-safe. The Postgres impl in db.py is the durable alternative. Both satisfy the Store
protocol in base.py, so swapping is config-only.

The dicts are module-level (process-wide) so the MemoryStore singleton shares them, matching the
behaviour routes relied on before the Store abstraction.
"""
import time

from ..schemas import (
    AnalysisJob,
    ChunkAnalysis,
    ClassUnit,
    GrowthPath,
    PathMemory,
    QAEntry,
    RunResult,
    Score,
    Segment,
    TargetedQuestion,
)
from .base import count_thread_answers

# Process-wide state, keyed by session_id.
_transcripts: dict[str, list[Segment]] = {}
_runs: dict[str, RunResult] = {}
_scores: dict[str, list[Score]] = {}
_chunk_analyses: dict[str, list[ChunkAnalysis]] = {}
_qa_ledger: dict[str, list[QAEntry]] = {}
_topics: dict[str, str] = {}
# Learning plan + cross-class memory, keyed by path_id.
_paths: dict[str, GrowthPath] = {}
_path_memory: dict[str, PathMemory] = {}
_analysis_jobs: dict[str, AnalysisJob] = {}


class MemoryStore:
    """Store impl backed by module-level dicts. Methods are async to match the Store protocol but
    do no I/O."""

    async def init(self) -> None:
        pass

    async def dispose(self) -> None:
        pass

    async def clear_session(self, session_id: str) -> None:
        for bucket in (_transcripts, _runs, _scores, _chunk_analyses, _qa_ledger, _topics,
                       _analysis_jobs):
            bucket.pop(session_id, None)

    # ---- transcript ----

    async def get_transcript(self, session_id: str) -> list[Segment]:
        # Copy, don't alias. The db store builds a fresh list per call, so a caller that holds the
        # result across an append_segment sees the new segment here and NOT there — the same code
        # then behaves differently depending on STORE_BACKEND. A shallow copy makes the two agree.
        return list(_transcripts.setdefault(session_id, []))

    async def append_segment(self, session_id: str, segment: Segment) -> None:
        _transcripts.setdefault(session_id, []).append(segment)

    # ---- confusion analyses ----

    async def get_analyses(self, session_id: str) -> list[ChunkAnalysis]:
        return list(_chunk_analyses.setdefault(session_id, []))    # copy: see get_transcript

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
        # Copy the list, not the entries: record_answer mutates entries through _qa_ledger, so
        # sharing the QAEntry objects is what keeps an answer visible to an already-fetched history.
        return list(_qa_ledger.setdefault(session_id, []))

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

    async def find_question(self, session_id: str, question_id: int) -> QAEntry | None:
        for entry in _qa_ledger.setdefault(session_id, []):
            if entry.question.id == question_id:
                return entry
        return None

    async def thread_turns(self, session_id: str, question_id: int) -> int:
        return count_thread_answers(_qa_ledger.setdefault(session_id, []), question_id)

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

    # ---- learning plan (growth path) ----

    async def save_path(self, path: GrowthPath) -> None:
        _paths[path.path_id] = path

    async def get_path(self, path_id: str) -> GrowthPath | None:
        return _paths.get(path_id)

    async def list_paths(self) -> list[GrowthPath]:
        return list(reversed(_paths.values()))

    async def save_class(self, path_id: str, cls: ClassUnit) -> None:
        """Replace a class in the stored path (used to persist generated notes)."""
        path = _paths.get(path_id)
        if path is None:
            return
        for i, c in enumerate(path.classes):
            if c.class_id == cls.class_id:
                path.classes[i] = cls
                return

    # ---- cross-class memory ----

    async def get_memory(self, path_id: str) -> PathMemory:
        return _path_memory.setdefault(path_id, PathMemory(path_id=path_id))

    async def update_memory(self, path_id: str, memory: PathMemory) -> None:
        _path_memory[path_id] = memory

    async def get_analysis_job(self, session_id: str) -> AnalysisJob | None:
        return _analysis_jobs.get(session_id)

    async def set_analysis_job(self, job: AnalysisJob) -> None:
        _analysis_jobs[job.session_id] = job
