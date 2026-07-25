"""Context store interface. Two implementations satisfy it: in-memory dicts (memory.py, the dev
default — fast, no deps, lost on restart) and Postgres (db.py — durable). Routes depend only on
this interface, so the backend is selected by config (STORE_BACKEND) with no route changes."""
from __future__ import annotations

from typing import Protocol

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


def count_thread_answers(ledger: list[QAEntry], question_id: int) -> int:
    """How many answers a conversation thread has taken, following parent_id links.

    Shared by both stores so the two can't disagree about what counts as a turn. Walks up to the
    root question and then back down every follow-up, because the cap must apply to the whole
    back-and-forth about one concept — not to each rephrasing of it.
    """
    by_id = {e.question.id: e for e in ledger}
    root, seen = question_id, set()
    while root in by_id and by_id[root].question.parent_id is not None and root not in seen:
        seen.add(root)
        root = by_id[root].question.parent_id

    thread = {root}
    changed = True
    while changed:
        changed = False
        for entry in ledger:
            parent = entry.question.parent_id
            if parent in thread and entry.question.id not in thread:
                thread.add(entry.question.id)
                changed = True
    return sum(1 for e in ledger if e.question.id in thread and e.answer is not None)


class Store(Protocol):
    # lifecycle (no-ops for the memory store)
    async def init(self) -> None: ...
    async def dispose(self) -> None: ...

    # transcript
    async def get_transcript(self, session_id: str) -> list[Segment]: ...
    async def append_segment(self, session_id: str, segment: Segment) -> None: ...

    # confusion analyses (per-chunk)
    async def get_analyses(self, session_id: str) -> list[ChunkAnalysis]: ...
    async def append_analysis(self, session_id: str, analysis: ChunkAnalysis) -> None: ...
    async def set_analyses(self, session_id: str, analyses: list[ChunkAnalysis]) -> None: ...

    # measurement run + raw scores
    async def set_run(self, session_id: str, result: RunResult, scores: list[Score]) -> None: ...
    async def get_run(self, session_id: str) -> RunResult | None: ...
    async def get_scores(self, session_id: str) -> list[Score]: ...

    # targeted-question ledger
    async def get_history(self, session_id: str) -> list[QAEntry]: ...
    async def next_question_id(self, session_id: str) -> int: ...
    async def record_questions(self, session_id: str, questions: list[TargetedQuestion]) -> None: ...
    async def record_answer(self, session_id: str, question_id: int, answer: str) -> bool: ...
    async def find_question(self, session_id: str, question_id: int) -> QAEntry | None: ...
    async def thread_turns(self, session_id: str, question_id: int) -> int: ...
    async def covered_chunk_ids(self, session_id: str) -> set[int]: ...

    # session topic
    async def set_topic(self, session_id: str, topic: str) -> None: ...
    async def get_topic(self, session_id: str) -> str: ...

    # learning plan (growth path) — keyed by path_id
    async def save_path(self, path: GrowthPath) -> None: ...
    async def get_path(self, path_id: str) -> GrowthPath | None: ...
    async def list_paths(self) -> list[GrowthPath]: ...
    async def save_class(self, path_id: str, cls: ClassUnit) -> None: ...

    # cross-class memory — keyed by path_id
    async def get_memory(self, path_id: str) -> PathMemory: ...
    async def update_memory(self, path_id: str, memory: PathMemory) -> None: ...

    # asynchronous transfer analysis job state
    async def get_analysis_job(self, session_id: str) -> AnalysisJob | None: ...
    async def set_analysis_job(self, job: AnalysisJob) -> None: ...
