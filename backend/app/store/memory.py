"""Dead-simple in-memory store so endpoints work end-to-end today. Swap for Postgres later
(report §7 has the schema). Not concurrency-safe; fine for the hackathon."""
import time

from ..schemas import ChunkAnalysis, QAEntry, RunResult, Score, Segment, TargetedQuestion

# session_id -> transcript
transcripts: dict[str, list[Segment]] = {}
# session_id -> latest run
runs: dict[str, RunResult] = {}
# session_id -> raw scores
scores: dict[str, list[Score]] = {}

# session_id -> latest confusion-engine analyses (Instrument C output contract)
chunk_analyses: dict[str, list[ChunkAnalysis]] = {}
# session_id -> targeted-question ledger (asked questions + the user's answers)
qa_ledger: dict[str, list[QAEntry]] = {}


def get_transcript(session_id: str) -> list[Segment]:
    return transcripts.setdefault(session_id, [])


def append_segment(session_id: str, segment: Segment) -> None:
    transcripts.setdefault(session_id, []).append(segment)


# ---- confusion analyses + targeted-question memory ----

def set_analyses(session_id: str, analyses: list[ChunkAnalysis]) -> None:
    chunk_analyses[session_id] = analyses


def append_analysis(session_id: str, analysis: ChunkAnalysis) -> None:
    """Append one confusion analysis, replacing any existing entry with the same chunk_id (so a
    re-analyzed utterance updates in place). Used by the streaming audio path."""
    lst = chunk_analyses.setdefault(session_id, [])
    for i, a in enumerate(lst):
        if a.chunk_id == analysis.chunk_id:
            lst[i] = analysis
            return
    lst.append(analysis)


def get_analyses(session_id: str) -> list[ChunkAnalysis]:
    return chunk_analyses.setdefault(session_id, [])


def get_history(session_id: str) -> list[QAEntry]:
    return qa_ledger.setdefault(session_id, [])


def next_question_id(session_id: str) -> int:
    """Monotonic id for the next targeted question in this session."""
    ledger = qa_ledger.setdefault(session_id, [])
    return (max((e.question.id for e in ledger), default=-1)) + 1


def record_questions(session_id: str, questions: list[TargetedQuestion]) -> None:
    """Append newly-asked questions to the ledger (unanswered)."""
    ledger = qa_ledger.setdefault(session_id, [])
    ledger.extend(QAEntry(question=q) for q in questions)


def record_answer(session_id: str, question_id: int, answer: str) -> bool:
    """Attach an answer to a previously-asked question. Returns False if the id is unknown."""
    for entry in qa_ledger.setdefault(session_id, []):
        if entry.question.id == question_id:
            entry.answer = answer
            entry.answered_at = time.time()
            return True
    return False


def covered_chunk_ids(session_id: str) -> set[int]:
    """Chunks that already have an ANSWERED question — excluded from further selection so the
    agent moves on rather than re-drilling a resolved spot."""
    return {
        e.question.chunk_id
        for e in qa_ledger.setdefault(session_id, [])
        if e.answer is not None
    }
