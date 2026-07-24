"""HTTP surface. Thin — orchestration lives in agents/ and pipeline/. Open /docs to poke it."""
from fastapi import APIRouter, HTTPException

from ..agents.generator import generate_questions
from ..agents.student import student_turn
from ..agents.targeted import generate_targeted_questions
from ..confusion import engine
from ..pipeline.filter import filter_questions
from ..pipeline.scoring import score_ensemble
from ..schemas import (
    AnswerRequest,
    IngestRequest,
    NextQuestionsRequest,
    QAEntry,
    RunResult,
    Segment,
    TargetedQuestion,
    TeachTurnRequest,
    TeachTurnResponse,
)
from ..store import memory

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"ok": True}


@router.post("/teach/turn", response_model=TeachTurnResponse)
async def teach_turn(req: TeachTurnRequest) -> TeachTurnResponse:
    """One turn of the teaching loop: kid speaks, student replies in character."""
    resp = await student_turn(req.transcript, req.latest_utterance)
    memory.append_segment(req.session_id, resp.new_segment)
    return resp


@router.post("/measure", response_model=RunResult)
async def measure(session_id: str) -> RunResult:
    """Run the full transfer-delta measurement on a stored transcript.

    generate (with answer keys) -> filter (cold student) -> ensemble score -> delta.
    Correctness comes from the verifier (pipeline/grading.py), which grades each answer against
    the question's ground-truth key. No source is attached to stored sessions yet, so keys fall
    back to the generator's own topic knowledge.
    """
    transcript: list[Segment] = memory.get_transcript(session_id)
    questions = await generate_questions(transcript)

    survivors, survival_rate = await filter_questions(questions)
    result, scores = await score_ensemble(transcript, survivors, survival_rate, session_id)
    memory.runs[session_id] = result
    memory.scores[session_id] = scores
    return result


# ---- confusion-driven targeted questioning ----

@router.post("/confusion/ingest")
async def confusion_ingest(req: IngestRequest) -> dict:
    """Store per-chunk analyses from the confusion engine (the real ML posts here)."""
    memory.set_analyses(req.session_id, req.chunks)
    return {"session_id": req.session_id, "n_chunks": len(req.chunks)}


@router.post("/confusion/mock")
async def confusion_mock(session_id: str) -> dict:
    """Demo path while the ML is absent: run the heuristic mock over the stored transcript's
    segment texts and store the resulting analyses."""
    transcript = memory.get_transcript(session_id)
    if not transcript:
        raise HTTPException(404, f"no transcript for session {session_id!r}")
    analyses = engine.analyze([s.text for s in transcript])
    memory.set_analyses(session_id, analyses)
    return {"session_id": session_id, "n_chunks": len(analyses)}


@router.post("/questions/next", response_model=list[TargetedQuestion])
async def questions_next(req: NextQuestionsRequest) -> list[TargetedQuestion]:
    """Pick the lowest-confidence uncovered chunks and generate specialized, non-repeating
    questions for them, keyed off the session's Q&A memory."""
    analyses = memory.get_analyses(req.session_id)
    if not analyses:
        raise HTTPException(404, f"no analyses for session {req.session_id!r}; ingest or mock first")

    chunks = engine.select_low_confidence(
        analyses, k=req.n, exclude_ids=memory.covered_chunk_ids(req.session_id)
    )
    history = memory.get_history(req.session_id)
    questions = await generate_targeted_questions(
        chunks, history, start_id=memory.next_question_id(req.session_id)
    )
    memory.record_questions(req.session_id, questions)
    return questions


@router.post("/questions/answer")
async def questions_answer(req: AnswerRequest) -> dict:
    """Record the user's answer so the agent won't re-ask it."""
    ok = memory.record_answer(req.session_id, req.question_id, req.answer)
    if not ok:
        raise HTTPException(404, f"unknown question_id {req.question_id} for session {req.session_id!r}")
    return {"session_id": req.session_id, "question_id": req.question_id, "recorded": True}


@router.get("/questions/history/{session_id}", response_model=list[QAEntry])
async def questions_history(session_id: str) -> list[QAEntry]:
    return memory.get_history(session_id)
