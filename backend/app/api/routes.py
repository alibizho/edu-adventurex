"""HTTP surface. Thin — orchestration lives in agents/ and pipeline/. Open /docs to poke it."""
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..agents.generator import generate_questions
from ..agents.student import student_turn
from ..agents.targeted import generate_targeted_questions
from ..confusion import client, engine
from ..pipeline.filter import filter_questions
from ..pipeline.scoring import score_ensemble
from ..schemas import (
    AnswerRequest,
    ChunkAnalysis,
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

@router.get("/confusion/health")
async def confusion_health() -> dict:
    """Report whether the ml-service confusion engine is reachable (and its own health)."""
    return await client.health()


@router.post("/confusion/analyze", response_model=ChunkAnalysis)
async def confusion_analyze(
    session_id: str = Form(...),
    chunk_id: int = Form(0),
    history: str = Form("[]"),          # JSON array of prior transcripts (Space B context)
    enable_space_c: bool | None = Form(None),
    audio: UploadFile = File(...),
) -> ChunkAnalysis:
    """Forward a recorded utterance to the ml-service confusion engine and store the analysis.
    If `history` is empty, the session's prior segment texts are used as Space B context.
    Degrades to a neutral analysis if the ml-service is unreachable (see confusion/client.py)."""
    try:
        hist = json.loads(history) if history else []
        hist = [str(h) for h in hist] if isinstance(hist, list) else []
    except json.JSONDecodeError:
        hist = []
    if not hist:
        hist = [s.text for s in memory.get_transcript(session_id)]

    audio_bytes = await audio.read()
    analysis = await client.analyze_audio(
        audio_bytes, filename=audio.filename or "chunk.wav", chunk_id=chunk_id,
        history=hist, enable_space_c=enable_space_c,
    )
    memory.append_analysis(session_id, analysis)
    return analysis


@router.post("/confusion/ingest")
async def confusion_ingest(req: IngestRequest) -> dict:
    """Store per-chunk analyses from the confusion engine (an external ML can also post here)."""
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
