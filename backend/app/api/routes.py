import time

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from .forms import json_string_list
from ..agents.generator import generate_questions
from ..agents.student import student_turn
from ..agents.targeted import generate_targeted_questions
from ..confusion import client, engine
from ..config import settings
from ..fusion import fuse
from ..pipeline.filter import filter_questions
from ..pipeline.scoring import score_ensemble
from ..schemas import (
    AnalysisJob,
    AnalysisStatusResponse,
    AnswerRequest,
    ChunkAnalysis,
    ChunkQuestionResponse,
    FusionResult,
    IngestRequest,
    NextQuestionsRequest,
    QAEntry,
    RunResult,
    Segment,
    SessionSnapshot,
    TargetedQuestion,
    TeachTurnRequest,
    TeachTurnResponse,
)
from ..store import store

router = APIRouter()

@router.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "llm_configured": bool(settings.student_api_key and settings.generator_api_key),
        "store_backend": settings.store_backend,
    }

@router.post("/teach/turn", response_model=TeachTurnResponse)
async def teach_turn(req: TeachTurnRequest) -> TeachTurnResponse:
    resp = await student_turn(req.transcript, req.latest_utterance)
    await store.append_segment(req.session_id, resp.new_segment)
    return resp

async def _measure_session(session_id: str) -> RunResult:
    transcript: list[Segment] = await store.get_transcript(session_id)
    questions = await generate_questions(transcript)

    survivors, survival_rate = await filter_questions(questions)
    result, scores = await score_ensemble(transcript, survivors, survival_rate, session_id)
    await store.set_run(session_id, result, scores)
    return result

async def _set_class_analysis_status(
    session_id: str, status: str, error: str | None = None
) -> None:
    if ":" not in session_id:
        return
    path_id, class_id = session_id.split(":", 1)
    try:
        memory = await store.get_memory(path_id)
    except KeyError:
        return
    progress = memory.class_progress.get(class_id)
    if progress is None:
        return
    progress.analysis_status = status
    progress.analysis_error = error
    memory.class_progress[class_id] = progress
    await store.update_memory(path_id, memory)

@router.post("/measure", response_model=RunResult)
async def measure(session_id: str) -> RunResult:
    return await _measure_session(session_id)

async def _run_analysis_job(session_id: str) -> None:
    await _set_class_analysis_status(session_id, "running")
    await store.set_analysis_job(AnalysisJob(
        session_id=session_id, status="running", updated_at=time.time()
    ))
    try:
        await _measure_session(session_id)
        await store.set_analysis_job(AnalysisJob(
            session_id=session_id, status="complete", updated_at=time.time()
        ))
        await _set_class_analysis_status(session_id, "complete")
    except Exception as exc:
        error = str(exc)[:500]
        await store.set_analysis_job(AnalysisJob(
            session_id=session_id,
            status="failed",
            error=error,
            updated_at=time.time(),
        ))
        await _set_class_analysis_status(session_id, "failed", error)

@router.post("/analysis/{session_id}", response_model=AnalysisJob, status_code=202)
async def analysis_start(session_id: str, background_tasks: BackgroundTasks) -> AnalysisJob:
    if not await store.get_transcript(session_id):
        raise HTTPException(404, f"no transcript for session {session_id!r}")
    current = await store.get_analysis_job(session_id)
    if current and current.status in {"pending", "running", "complete"}:
        return current
    job = AnalysisJob(session_id=session_id, status="pending", updated_at=time.time())
    await store.set_analysis_job(job)
    await _set_class_analysis_status(session_id, "pending")
    background_tasks.add_task(_run_analysis_job, session_id)
    return job

@router.get("/analysis/{session_id}", response_model=AnalysisStatusResponse)
async def analysis_status(session_id: str) -> AnalysisStatusResponse:
    job = await store.get_analysis_job(session_id)
    run = await store.get_run(session_id)
    if job is None and run is None:
        raise HTTPException(404, f"no analysis job for session {session_id!r}")
    if job is None:
        job = AnalysisJob(session_id=session_id, status="complete", updated_at=time.time())
    analyses = await store.get_analyses(session_id)
    fusion_result = None
    if analyses or run is not None:
        fusion_result = fuse(
            session_id,
            analyses,
            await store.get_scores(session_id),
            run.per_question if run else [],
        )
    return AnalysisStatusResponse(**job.model_dump(), run=run, fusion=fusion_result)

@router.get("/sessions/{session_id}", response_model=SessionSnapshot)
async def session_snapshot(session_id: str) -> SessionSnapshot:
    transcript = await store.get_transcript(session_id)
    analyses = await store.get_analyses(session_id)
    questions = await store.get_history(session_id)
    run = await store.get_run(session_id)
    if not transcript and not analyses and not questions and run is None:
        raise HTTPException(404, f"unknown session {session_id!r}")
    fusion_result = None
    if analyses or run is not None:
        fusion_result = fuse(
            session_id,
            analyses,
            await store.get_scores(session_id),
            run.per_question if run else [],
        )
    return SessionSnapshot(
        session_id=session_id,
        transcript=transcript,
        analyses=analyses,
        questions=questions,
        run=run,
        fusion=fusion_result,
    )

@router.get("/confusion/health")
async def confusion_health() -> dict:
    return await client.health()

@router.post("/confusion/analyze", response_model=ChunkAnalysis)
async def confusion_analyze(
    session_id: str = Form(...),
    chunk_id: int = Form(0),
    history: str = Form("[]"),
    enable_space_c: bool | None = Form(None),
    overall_topic: str = Form(""),
    curriculum_context: str = Form(""),
    key_concepts: str = Form("[]"),
    audio: UploadFile = File(...),
) -> ChunkAnalysis:
    hist = json_string_list(history)
    if not hist:
        hist = [s.text for s in await store.get_transcript(session_id)]

    audio_bytes = await audio.read()
    analysis = await client.analyze_audio(
        audio_bytes, filename=audio.filename or "chunk.wav", chunk_id=chunk_id,
        history=hist, enable_space_c=enable_space_c,
        overall_topic=overall_topic,
        curriculum_context=curriculum_context,
        key_concepts=json_string_list(key_concepts),
    )
    await store.append_analysis(session_id, analysis)
    return analysis

@router.post("/questions/from_chunk", response_model=ChunkQuestionResponse)
async def questions_from_chunk(
    session_id: str = Form(...),
    chunk_id: int = Form(0),
    topic: str | None = Form(None),
    history: str = Form("[]"),
    curriculum_context: str = Form(""),
    key_concepts: str = Form("[]"),
    enable_space_c: bool | None = Form(False),
    audio: UploadFile = File(...),
) -> ChunkQuestionResponse:
    hist = json_string_list(history)
    if not hist:
        hist = [a.text for a in await store.get_analyses(session_id)] or [
            s.text for s in await store.get_transcript(session_id)
        ]

    audio_bytes = await audio.read()
    analysis = await client.analyze_audio(
        audio_bytes, filename=audio.filename or "chunk.wav", chunk_id=chunk_id,
        history=hist, enable_space_c=enable_space_c,
        overall_topic=topic or "",
        curriculum_context=curriculum_context,
        key_concepts=json_string_list(key_concepts),
    )
    await store.append_analysis(session_id, analysis)

    if topic:
        await store.set_topic(session_id, topic)
    topic = topic or await store.get_topic(session_id)

    if analysis.student_question and analysis.student_question.question_text.strip():
        generated = analysis.student_question
        question = TargetedQuestion(
            id=await store.next_question_id(session_id),
            chunk_id=analysis.chunk_id,
            text=generated.question_text.strip(),
            anomaly_type=generated.anomaly_type,
            rationale=f"GPU confusion signal on {generated.target_concept}",
        )
        await store.record_questions(session_id, [question])
        return ChunkQuestionResponse(asked=True, analysis=analysis, question=question)

    if not engine.is_confused(analysis):
        return ChunkQuestionResponse(asked=False, analysis=analysis, question=None)

    questions = await generate_targeted_questions(
        [analysis], await store.get_history(session_id),
        start_id=await store.next_question_id(session_id), topic=topic or None,
    )
    await store.record_questions(session_id, questions)
    return ChunkQuestionResponse(
        asked=True, analysis=analysis,
        question=questions[0] if questions else None,
    )

@router.post("/confusion/ingest")
async def confusion_ingest(req: IngestRequest) -> dict:
    await store.set_analyses(req.session_id, req.chunks)
    return {"session_id": req.session_id, "n_chunks": len(req.chunks)}

@router.post("/confusion/mock")
async def confusion_mock(session_id: str) -> dict:
    transcript = await store.get_transcript(session_id)
    if not transcript:
        raise HTTPException(404, f"no transcript for session {session_id!r}")
    analyses = engine.analyze([s.text for s in transcript])
    await store.set_analyses(session_id, analyses)
    return {"session_id": session_id, "n_chunks": len(analyses)}

@router.get("/fusion/{session_id}", response_model=FusionResult)
async def fusion_view(session_id: str) -> FusionResult:
    analyses = await store.get_analyses(session_id)
    run = await store.get_run(session_id)
    scores = await store.get_scores(session_id)
    if not analyses and run is None:
        raise HTTPException(404, f"no analyses or measurement run for session {session_id!r}")
    per_question = run.per_question if run else []
    return fuse(session_id, analyses, scores, per_question)

@router.post("/questions/next", response_model=list[TargetedQuestion])
async def questions_next(req: NextQuestionsRequest) -> list[TargetedQuestion]:
    analyses = await store.get_analyses(req.session_id)
    if not analyses:
        raise HTTPException(404, f"no analyses for session {req.session_id!r}; ingest or mock first")

    chunks = engine.select_low_confidence(
        analyses, k=req.n, exclude_ids=await store.covered_chunk_ids(req.session_id)
    )
    history = await store.get_history(req.session_id)
    questions = await generate_targeted_questions(
        chunks, history, start_id=await store.next_question_id(req.session_id)
    )
    await store.record_questions(req.session_id, questions)
    return questions

@router.post("/questions/answer")
async def questions_answer(req: AnswerRequest) -> dict:
    ok = await store.record_answer(req.session_id, req.question_id, req.answer)
    if not ok:
        raise HTTPException(404, f"unknown question_id {req.question_id} for session {req.session_id!r}")
    return {"session_id": req.session_id, "question_id": req.question_id, "recorded": True}

@router.get("/questions/history/{session_id}", response_model=list[QAEntry])
async def questions_history(session_id: str) -> list[QAEntry]:
    return await store.get_history(session_id)
