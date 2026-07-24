"""HTTP surface. Thin — orchestration lives in agents/ and pipeline/. Open /docs to poke it."""
import json
import time

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

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


def _json_string_list(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


@router.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "llm_configured": bool(settings.student_api_key and settings.generator_api_key),
        "store_backend": settings.store_backend,
    }


@router.post("/teach/turn", response_model=TeachTurnResponse)
async def teach_turn(req: TeachTurnRequest) -> TeachTurnResponse:
    """One turn of the teaching loop: kid speaks, student replies in character."""
    resp = await student_turn(req.transcript, req.latest_utterance)
    await store.append_segment(req.session_id, resp.new_segment)
    return resp


async def _measure_session(session_id: str) -> RunResult:
    """Run the full transfer-delta measurement on a stored transcript.

    generate (with answer keys) -> filter (cold student) -> ensemble score -> delta.
    Correctness comes from the verifier (pipeline/grading.py), which grades each answer against
    the question's ground-truth key. No source is attached to stored sessions yet, so keys fall
    back to the generator's own topic knowledge.
    """
    transcript: list[Segment] = await store.get_transcript(session_id)
    questions = await generate_questions(transcript)

    survivors, survival_rate = await filter_questions(questions)
    result, scores = await score_ensemble(transcript, survivors, survival_rate, session_id)
    await store.set_run(session_id, result, scores)
    return result


async def _set_class_analysis_status(
    session_id: str, status: str, error: str | None = None
) -> None:
    """Mirror an analysis job onto its durable class progress when the session is path:class."""
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
    overall_topic: str = Form(""),
    curriculum_context: str = Form(""),
    key_concepts: str = Form("[]"),
    audio: UploadFile = File(...),
) -> ChunkAnalysis:
    """Forward a recorded utterance to the ml-service confusion engine and store the analysis.
    If `history` is empty, the session's prior segment texts are used as Space B context.
    Degrades to a neutral analysis if the ml-service is unreachable (see confusion/client.py)."""
    hist = _json_string_list(history)
    if not hist:
        hist = [s.text for s in await store.get_transcript(session_id)]

    audio_bytes = await audio.read()
    analysis = await client.analyze_audio(
        audio_bytes, filename=audio.filename or "chunk.wav", chunk_id=chunk_id,
        history=hist, enable_space_c=enable_space_c,
        overall_topic=overall_topic,
        curriculum_context=curriculum_context,
        key_concepts=_json_string_list(key_concepts),
    )
    await store.append_analysis(session_id, analysis)
    return analysis


@router.post("/questions/from_chunk", response_model=ChunkQuestionResponse)
async def questions_from_chunk(
    session_id: str = Form(...),
    chunk_id: int = Form(0),
    topic: str | None = Form(None),
    history: str = Form("[]"),          # JSON array of prior transcripts (Space B context)
    curriculum_context: str = Form(""),
    key_concepts: str = Form("[]"),
    enable_space_c: bool | None = Form(False),   # Space C (fact-check) off by default — its
                                                  # pedantic factual_errors false-positive on
                                                  # correct speech; pass true to re-enable.
    audio: UploadFile = File(...),
) -> ChunkQuestionResponse:
    """Real-time path: take one spoken chunk (the teacher paused), run it through the ml-service
    confusion engine, and — ONLY if the chunk is confused — generate one targeted question about it.
    "Confused" = confidence below `question_confidence_threshold` OR a lexical hesitation marker
    (backstop for when Space A misses obvious hedging); ml-service anomalies are optional
    (`question_gate_on_anomalies`, off by default — the on-box judges are noisy on short utterances).
    Space C (fact-check) is off by default for this flow (its factual_errors false-positive on
    correct speech). The AI students stay silent; the question is the output. Degrades to a neutral
    analysis (`asked=False`) if the ml-service is unreachable."""
    hist = _json_string_list(history)
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
        key_concepts=_json_string_list(key_concepts),
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
    """Store per-chunk analyses from the confusion engine (an external ML can also post here)."""
    await store.set_analyses(req.session_id, req.chunks)
    return {"session_id": req.session_id, "n_chunks": len(req.chunks)}


@router.post("/confusion/mock")
async def confusion_mock(session_id: str) -> dict:
    """Demo path while the ML is absent: run the heuristic mock over the stored transcript's
    segment texts and store the resulting analyses."""
    transcript = await store.get_transcript(session_id)
    if not transcript:
        raise HTTPException(404, f"no transcript for session {session_id!r}")
    analyses = engine.analyze([s.text for s in transcript])
    await store.set_analyses(session_id, analyses)
    return {"session_id": session_id, "n_chunks": len(analyses)}


# ---- fusion: confidence x competence (report §6) ----

@router.get("/fusion/{session_id}", response_model=FusionResult)
async def fusion_view(session_id: str) -> FusionResult:
    """Cross the stored confusion analyses (disturbance) with the transfer-delta run (competence)
    into the per-segment quadrant map + calibration. Run /confusion/analyze (or /mock) and /measure
    first; either alone still returns partial results (segments it can't cross are 'unknown')."""
    analyses = await store.get_analyses(session_id)
    run = await store.get_run(session_id)
    scores = await store.get_scores(session_id)
    if not analyses and run is None:
        raise HTTPException(404, f"no analyses or measurement run for session {session_id!r}")
    per_question = run.per_question if run else []
    return fuse(session_id, analyses, scores, per_question)


@router.post("/questions/next", response_model=list[TargetedQuestion])
async def questions_next(req: NextQuestionsRequest) -> list[TargetedQuestion]:
    """Pick the lowest-confidence uncovered chunks and generate specialized, non-repeating
    questions for them, keyed off the session's Q&A memory."""
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
    """Record the user's answer so the agent won't re-ask it."""
    ok = await store.record_answer(req.session_id, req.question_id, req.answer)
    if not ok:
        raise HTTPException(404, f"unknown question_id {req.question_id} for session {req.session_id!r}")
    return {"session_id": req.session_id, "question_id": req.question_id, "recorded": True}


@router.get("/questions/history/{session_id}", response_model=list[QAEntry])
async def questions_history(session_id: str) -> list[QAEntry]:
    return await store.get_history(session_id)
