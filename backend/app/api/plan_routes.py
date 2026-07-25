"""Learning-plan HTTP surface. The new "learn by teaching" front of the experience:

    POST /plan/scope                           confirm / narrow the topic
    POST /plan/build                           build the ~5-class plan, teacher's notes and all
    POST /plan/build/stream                    the same build, reported stage by stage (NDJSON)
    GET  /plan                                 every stored plan (the UI's path picker)
    GET  /plan/{path_id}                       fetch the plan
    GET  /plan/{path_id}/memory                cross-class memory + per-class progress
    POST /plan/{path_id}/class/{cid}/notes     (re)write this class's teacher's notes — normally
                                               already written by /plan/build
    POST /plan/{path_id}/class/{cid}/teach/turn teach a turn; a question fires only when unconfident
    POST /plan/{path_id}/class/{cid}/teach/audio-turn  same, from recorded speech via the ml-service
    POST /plan/{path_id}/class/{cid}/reset      "Start this class over"; erases the class's session
    POST /plan/{path_id}/class/{cid}/end        "End class"; folds the class into cross-class memory

Thin — the LLM work lives in app/curriculum/, memory + reuse of the confusion/targeted pipeline in
app/curriculum/teaching.py.
"""
import json

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .forms import json_string_list

from ..curriculum.build import (
    STRUCTURE_FAILED,
    CurriculumGenerationError,
    build_plan,
    build_plan_events,
    generate_class_notes,
    scope_topic,
)
from ..confusion import client, engine
from ..curriculum.teaching import (
    class_audio_turn,
    class_teach_turn,
    end_class,
    reset_class,
    run_objective_check,
)
from ..schemas import (
    AudioClassTeachResponse,
    BuildPlanRequest,
    ClassTeachResponse,
    ClassUnit,
    EndClassRequest,
    GrowthPath,
    PathMemory,
    ScopeRequest,
    SpeechProsody,
    TeachTurnBody,
    TopicScope,
)
from ..store import store

router = APIRouter(prefix="/plan", tags=["plan"])


async def _load_path(path_id: str) -> GrowthPath:
    path = await store.get_path(path_id)
    if path is None:
        raise HTTPException(404, f"unknown path {path_id!r}")
    return path


def _find_class(path: GrowthPath, class_id: str) -> ClassUnit:
    for c in path.classes:
        if c.class_id == class_id:
            return c
    raise HTTPException(404, f"unknown class {class_id!r} in path {path.path_id!r}")


@router.post("/scope", response_model=TopicScope)
async def plan_scope(req: ScopeRequest) -> TopicScope:
    """Confirm the topic, or — if too broad — return 3 narrower options for the learner to pick."""
    try:
        return await scope_topic(req)
    except CurriculumGenerationError as exc:
        raise HTTPException(
            502,
            "THE LANGUAGE MODEL COULD NOT PRODUCE A VALID TOPIC STRUCTURE. PLEASE TRY AGAIN.",
        ) from exc


@router.post("/build", response_model=GrowthPath)
async def plan_build(req: BuildPlanRequest) -> GrowthPath:
    """Build and store the teaching plan: the classes, plus every class's teacher's notes, written
    in parallel with the whole outline in view so no two classes teach the same thing.

    Slower than it used to be — this is N+1 model calls, not one — and deliberately so: material
    written per class on demand had no way to know what the other classes covered.

    A class whose notes call failed comes back with `notes_generated: false` and is filled by the
    /notes route when the learner opens it. The course itself is never lost to that.
    """
    try:
        path = await build_plan(req)
    except CurriculumGenerationError as exc:
        raise HTTPException(502, STRUCTURE_FAILED) from exc
    await store.save_path(path)
    return path


@router.post("/build/stream")
async def plan_build_stream(req: BuildPlanRequest) -> StreamingResponse:
    """The same build, reported line by line as newline-delimited JSON.

    The build is N+1 model calls and runs the better part of a minute; a spinner over that reads
    as a hang. Stages: `topic` (echoing what was confirmed), `structuring`, one `class` per title
    the structuring call returned, `writing`, then one `written` per class as its material lands —
    in arrival order, not course order — and finally `done` carrying the stored GrowthPath.

    A failure arrives as an `error` event, not a status code: by then the response has already
    started, so 502 is no longer available. Clients must read to `done` or `error`.
    """
    async def stream():
        async for event in build_plan_events(req):
            if event["stage"] == "done":
                path: GrowthPath = event["path"]
                # Stored before the client is told, so the redirect that follows always finds it.
                await store.save_path(path)
                event = {"stage": "done", "path": path.model_dump()}
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.get("", response_model=list[GrowthPath])
async def plan_list() -> list[GrowthPath]:
    return await store.list_paths()


@router.get("/{path_id}", response_model=GrowthPath)
async def plan_get(path_id: str) -> GrowthPath:
    return await _load_path(path_id)


@router.get("/{path_id}/memory", response_model=PathMemory)
async def plan_memory(path_id: str) -> PathMemory:
    await _load_path(path_id)
    return await store.get_memory(path_id)


@router.post("/{path_id}/class/{class_id}/notes", response_model=ClassUnit)
async def class_notes(path_id: str, class_id: str, regenerate: bool = False) -> ClassUnit:
    """Write the teacher's notes (Markdown) for this class, memory-aware so it doesn't repeat what
    the learner has already taught. Persisted onto the plan and returned.

    /plan/build normally writes these already, so this route is the backfill and survives for three
    reasons: paths built before notes were eager, `?regenerate=true` once cross-class memory has
    moved on and the primer should stop re-teaching what the learner has since covered, and the
    class whose eager call failed.

    Already-generated notes are returned as-is — re-entering a class shouldn't rewrite (and
    re-bill) its primer.
    """
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
    if cls.notes_generated and cls.teacher_notes.strip() and not regenerate:
        return cls
    memory = await store.get_memory(path_id)
    cls.teacher_notes = await generate_class_notes(path, cls, memory)
    cls.notes_generated = True
    await store.save_class(path_id, cls)
    return cls


@router.post("/{path_id}/class/{class_id}/teach/turn", response_model=ClassTeachResponse)
async def class_teach(path_id: str, class_id: str, body: TeachTurnBody) -> ClassTeachResponse:
    """One teaching turn: the AI student replies; a targeted question fires only when the learner
    sounds unconfident (and isn't a near-duplicate of a question already asked in any class)."""
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
    return await class_teach_turn(path_id, class_id, cls, body.latest_utterance)


@router.post(
    "/{path_id}/class/{class_id}/teach/audio-turn",
    response_model=AudioClassTeachResponse,
)
async def class_teach_audio(
    path_id: str,
    class_id: str,
    background_tasks: BackgroundTasks,
    chunk_id: int = Form(0),
    history: str = Form("[]"),
    silent: bool = Form(False),   # live classroom: record + analyze, but only speak to ask.
    # Set when the learner is answering one specific student face to face. That turn is graded
    # against the question's answer key and may earn a follow-up, instead of going through the
    # classroom's question logic.
    answering_question_id: int | None = Form(None),
    # Browser-measured delivery. The GPU scores hesitation relative to the same utterance and so
    # can't see speech that's unsure throughout; these are absolute (see SpeechProsody).
    speech_ms: int = Form(0),
    total_ms: int = Form(0),
    pause_count: int = Form(0),
    longest_pause_ms: int = Form(0),
    mean_level: float = Form(0.0),
    peak_level: float = Form(0.0),
    audio: UploadFile = File(...),
) -> AudioClassTeachResponse:
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
    audio_bytes = await audio.read()
    if len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(413, "audio chunk exceeds 15 MB")
    parsed_history = json_string_list(history)

    memory = await store.get_memory(path_id)
    curriculum_context = "\n\n".join(
        part.strip()
        for part in (cls.objective, cls.teacher_notes, path.source_material_summary or "")
        if part and part.strip()
    )
    key_concepts = list(
        dict.fromkeys(
            [cls.title, *memory.covered_concepts, *memory.expanded_concepts]
        )
    )
    # Carried from the previous turn's ledger: the GPU is stateless, so this is what lets its
    # student question keep chasing one weak spot across utterances.
    progress = memory.class_progress.get(class_id)
    analysis, degraded = await client.analyze_audio_with_status(
        audio_bytes,
        filename=audio.filename or "chunk.wav",
        chunk_id=chunk_id,
        history=parsed_history,
        enable_space_c=bool(curriculum_context),
        overall_topic=path.confirmed_topic,
        curriculum_context=curriculum_context,
        key_concepts=key_concepts,
        focus_target=progress.focus_target if progress else "",
    )
    if total_ms > 0:
        engine.fuse_prosody(analysis, SpeechProsody(
            speech_ms=speech_ms, total_ms=total_ms, pause_count=pause_count,
            longest_pause_ms=longest_pause_ms, mean_level=mean_level, peak_level=peak_level,
        ))

    result = await class_audio_turn(
        path_id, class_id, cls, analysis, degraded, silent=silent,
        answering_question_id=answering_question_id,
    )
    # Objective coverage is judged after the response goes out, batched over several utterances —
    # the learner is still talking and must not wait on a verifier call.
    if not result.degraded:
        background_tasks.add_task(run_objective_check, path_id, class_id, cls)
    return result


@router.post("/{path_id}/class/{class_id}/reset", response_model=PathMemory)
async def class_reset(path_id: str, class_id: str) -> PathMemory:
    """'Start this class over': erase the class's session and its progress, and take back what it
    contributed to cross-class memory. The plan and its teacher's notes are untouched — the
    learner gets the same class again, not a new one."""
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
    return await reset_class(path_id, class_id, cls)


@router.post("/{path_id}/class/{class_id}/end", response_model=PathMemory)
async def class_end(
    path_id: str,
    class_id: str,
    body: EndClassRequest | None = None,
) -> PathMemory:
    """'End class' button: fold this class into cross-class memory and return the updated memory."""
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
    return await end_class(
        path_id,
        class_id,
        cls,
        body.completion_mode if body else "self-teaching",
    )
