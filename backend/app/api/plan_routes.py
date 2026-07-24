"""Learning-plan HTTP surface. The new "learn by teaching" front of the experience:

    POST /plan/scope                           confirm / narrow the topic
    POST /plan/build                           build the ~5-class plan (no notes yet)
    GET  /plan                                 every stored plan (the UI's path picker)
    GET  /plan/{path_id}                       fetch the plan
    GET  /plan/{path_id}/memory                cross-class memory + per-class progress
    POST /plan/{path_id}/class/{cid}/notes     generate this class's teacher's notes (Markdown)
    POST /plan/{path_id}/class/{cid}/teach/turn teach a turn; a question fires only when unconfident
    POST /plan/{path_id}/class/{cid}/teach/audio-turn  same, from recorded speech via the ml-service
    POST /plan/{path_id}/class/{cid}/end        "End class"; folds the class into cross-class memory

Thin — the LLM work lives in app/curriculum/, memory + reuse of the confusion/targeted pipeline in
app/curriculum/teaching.py.
"""
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..curriculum.build import (
    CurriculumGenerationError,
    build_plan,
    generate_class_notes,
    scope_topic,
)
from ..confusion import client
from ..curriculum.teaching import class_audio_turn, class_teach_turn, end_class
from ..schemas import (
    AudioClassTeachResponse,
    BuildPlanRequest,
    ClassTeachResponse,
    ClassUnit,
    EndClassRequest,
    GrowthPath,
    PathMemory,
    ScopeRequest,
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
    """Build and store the teaching plan (classes present; teacher's notes generated lazily)."""
    try:
        path = await build_plan(req)
    except CurriculumGenerationError as exc:
        raise HTTPException(
            502,
            "THE LANGUAGE MODEL COULD NOT PRODUCE A VALID COURSE STRUCTURE. PLEASE TRY AGAIN.",
        ) from exc
    await store.save_path(path)
    return path


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
async def class_notes(path_id: str, class_id: str) -> ClassUnit:
    """Generate the teacher's notes (Markdown) for this class, memory-aware so it doesn't repeat
    earlier classes. Persisted onto the plan and returned."""
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
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
    chunk_id: int = Form(0),
    history: str = Form("[]"),
    audio: UploadFile = File(...),
) -> AudioClassTeachResponse:
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
    audio_bytes = await audio.read()
    if len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(413, "audio chunk exceeds 15 MB")
    try:
        parsed = json.loads(history) if history else []
        parsed_history = [str(item) for item in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        parsed_history = []

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
    analysis, degraded = await client.analyze_audio_with_status(
        audio_bytes,
        filename=audio.filename or "chunk.wav",
        chunk_id=chunk_id,
        history=parsed_history,
        enable_space_c=bool(curriculum_context),
        overall_topic=path.confirmed_topic,
        curriculum_context=curriculum_context,
        key_concepts=key_concepts,
    )
    return await class_audio_turn(path_id, class_id, cls, analysis, degraded)


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
