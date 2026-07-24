"""Learning-plan HTTP surface. The new "learn by teaching" front of the experience:

    POST /plan/scope                          confirm / narrow the topic
    POST /plan/build                          build the ~5-class plan (no notes yet)
    GET  /plan/{path_id}                       fetch the plan
    POST /plan/{path_id}/class/{cid}/notes     generate this class's teacher's notes (Markdown)
    POST /plan/{path_id}/class/{cid}/teach/turn teach a turn; a question fires only when unconfident
    POST /plan/{path_id}/class/{cid}/end        "End class"; folds the class into cross-class memory

Thin — the LLM work lives in app/curriculum/, memory + reuse of the confusion/targeted pipeline in
app/curriculum/teaching.py.
"""
from fastapi import APIRouter, HTTPException

from ..curriculum.build import build_plan, generate_class_notes, scope_topic
from ..curriculum.teaching import class_teach_turn, end_class
from ..schemas import (
    BuildPlanRequest,
    ClassTeachResponse,
    ClassUnit,
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
    return await scope_topic(req)


@router.post("/build", response_model=GrowthPath)
async def plan_build(req: BuildPlanRequest) -> GrowthPath:
    """Build and store the teaching plan (classes present; teacher's notes generated lazily)."""
    path = await build_plan(req)
    await store.save_path(path)
    return path


@router.get("/{path_id}", response_model=GrowthPath)
async def plan_get(path_id: str) -> GrowthPath:
    return await _load_path(path_id)


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


@router.post("/{path_id}/class/{class_id}/end", response_model=PathMemory)
async def class_end(path_id: str, class_id: str) -> PathMemory:
    """'End class' button: fold this class into cross-class memory and return the updated memory."""
    path = await _load_path(path_id)
    cls = _find_class(path, class_id)
    return await end_class(path_id, class_id, cls)
