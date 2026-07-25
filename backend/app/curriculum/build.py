from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from ..config import settings
from ..llm_lc import generator_llm
from ..schemas import (
    BuildPlanRequest,
    ClassObjective,
    ClassUnit,
    GrowthPath,
    PathMemory,
    ScopeRequest,
    TopicScope,
)
from .prompts import NOTES_SYSTEM, SCOPE_SYSTEM, STRUCTURE_SYSTEM

log = logging.getLogger("curriculum.build")

class _ClassDraft(BaseModel):
    class_id: str
    title: str
    objective: str
    objectives: list[str] = Field(default_factory=list)
    difficulty: str = "beginner"
    prerequisites: list[str] = Field(default_factory=list)

class _CurriculumDraft(BaseModel):
    classes: list[_ClassDraft]
    recommended_order: list[str] = Field(default_factory=list)

class CurriculumGenerationError(RuntimeError):
    pass

STRUCTURE_FAILED = (
    "THE LANGUAGE MODEL COULD NOT PRODUCE A VALID COURSE STRUCTURE. PLEASE TRY AGAIN."
)

def _json_system(prompt: str, schema: type[BaseModel]) -> str:
    compact_schema = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return (
        f"{prompt}\n\nReturn ONLY one valid JSON object matching this schema exactly:\n"
        f"{compact_schema}"
    )

def _material_section(material: str | None, limit: int = 6000) -> str:
    material = (material or "").strip()
    return f"\n\nMATERIAL PROVIDED (excerpt):\n{material[:limit]}" if material else ""

async def scope_topic(req: ScopeRequest) -> TopicScope:
    llm = generator_llm(temperature=0.2).with_structured_output(
        TopicScope, method="json_mode"
    )
    pref = (
        f"\n\nPREFERRED NUMBER OF CLASSES: {req.preferred_classes}"
        if req.preferred_classes
        else ""
    )
    user = f'STUDENT REQUEST: "{req.original_input}"{_material_section(req.material_text)}{pref}'
    try:
        result = await llm.ainvoke(
            [
                {"role": "system", "content": _json_system(SCOPE_SYSTEM, TopicScope)},
                {"role": "user", "content": user},
            ]
        )
    except OutputParserException as exc:
        raise CurriculumGenerationError("topic scoping returned invalid JSON") from exc
    if result is None:
        raise CurriculumGenerationError("topic scoping returned no structured output")
    return result

async def structure_curriculum(
    confirmed_topic: str, num_classes: int, material: str | None = None
) -> tuple[list[ClassUnit], list[str]]:
    llm = generator_llm(temperature=0.3).with_structured_output(
        _CurriculumDraft, method="json_mode"
    )
    user = (
        f'TOPIC: "{confirmed_topic}"\n'
        f"NUMBER OF CLASSES: {num_classes}"
        f"{_material_section(material)}"
    )
    try:
        draft: _CurriculumDraft = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": _json_system(STRUCTURE_SYSTEM, _CurriculumDraft),
                },
                {"role": "user", "content": user},
            ]
        )
    except OutputParserException as exc:
        raise CurriculumGenerationError("curriculum structure returned invalid JSON") from exc
    if draft is None or not draft.classes:
        raise CurriculumGenerationError("curriculum structure returned no classes")
    classes = [
        ClassUnit(
            class_id=c.class_id,
            title=c.title,
            objective=c.objective,
            objectives=[
                ClassObjective(id=f"o{i}", text=text.strip())
                for i, text in enumerate(c.objectives, start=1)
                if text.strip()
            ],
            difficulty=c.difficulty,
            prerequisites=c.prerequisites,
        )
        for c in draft.classes
    ]
    order = draft.recommended_order or [c.class_id for c in classes]
    return classes, order

async def build_plan(req: BuildPlanRequest) -> GrowthPath:
    n = req.num_classes if req.num_classes is not None else settings.default_classes
    n = max(1, n)
    material = (req.material_text or "").strip()
    classes, order = await structure_curriculum(req.confirmed_topic, n, material)
    path = GrowthPath(
        path_id=f"gp-{uuid.uuid4().hex[:8]}",
        original_input=req.original_input,
        confirmed_topic=req.confirmed_topic,
        total_classes=len(classes),
        recommended_order=order,
        classes=classes,
        source_material_summary=(material[:500] + "…") if material else None,
    )
    failed = await generate_all_class_notes(path, material)
    if failed:
        log.warning(
            "plan %s built with %d/%d class(es) missing notes: %s",
            path.path_id, len(failed), len(path.classes), ", ".join(failed),
        )
    return path

async def build_plan_events(req: BuildPlanRequest) -> AsyncIterator[dict[str, Any]]:
    n = max(1, req.num_classes if req.num_classes is not None else settings.default_classes)
    material = (req.material_text or "").strip()
    yield {"stage": "topic", "topic": req.confirmed_topic, "classes": n}

    yield {"stage": "structuring", "topic": req.confirmed_topic, "classes": n}
    try:
        classes, order = await structure_curriculum(req.confirmed_topic, n, material)
    except CurriculumGenerationError as exc:
        log.warning("curriculum structure failed: %r", exc)
        yield {"stage": "error", "message": STRUCTURE_FAILED}
        return

    path = GrowthPath(
        path_id=f"gp-{uuid.uuid4().hex[:8]}",
        original_input=req.original_input,
        confirmed_topic=req.confirmed_topic,
        total_classes=len(classes),
        recommended_order=order,
        classes=classes,
        source_material_summary=(material[:500] + "…") if material else None,
    )
    for index, cls in enumerate(path.classes, start=1):
        yield {"stage": "class", "index": index, "total": len(path.classes), "title": cls.title}

    yield {"stage": "writing", "total": len(path.classes)}
    done: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
    writer = asyncio.create_task(
        generate_all_class_notes(
            path, material, on_done=lambda c, ok: done.put_nowait((c.title, ok))
        )
    )
    written = 0
    while written < len(path.classes):
        try:
            title, ok = await asyncio.wait_for(done.get(), timeout=0.5)
        except TimeoutError:
            if writer.done():
                break
            continue
        written += 1
        yield {
            "stage": "written", "index": written, "total": len(path.classes),
            "title": title, "ok": ok,
        }
    await writer
    yield {"stage": "done", "path": path}

def _ordered_split(
    path: GrowthPath, cls: ClassUnit
) -> tuple[list[ClassUnit], list[ClassUnit]]:
    order = list(path.recommended_order) or [c.class_id for c in path.classes]
    order += [c.class_id for c in path.classes if c.class_id not in order]
    by_id = {c.class_id: c for c in path.classes}
    idx = order.index(cls.class_id) if cls.class_id in order else len(order)
    earlier = [by_id[cid] for cid in order[:idx] if cid in by_id]
    later = [by_id[cid] for cid in order[idx + 1:] if cid in by_id]
    return earlier, later

def _outline_block(label: str, classes: list[ClassUnit]) -> str:
    if not classes:
        return f"{label}: (none)"
    body = "\n".join(
        f"  {c.title}\n" + "\n".join(f"    - {o.text}" for o in c.checklist())
        for c in classes
    )
    return f"{label}:\n{body}"

def _notes_user_message(
    path: GrowthPath,
    cls: ClassUnit,
    *,
    covered_concepts: Sequence[str] = (),
    material: str | None = None,
) -> str:
    earlier, later = _ordered_split(path, cls)
    objectives = "\n".join(f"    - {o.text}" for o in cls.checklist())
    covered = ", ".join(dict.fromkeys(c for c in covered_concepts if c.strip()))
    covered_line = f"\n\nALSO ALREADY COVERED IN PREVIOUS SESSIONS: {covered}" if covered else ""
    return (
        f'OVERALL TOPIC: "{path.confirmed_topic}"\n'
        f'THIS CLASS ({len(earlier) + 1} of {len(path.classes)}): "{cls.title}"\n'
        f"DIFFICULTY: {cls.difficulty}\n\n"
        f"TEACH EXACTLY THESE OBJECTIVES, AND NOTHING BEYOND THEM:\n{objectives}\n\n"
        f"{_outline_block('EARLIER CLASSES (already taught — do not re-teach)', earlier)}\n"
        f"{_outline_block('LATER CLASSES (taught after this one — do not pre-empt)', later)}"
        f"{covered_line}"
        f"{_material_section(material, limit=4000)}"
    )

async def _write_notes(user: str) -> str:
    resp = await generator_llm(temperature=0.35).ainvoke(
        [{"role": "system", "content": NOTES_SYSTEM}, {"role": "user", "content": user}]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return content.strip()

async def generate_all_class_notes(
    path: GrowthPath,
    material: str | None = None,
    *,
    concurrency: int | None = None,
    on_done: Callable[[ClassUnit, bool], None] | None = None,
) -> list[str]:
    sem = asyncio.Semaphore(max(1, concurrency or settings.notes_concurrency))

    async def one(cls: ClassUnit) -> str:
        async with sem:
            notes = await asyncio.wait_for(
                _write_notes(_notes_user_message(path, cls, material=material)),
                timeout=settings.notes_timeout,
            )
        if notes.strip():
            cls.teacher_notes = notes
            cls.notes_generated = True
        if on_done is not None:
            on_done(cls, cls.notes_generated)
        return notes

    results = await asyncio.gather(
        *(one(cls) for cls in path.classes), return_exceptions=True
    )

    failed: list[str] = []
    for cls, result in zip(path.classes, results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException) or not str(result).strip():
            failed.append(cls.class_id)
            log.warning("notes generation failed for %s: %r", cls.class_id, result)
            if on_done is not None and isinstance(result, BaseException):
                on_done(cls, False)
    return failed

async def generate_class_notes(path: GrowthPath, cls: ClassUnit, memory: PathMemory) -> str:
    return await _write_notes(
        _notes_user_message(
            path,
            cls,
            covered_concepts=memory.covered_concepts,
            material=path.source_material_summary,
        )
    )
