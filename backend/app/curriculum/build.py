"""Curriculum builder — the build steps, all LangChain over the DeepSeek endpoint:

  scope_topic            -> TopicScope      (confirm / narrow the topic)
  build_plan             -> GrowthPath      (structure into ~5 ordered classes, notes and all)
  generate_all_class_notes -> None          (every class's notes at once, whole outline in view)
  generate_class_notes   -> Markdown str    (one class's notes, memory-aware; the lazy backfill)

Structured shapes are produced with `.with_structured_output(..., method="json_mode")`. Draft
models keep the LLM from filling `teacher_notes` / `notes_generated` in the structuring call —
titles and objectives first, then the notes step writes every class's material against that
outline, which is what stops two classes teaching the same thing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence

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
    # The checkable breakdown of `objective`. Asked for in the same structuring call rather than a
    # follow-up, so goals cost nothing extra to produce.
    objectives: list[str] = Field(default_factory=list)
    difficulty: str = "beginner"
    prerequisites: list[str] = Field(default_factory=list)


class _CurriculumDraft(BaseModel):
    classes: list[_ClassDraft]
    recommended_order: list[str] = Field(default_factory=list)


class CurriculumGenerationError(RuntimeError):
    """The model responded, but did not produce the requested curriculum shape."""


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
    """Confirm the topic or, if too broad, return 3 narrower options."""
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
    """Return (classes, recommended_order) — titles/objectives/ordering only, no notes."""
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
    """Assemble the GrowthPath and write every class's teacher's notes.

    Both steps here, not one now and the rest later: the material for a class is written against
    the finished outline, so each class knows what its siblings teach and stays off their ground.
    Generated one class at a time (in parallel) rather than in one call — a course's worth of
    Markdown in a single response runs into the output cap and truncates the last classes.

    The notes step never raises. A course that could not be STRUCTURED is a 502 (there is nothing
    to show); a course whose notes tier was rate-limited still gives the learner their classes,
    with the /notes route backfilling whatever is missing.
    """
    # `is not None` so an explicit 0 isn't silently coerced to the default; clamp to a sane minimum.
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
        # Only a prefix is persisted, so the notes calls below — the one moment the whole upload
        # is in hand — are given `material` itself rather than this.
        source_material_summary=(material[:500] + "…") if material else None,
    )
    failed = await generate_all_class_notes(path, material)
    if failed:
        log.warning(
            "plan %s built with %d/%d class(es) missing notes: %s",
            path.path_id, len(failed), len(path.classes), ", ".join(failed),
        )
    return path


def _ordered_split(
    path: GrowthPath, cls: ClassUnit
) -> tuple[list[ClassUnit], list[ClassUnit]]:
    """(earlier, later) around `cls` in the recommended teaching order.

    Both halves matter to the notes prompt: earlier classes are what must not be re-taught, later
    ones are what must not be pre-empted. A class missing from `recommended_order` is appended
    rather than dropped — the model writes that list, and a class the outline forgot would
    otherwise be invisible to every one of its siblings.
    """
    order = list(path.recommended_order) or [c.class_id for c in path.classes]
    order += [c.class_id for c in path.classes if c.class_id not in order]
    by_id = {c.class_id: c for c in path.classes}
    # Absent from the order = treated as last, so everything else counts as earlier.
    idx = order.index(cls.class_id) if cls.class_id in order else len(order)
    earlier = [by_id[cid] for cid in order[:idx] if cid in by_id]
    later = [by_id[cid] for cid in order[idx + 1:] if cid in by_id]
    return earlier, later


def _outline_block(label: str, classes: list[ClassUnit]) -> str:
    """One side of the outline: each class as its title plus the objectives it owns.

    Titles alone were the old context, and they are too coarse — "Forces was covered" leaves the
    model to guess what that meant, which is how two classes end up teaching the same idea.
    """
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
    """The notes prompt, shared by the eager build and the lazy /notes route so the two can't
    drift. Everything the class must not repeat and must not pre-empt is stated explicitly."""
    earlier, later = _ordered_split(path, cls)
    objectives = "\n".join(f"    - {o.text}" for o in cls.checklist())
    # Only what the learner actually taught in earlier sessions. Empty at build time by
    # definition, so the line is dropped rather than rendered as a dead "(nothing yet)".
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
    path: GrowthPath, material: str | None = None, *, concurrency: int | None = None
) -> list[str]:
    """Write every class's notes at once, each call seeing the whole outline.

    This is what stops classes repeating each other: one call per class, but every one of them is
    told what its siblings own, so the boundaries are decided by the outline rather than guessed
    from a list of titles. Mutates `path.classes` in place.

    Returns the class_ids whose notes could not be written. They keep `notes_generated=False`, and
    the lazy /notes route fills them on demand — a rate-limited notes tier must not cost the
    learner the course they just agreed to.
    """
    sem = asyncio.Semaphore(max(1, concurrency or settings.notes_concurrency))

    async def one(cls: ClassUnit) -> str:
        async with sem:
            # Inside the semaphore: the budget covers the call, not the time spent queued behind
            # an earlier wave.
            return await asyncio.wait_for(
                _write_notes(_notes_user_message(path, cls, material=material)),
                timeout=settings.notes_timeout,
            )

    results = await asyncio.gather(
        *(one(cls) for cls in path.classes), return_exceptions=True
    )

    failed: list[str] = []
    for cls, result in zip(path.classes, results):
        # A cancelled request (the client hung up) is not eight model failures.
        if isinstance(result, asyncio.CancelledError):
            raise result
        # A blank primer counts as failure: persisted with notes_generated=True it would be a
        # permanently empty class, reachable again only via ?regenerate=true.
        if isinstance(result, BaseException) or not str(result).strip():
            failed.append(cls.class_id)
            log.warning("notes generation failed for %s: %r", cls.class_id, result)
            continue
        cls.teacher_notes = str(result)
        cls.notes_generated = True
    return failed


async def generate_class_notes(path: GrowthPath, cls: ClassUnit, memory: PathMemory) -> str:
    """One class's primer on demand. Since `build_plan` writes them all up-front this is the
    backfill: paths built before notes were eager, a deliberate `?regenerate=true` once cross-class
    memory has moved on, and the class whose eager call failed.

    Memory-aware, which the eager path structurally cannot be — at build time nothing has been
    taught yet.
    """
    return await _write_notes(
        _notes_user_message(
            path,
            cls,
            covered_concepts=memory.covered_concepts,
            material=path.source_material_summary,
        )
    )
