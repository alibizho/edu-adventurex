"""Curriculum builder — the three lazy build steps, all LangChain over the DeepSeek endpoint:

  scope_topic         -> TopicScope         (confirm / narrow the topic)
  build_plan          -> GrowthPath         (structure into ~5 ordered classes, no notes yet)
  generate_class_notes -> Markdown str      (brief teacher's notes for one class, memory-aware)

Structured shapes are produced with `.with_structured_output(..., method="json_mode")`.
Draft models keep the LLM
from filling `teacher_notes` / `notes_generated` — those are set lazily by the notes step.
"""
from __future__ import annotations

import json
import uuid

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
    """Assemble a GrowthPath skeleton (classes present, notes not yet generated)."""
    # `is not None` so an explicit 0 isn't silently coerced to the default; clamp to a sane minimum.
    n = req.num_classes if req.num_classes is not None else settings.default_classes
    n = max(1, n)
    material = (req.material_text or "").strip()
    classes, order = await structure_curriculum(req.confirmed_topic, n, material)
    return GrowthPath(
        path_id=f"gp-{uuid.uuid4().hex[:8]}",
        original_input=req.original_input,
        confirmed_topic=req.confirmed_topic,
        total_classes=len(classes),
        recommended_order=order,
        classes=classes,
        source_material_summary=(material[:500] + "…") if material else None,
    )


def _earlier_class_titles(path: GrowthPath, cls: ClassUnit) -> list[str]:
    """Titles of the classes that come before `cls` in the recommended order — treated as already
    covered so notes don't re-teach them, even when notes are generated before those classes are
    taught/ended (cross-class memory only fills covered_concepts on end_class)."""
    order = path.recommended_order or [c.class_id for c in path.classes]
    by_id = {c.class_id: c for c in path.classes}
    idx = order.index(cls.class_id) if cls.class_id in order else len(order)
    return [by_id[cid].title for cid in order[:idx] if cid in by_id]


async def generate_class_notes(path: GrowthPath, cls: ClassUnit, memory: PathMemory) -> str:
    """One 'before the class' step: a brief Markdown primer, told what earlier classes covered so
    it builds on them instead of repeating."""
    covered_list = list(
        dict.fromkeys(_earlier_class_titles(path, cls) + memory.covered_concepts)
    )
    covered = ", ".join(covered_list) or "(nothing yet — this is an early class)"
    material_hint = (
        f"\n\nSOURCE MATERIAL SUMMARY:\n{path.source_material_summary}"
        if path.source_material_summary
        else ""
    )
    user = (
        f'OVERALL TOPIC: "{path.confirmed_topic}"\n'
        f'CLASS: "{cls.title}"\n'
        f"OBJECTIVE: {cls.objective}\n"
        f"DIFFICULTY: {cls.difficulty}\n"
        f"ALREADY COVERED IN EARLIER CLASSES (do not re-teach): {covered}"
        f"{material_hint}"
    )
    resp = await generator_llm(temperature=0.35).ainvoke(
        [{"role": "system", "content": NOTES_SYSTEM}, {"role": "user", "content": user}]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return content.strip()
