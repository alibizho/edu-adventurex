from __future__ import annotations

import re

from ..agents.prompts import STUDENT_SYSTEM
from ..config import settings
from ..llm import student_chat, verifier_chat
from ..schemas import ClassObjective, ClassUnit, Segment
from .prompts import COVERAGE_SYSTEM, GOAL_PROBE_SYSTEM

_CREDIT = re.compile(r"^\s*(\d+)\s*\|\s*(.+?)\s*$")

def transcript_excerpt(segments: list[Segment], limit: int = 4000) -> str:
    return "\n".join(s.text for s in segments)[-limit:]

async def judge_coverage(
    cls: ClassUnit, open_objectives: list[ClassObjective], transcript: str
) -> dict[str, str]:
    if not open_objectives or not transcript.strip():
        return {}

    numbered = "\n".join(f"{i}. {o.text}" for i, o in enumerate(open_objectives, start=1))
    user = (
        f"CLASS: {cls.title} — {cls.objective}\n\n"
        f"OPEN OBJECTIVES:\n{numbered}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    verdict = await verifier_chat(COVERAGE_SYSTEM, user, temperature=0.0)

    covered: dict[str, str] = {}
    for line in verdict.splitlines():
        if line.strip().upper().startswith("NONE"):
            continue
        match = _CREDIT.match(line)
        if not match:
            continue
        index = int(match.group(1))
        if 1 <= index <= len(open_objectives):
            covered[open_objectives[index - 1].id] = match.group(2)[:300]
    return covered

async def goal_probe(cls: ClassUnit, objective: ClassObjective, transcript: str) -> str:
    user = (
        f"CLASS: {cls.title} — {cls.objective}\n\n"
        f"WHAT THEY HAVE NOT COVERED YET:\n{objective.text}\n\n"
        f"WHAT THEY HAVE BEEN SAYING:\n{transcript[-1500:]}"
    )
    question = await student_chat(
        f"{STUDENT_SYSTEM}\n\n{GOAL_PROBE_SYSTEM}", user, temperature=0.6
    )
    return question.strip().strip('"').split("\n")[0][:300]

def should_check(new_segments: int) -> bool:
    return new_segments >= max(1, settings.objective_check_every)
