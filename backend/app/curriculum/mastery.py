"""Objective mastery: did the learner actually explain the things this class is about?

Readiness used to be `25 + turn_count * 15` — a turn counter wearing a percentage, which said 95%
after five turns of talking about anything at all. Here it comes from what was said instead.

Two jobs, both cheap and both off the teaching path:
  judge_coverage   -> which open objectives the new speech demonstrably covered
  goal_probe       -> a student's question about one that is still open

Both run from a background task (see `teaching.run_objective_check`), batched over several
utterances, so continuous teaching never waits on either.
"""
from __future__ import annotations

import re

from ..agents.prompts import STUDENT_SYSTEM
from ..config import settings
from ..llm import student_chat, verifier_chat
from ..schemas import ClassObjective, ClassUnit, Segment
from .prompts import COVERAGE_SYSTEM, GOAL_PROBE_SYSTEM

# "3 | because the forces cancel out" — index, then the quote that earned it.
_CREDIT = re.compile(r"^\s*(\d+)\s*\|\s*(.+?)\s*$")


def transcript_excerpt(segments: list[Segment], limit: int = 4000) -> str:
    """The learner's own words, newest last. Trimmed from the front: a long class shouldn't grow
    the prompt without bound, and recent speech is what hasn't been judged yet."""
    return "\n".join(s.text for s in segments)[-limit:]


async def judge_coverage(
    cls: ClassUnit, open_objectives: list[ClassObjective], transcript: str
) -> dict[str, str]:
    """Return {objective_id: evidence quote} for objectives this speech covered.

    One verifier call for the whole batch. Anything unparseable is treated as "covered nothing" —
    a missed credit costs the learner a few more seconds of teaching, whereas a false credit marks
    a class understood when it wasn't, which is the failure that matters.
    """
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
    """One student question aimed at an objective the learner hasn't covered yet."""
    user = (
        f"CLASS: {cls.title} — {cls.objective}\n\n"
        f"WHAT THEY HAVE NOT COVERED YET:\n{objective.text}\n\n"
        f"WHAT THEY HAVE BEEN SAYING:\n{transcript[-1500:]}"
    )
    # The student persona keeps the voice consistent with every other question in the room;
    # GOAL_PROBE_SYSTEM supplies the one thing it doesn't know — that this is a gap, not confusion.
    question = await student_chat(
        f"{STUDENT_SYSTEM}\n\n{GOAL_PROBE_SYSTEM}", user, temperature=0.6
    )
    return question.strip().strip('"').split("\n")[0][:300]


def should_check(new_segments: int) -> bool:
    return new_segments >= max(1, settings.objective_check_every)
