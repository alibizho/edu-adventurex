from ..config import settings
from ..schemas import Segment
from .prompts import PERSONA_SEEDS, TAUGHT_SYSTEM

def render_transcript(transcript: list[Segment]) -> str:
    return "\n".join(f"[{s.id}] {s.text}" for s in transcript)

def taught_system(persona: str, transcript: list[Segment]) -> str:
    return TAUGHT_SYSTEM.format(persona=persona, transcript=render_transcript(transcript))

def taught_personas() -> list[str]:
    return PERSONA_SEEDS[: settings.n_taught]

def cold_personas() -> list[str]:
    return PERSONA_SEEDS[: settings.n_cold]
