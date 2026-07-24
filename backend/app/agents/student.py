"""The in-character student the kid teaches. Only asks / restates / admits confusion."""
from ..llm import student_chat
from ..schemas import Segment, TeachTurnResponse
from .prompts import STUDENT_SYSTEM


def _render_transcript(transcript: list[Segment]) -> str:
    return "\n".join(f"[{s.id}] {s.text}" for s in transcript)


async def student_turn(transcript: list[Segment], latest_utterance: str) -> TeachTurnResponse:
    """Produce the student's next reply and append the kid's utterance as a new segment."""
    context = _render_transcript(transcript)
    user = (
        f"Conversation so far:\n{context}\n\n"
        f"The kid just said:\n{latest_utterance}\n\n"
        f"Respond in character (ask, restate, or admit confusion — never explain)."
    )
    reply = await student_chat(STUDENT_SYSTEM, user, temperature=0.7)

    next_id = (max((s.id for s in transcript), default=-1)) + 1
    new_segment = Segment(id=next_id, idx=len(transcript), text=latest_utterance)
    return TeachTurnResponse(student_reply=reply, new_segment=new_segment)
