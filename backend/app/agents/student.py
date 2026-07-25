"""The in-character student the kid teaches. Only asks / restates / admits confusion."""
from ..llm import student_chat
from ..schemas import Segment, TeachTurnResponse
from .prompts import STUDENT_SYSTEM


def _render_transcript(transcript: list[Segment]) -> str:
    return "\n".join(f"[{s.id}] {s.text}" for s in transcript)


def next_segment(transcript: list[Segment], text: str) -> Segment:
    """The kid's utterance as the next segment on the spine. Separate from `student_turn` because
    the live classroom records what was said on every chunk but only pays for a reply when a
    student actually has something to ask — both paths must number segments identically."""
    next_id = (max((s.id for s in transcript), default=-1)) + 1
    return Segment(id=next_id, idx=len(transcript), text=text)


async def student_turn(transcript: list[Segment], latest_utterance: str) -> TeachTurnResponse:
    """Produce the student's next reply and append the kid's utterance as a new segment."""
    context = _render_transcript(transcript)
    user = (
        f"Conversation so far:\n{context}\n\n"
        f"The kid just said:\n{latest_utterance}\n\n"
        f"Respond in character (ask, restate, or admit confusion — never explain)."
    )
    reply = await student_chat(STUDENT_SYSTEM, user, temperature=0.7)
    return TeachTurnResponse(student_reply=reply, new_segment=next_segment(transcript, latest_utterance))
