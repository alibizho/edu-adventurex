from ..llm import student_chat
from ..schemas import Segment, TeachTurnResponse
from .prompts import STUDENT_SYSTEM

def _render_transcript(transcript: list[Segment]) -> str:
    return "\n".join(f"[{s.id}] {s.text}" for s in transcript)

def next_segment(transcript: list[Segment], text: str) -> Segment:
    next_id = (max((s.id for s in transcript), default=-1)) + 1
    return Segment(id=next_id, idx=len(transcript), text=text)

async def student_turn(transcript: list[Segment], latest_utterance: str) -> TeachTurnResponse:
    context = _render_transcript(transcript)
    user = (
        f"Conversation so far:\n{context}\n\n"
        f"The kid just said:\n{latest_utterance}\n\n"
        f"Respond in character (ask, restate, or admit confusion — never explain)."
    )
    reply = await student_chat(STUDENT_SYSTEM, user, temperature=0.7)
    return TeachTurnResponse(student_reply=reply, new_segment=next_segment(transcript, latest_utterance))
