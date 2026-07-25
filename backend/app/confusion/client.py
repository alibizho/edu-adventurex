import json
import logging

import httpx

from ..config import settings
from ..schemas import ChunkAnalysis

log = logging.getLogger("confusion.client")

def _neutral(chunk_id: int, reason: str) -> ChunkAnalysis:
    log.warning("ml-service analyze failed (%s); returning neutral analysis for chunk %d",
                reason, chunk_id)
    return ChunkAnalysis(chunk_id=chunk_id, text="", confidence=1.0)

async def analyze_audio_with_status(
    audio: bytes,
    filename: str = "chunk.wav",
    chunk_id: int = 0,
    history: list[str] | None = None,
    enable_space_c: bool | None = None,
    overall_topic: str = "",
    curriculum_context: str = "",
    key_concepts: list[str] | None = None,
    focus_target: str = "",
) -> tuple[ChunkAnalysis, bool]:
    data: dict[str, str] = {
        "chunk_id": str(chunk_id),
        "history": json.dumps(history or []),
        "overall_topic": overall_topic,
        "curriculum_context": curriculum_context,
        "key_concepts": json.dumps(key_concepts or []),
        "focus_target": focus_target,
    }
    if enable_space_c is not None:
        data["enable_space_c"] = str(enable_space_c).lower()
    files = {"audio": (filename, audio, "audio/wav")}
    url = f"{settings.ml_service_url.rstrip('/')}/analyze"

    try:
        async with httpx.AsyncClient(timeout=settings.ml_service_timeout) as http:
            resp = await http.post(url, data=data, files=files)
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        return _neutral(chunk_id, repr(e)), True

    try:
        return ChunkAnalysis.model_validate(resp.json()), False
    except (ValueError, KeyError) as e:
        return _neutral(chunk_id, f"bad payload: {e!r}"), True

async def analyze_audio(
    audio: bytes,
    filename: str = "chunk.wav",
    chunk_id: int = 0,
    history: list[str] | None = None,
    enable_space_c: bool | None = None,
    overall_topic: str = "",
    curriculum_context: str = "",
    key_concepts: list[str] | None = None,
    focus_target: str = "",
) -> ChunkAnalysis:
    analysis, _ = await analyze_audio_with_status(
        audio,
        filename=filename,
        chunk_id=chunk_id,
        history=history,
        enable_space_c=enable_space_c,
        overall_topic=overall_topic,
        curriculum_context=curriculum_context,
        key_concepts=key_concepts,
        focus_target=focus_target,
    )
    return analysis

async def health() -> dict:
    url = f"{settings.ml_service_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=settings.ml_service_health_timeout) as http:
            resp = await http.get(url)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as e:
        return {"reachable": False, "error": repr(e), "url": url}
    except ValueError as e:
        return {"reachable": False, "error": f"bad health payload: {e!r}", "url": url}
    return {"reachable": bool(body.get("ok", True)), **body}
