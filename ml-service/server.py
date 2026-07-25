"""FastAPI wrapper around ConfusionEngine. Deploy this on the HyperAI GPU box; the main backend
calls POST /analyze per recorded utterance.

Run:  uvicorn server:app --host 0.0.0.0 --port 8100

Note what this module does NOT import at the top: `engine` (and through it `config`, torch, and
every model) is imported inside the lifespan instead. That keeps `import server` cheap enough for
backend/tests/test_ml_service_contract.py to check the wire contract on a CPU laptop with no
torch installed.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from schemas import ChunkAnalysis

MAX_AUDIO_BYTES = 15 * 1024 * 1024      # one paused utterance, not a lecture recording
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}

# Typed as Any so this module never has to import ConfusionEngine (see the module docstring).
_engine: Any | None = None
# Models are loaded once and are not re-entrant; serialize /analyze so two callers can't drive the
# same GPU weights concurrently. Requests queue rather than OOM.
_analysis_lock = asyncio.Lock()


def _json_string_list(raw: str) -> list[str]:
    """Parse a JSON array form field, degrading to [] rather than 422-ing the whole request."""
    try:
        parsed = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    from engine import ConfusionEngine

    # Loading weights takes minutes and is blocking; off the event loop so /health can answer
    # `ok: false` while it warms up instead of the socket just hanging.
    _engine = await run_in_threadpool(ConfusionEngine)
    try:
        yield
    finally:
        _engine = None


app = FastAPI(title="wut-confusion-engine", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Reads env directly rather than importing config — see the module docstring."""
    engine = _engine
    return {
        "ok": engine is not None,
        "device": getattr(engine, "device", None),
        "whisper": os.environ.get("WHISPER_MODEL", "configured-at-startup"),
        "space_c": os.environ.get("ENABLE_SPACE_C", "1") == "1",
        "judge": os.environ.get("JUDGE_BACKEND", "local"),
        # True once an utterance came back with uniform word timings, meaning Whisper's word
        # alignment is failing and the pace/cognitive-load signal is unavailable. Surfaced here so
        # the box can be diagnosed from /health instead of needing shell access to read the logs.
        "pace_degraded": bool(getattr(engine, "_warned_pace", False)),
    }


@app.post("/analyze", response_model=ChunkAnalysis)
async def analyze(
    chunk_id: int = Form(0),
    history: str = Form("[]"),               # JSON list of prior transcripts (stateless context)
    enable_space_c: bool | None = Form(None),
    overall_topic: str = Form(""),           # ---- curriculum grounding: what's being taught, the
    curriculum_context: str = Form(""),      #      objective + notes + material, and the concepts
    key_concepts: str = Form("[]"),          #      already covered. Drives Space C + the question.
    audio: UploadFile = File(...),
) -> ChunkAnalysis:
    """Analyze one spoken utterance -> ChunkAnalysis (backend contract)."""
    engine = _engine
    if engine is None:
        # Startup takes minutes. 503 tells the backend to degrade rather than treat this as a bug.
        raise HTTPException(503, "confusion engine is not ready")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "audio file is empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio chunk exceeds 15 MB")

    # faster-whisper picks its demuxer off the extension, so keep the caller's — but only from a
    # known set, since it lands in a filesystem path.
    suffix = Path(audio.filename or "chunk.wav").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        async with _analysis_lock:
            # analyze() is long, blocking and GPU-bound; run_in_threadpool keeps the event loop
            # free to answer /health while it runs.
            return await run_in_threadpool(
                engine.analyze,
                temp_path,
                _json_string_list(history),
                chunk_id,
                enable_space_c,
                overall_topic.strip(),
                curriculum_context.strip(),
                _json_string_list(key_concepts),
            )
    finally:
        # The box has a 20 GB quota — a leaked temp WAV per request fills it.
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
