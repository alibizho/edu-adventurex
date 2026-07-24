"""FastAPI wrapper around ConfusionEngine. Deploy this on the HyperAI GPU box; the main backend
calls POST /analyze per recorded utterance.

Run:  uvicorn server:app --host 0.0.0.0 --port 8100
"""
import json
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile

import config as C
from engine import ConfusionEngine
from schemas import AnalyzeRequest

engine: ConfusionEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ConfusionEngine()          # load all models once at startup
    yield
    engine = None


app = FastAPI(title="Confusion Engine (Instrument B/C)", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    import torch
    return {
        "ok": engine is not None,
        "device": C.DEVICE,
        "whisper": C.WHISPER_MODEL,
        "space_c": C.ENABLE_SPACE_C,
        "judge": C.JUDGE_BACKEND,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if C.DEVICE == "cuda" else 0,
    }


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    session_id: str = Form("default"),
    chunk_id: int = Form(0),
    history: str = Form("[]"),               # JSON list of prior transcripts (stateless context)
    enable_space_c: bool | None = Form(None),
) -> dict:
    """Analyze one spoken utterance -> ChunkAnalysis (backend contract)."""
    assert engine is not None, "engine not loaded"
    hist = json.loads(history) if history else []

    suffix = os.path.splitext(audio.filename or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        path = tmp.name
    try:
        result = engine.analyze(path, history=hist, chunk_id=chunk_id, enable_space_c=enable_space_c)
    finally:
        os.remove(path)
    return result.model_dump()
