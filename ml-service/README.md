# ml-service — Confusion Engine (Instrument B/C)

The GPU service. Refactor of the ML teammate's `audi.ipynb` (the tri-modal confusion detector) into a deployable FastAPI app sized to fit **HyperAI RTX 5090 (32 GB VRAM / 20 GB
storage)**. The main backend calls it per recorded utterance and gets back a `ChunkAnalysis`.

## What it does

Per spoken utterance, three "spaces":
- **A · audio ↔ text** — hesitation / recall failure (Wav2Vec2 + mDeBERTa → trained alignment brain)
- **B · text ↔ text** — self-contradiction vs earlier speech (judge LLM)
- **C · text ↔ knowledge** — factual error vs a Wikipedia vector DB (BGE-M3 + Chroma, judge LLM)

Output = per-chunk `confidence ∈ [0,1]` (HIGH = clear) + typed anomalies
(`recall_failure` / `logic_error` / `factual_error`), matching `backend/app/schemas.py`.

## Model budget — what changed and why

| Component | Notebook | Here | How |
|---|---|---:|---|
| Whisper ASR | large-v3 · fp16 (~3.1 GB) | **large-v3-turbo · int8 (~0.8 GB)** | keeps zh + word timestamps, ~5–8× faster |
| Wav2Vec2-XLSR | 300M fp32 (~1.3 GB) | 300M **fp16 (~0.65 GB)** | tied to the trained brain, can't swap |
| mDeBERTa-v3 | base fp32 (~0.56 GB) | base fp16 (~0.3 GB) | small, kept |
| BGE-M3 (Space C) | fp16 (~2.3 GB) | fp16 (~2.3 GB), **optional** | `ENABLE_SPACE_C=0` drops it entirely |
| Qwen judge | 1.5B fp16 (~3.1 GB) | **1.5B 4-bit (~1.0 GB)** or **API (0 GB)** | `JUDGE_BACKEND=api` hosts no LLM here |
| Vector DB | 350k chunks (~5 GB) | **~35k chunks (~0.5 GB)** | `INGEST_MAX_ROWS` 20k→3k, less overlap, index-bug fix |
| **Disk total** | **~15 GB (+cache dupes → 18–20+)** | **~7 GB** (or ~4 GB with Space C off) | fits the 20 GB quota with room |

Everything loads in `fp16`/`int8` under `torch.inference_mode()`, models pulled once to a single
`MODELS_DIR` (no `~/.cache/huggingface` duplicate).

## Deploy on HyperAI

Redeem code `HyperAI_AdventureX202607` → launch an **RTX 5090** instance (use the RTX PRO 6000 just
for the one-off DB build if you want headroom).

```bash
# 1. deps  (torch/torchaudio come with the base image — don't reinstall a mismatched build)
pip install -r requirements.txt

# 2. models -> single dir on the persistent volume
export HF_ENDPOINT=https://hf-mirror.com          # REQUIRED on HyperAI (CN); set in the shell,
                                                  # not just in-script, or it downloads from real HF
hf download facebook/wav2vec2-large-xlsr-53 --local-dir ./models/wav2vec2_xlsr
hf download microsoft/mdeberta-v3-base      --local-dir ./models/mdeberta_v3
hf download BAAI/bge-m3                      --local-dir ./models/bge_m3
# (or: bash fetch_models.sh — resumable ModelScope downloads, better on flaky links)
#    put the trained brain at ./saved_models/alignment_engine_STAGE1.pth  (from the teammate)

# 3. (Space C only) build the slim ground-truth DB once
python ingest.py

# 4. serve
uvicorn server:app --host 0.0.0.0 --port 8100
#    HyperAI: expose port 8100, then hit http://<instance>:8100/health
```

**Leaner variants** (edit `.env` / export before step 4):
- disturbance-only, no fact check: `ENABLE_SPACE_C=0` → skip step 3, ~4 GB, no BGE/Chroma.
- no local LLM: `JUDGE_BACKEND=api JUDGE_API_BASE=<glm> JUDGE_API_KEY=... JUDGE_API_MODEL=glm-4.6`
  → drops Qwen + bitsandbytes, judging goes to the API the backend already uses.

## Call it

```bash
curl -F audio=@utterance.wav -F chunk_id=4 -F 'history=["earlier transcript..."]' \
     http://localhost:8100/analyze
```
```json
{ "chunk_id": 4, "text": "so the packet goes to the, um... router?",
  "confidence": 0.38, "localized_target": "router",
  "anomalies": [ {"type": "recall_failure", "source": "space_a/audio-text", "score": 0.71,
                  "evidence": "hesitation on 'router'"} ],
  "detail": [ {"word": "so", "hesitation_zscore": -0.4, "is_anomaly": false }, ... ] }
```

## Backend integration

`backend/app/confusion/engine.py` currently ships a text-only heuristic mock. When this service is
up, point the backend at it (audio path) and keep the mock as the offline fallback — same
`ChunkAnalysis` contract, so nothing downstream changes.

## Files

```
config.py           env knobs (models dir, whisper size, space_c/judge flags, thresholds)
schemas.py          ChunkAnalysis / Anomaly (mirrors the backend contract)
alignment.py        AlignmentEngine (the trainable brain)
engine.py           ConfusionEngine — refactored analyze(): stateless, fp16, pluggable judge
server.py           FastAPI: POST /analyze, GET /health
fetch_models.sh     resumable ModelScope model fetch (run once per fresh volume)
start.sh            idempotent startup: restore deps/env, then serve on :8100
ingest.py           slim Space-C vector DB builder
```
