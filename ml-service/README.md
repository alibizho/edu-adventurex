# ml-service — Confusion Engine (Instrument B/C)

The GPU service. Refactor of the ML teammate's `audi.ipynb` (the tri-modal confusion detector) into a deployable FastAPI app sized to fit **HyperAI RTX 5090 (32 GB VRAM / 20 GB
storage)**. The main backend calls it per recorded utterance and gets back a `ChunkAnalysis`.

## What it does

Per spoken utterance, three "spaces":
- **A · audio ↔ text** — hesitation / recall failure (Wav2Vec2 + mDeBERTa → trained alignment brain)
- **B · text ↔ text** — self-contradiction vs earlier speech (judge LLM)
- **C · text ↔ knowledge** — factual error vs the class being taught (judge LLM)

Output = per-chunk `confidence ∈ [0,1]` (HIGH = clear) + typed anomalies
(`recall_failure` / `logic_error` / `factual_error` / `cognitive_load` / `fluency_issue` /
`off_topic` / `beyond`), matching `backend/app/schemas.py`.

**Curriculum grounding.** When the caller sends `overall_topic`, `curriculum_context` (the class
objective + teacher's notes + source material) and `key_concepts`, Space C grades against *that*
instead of the judge's general knowledge. Only then can it separate three things a bare fact-check
conflates: wrong (`factual_error`), off the syllabus (`off_topic`), and correct-but-past-the-syllabus
(`beyond` — which comes back as `curriculum_update.added_concepts` so the backend can grow the class
rather than penalise the learner).

**Cross-modal fusion.** If the words are right but a large share of them cost visible effort to say
(`FLUENCY_LOAD_RATIO_THRESHOLD`), that's a `fluency_issue` — recitation without understanding.
Neither space sees it alone; it only exists in the disagreement between them.

**The question is written here.** With the anomaly and the exact offending word still in hand, the
judge writes the AI student's interruption and returns it as `student_question`. The backend relays
it, and only falls back to its own LLM question generator when this is absent.

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

`POST /analyze`, multipart. `audio` is required; everything else has a default.

| Field | Default | Purpose |
|---|---|---|
| `audio` | — | one paused utterance, ≤ 15 MB (`.wav`/`.mp3`/`.m4a`/`.webm`/`.ogg`) |
| `chunk_id` | `0` | aligns with the backend's segment spine |
| `history` | `[]` | JSON array of prior transcripts — Space B context |
| `enable_space_c` | server default | per-call fact-check override |
| `overall_topic` | `""` | the topic being taught |
| `curriculum_context` | `""` | class objective + teacher's notes + source material |
| `key_concepts` | `[]` | JSON array of concepts already covered |
| `focus_target` | `""` | the concept the learner keeps stumbling over, picked by the backend's struggle ledger. Keeps the student's question on one thread across utterances — this service holds no state between calls. |

```bash
curl -F audio=@utterance.wav -F chunk_id=4 -F 'history=["earlier transcript..."]' \
     -F overall_topic='Computer Networks' \
     -F curriculum_context='How a packet is forwarded hop by hop...' \
     -F 'key_concepts=["routing table","hop"]' \
     -F focus_target='routing table' \
     http://localhost:8100/analyze
```
```json
{ "chunk_id": 4, "text": "so the packet goes to the, um... router?",
  "confidence": 0.38, "localized_target": "router",
  "anomalies": [ {"type": "recall_failure", "source": "space_a/audio-text", "score": 0.71,
                  "evidence": "hesitation on 'router'"} ],
  "detail": [ {"word": "so", "hesitation_zscore": -0.4, "is_anomaly": false }, ... ],
  "student_question": { "question_text": "Wait, I thought the router was different? Can you clarify?",
                        "target_concept": "router", "anomaly_type": "recall_failure" },
  "curriculum_update": null }
```

`503` until the models finish loading (minutes), `400` on empty audio, `413` over 15 MB. Requests
are serialized behind a lock — the models are loaded once and aren't re-entrant.

## Backend integration

`backend/app/confusion/engine.py` ships a text-only heuristic mock. When this service is up, point
the backend at it with `ML_SERVICE_URL` and keep the mock as the offline fallback — same
`ChunkAnalysis` contract, so nothing downstream changes. `backend/app/confusion/client.py` reports
whether a given call actually reached this service, which is how the UI shows an honest "voice
teaching unavailable" instead of a silently neutral score.

The two services share no package, so `backend/tests/test_ml_service_contract.py` asserts
`schemas.py` here against `app/schemas.py` there, field for field. It runs on a plain CPU checkout
with no torch — that's why `server.py` imports `engine` lazily. Change one schema, change both.

## Files

```
config.py           env knobs (models dir, whisper size, space_c/judge flags, thresholds)
schemas.py          ChunkAnalysis / Anomaly / StudentQuestion (mirrors the backend contract)
alignment.py        AlignmentEngine (the trainable brain)
engine.py           ConfusionEngine — stateless analyze(): fp16, pluggable judge, curriculum-grounded
server.py           FastAPI: POST /analyze, GET /health (imports engine lazily — see the docstring)
fetch_models.sh     resumable ModelScope model fetch (run once per fresh volume)
start.sh            idempotent startup: restore deps/env, then serve on :8100
ingest.py           slim Space-C vector DB builder
```
