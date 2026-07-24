"""Config for the confusion GPU service. All knobs are env vars so nothing is hardcoded like
in the original notebook. Tuned for HyperAI (RTX 5090, 32 GB VRAM, 20 GB storage)."""
import os

# Force the CN mirror (HyperAI is in CN). setdefault was NOT enough: the base image often exports
# an EMPTY or huggingface.co HF_ENDPOINT, which setdefault won't override -> every download hits
# real HF and fails with LocalEntryNotFoundError. Override those two cases. Imported before any
# huggingface_hub import, so faster-whisper / transformers pick up the mirror. To use a different
# mirror, export HF_ENDPOINT=https://your-mirror (anything not containing 'huggingface.co').
_ep = os.environ.get("HF_ENDPOINT", "")
if (not _ep) or ("huggingface.co" in _ep):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

# --- where models live. ONE copy on the persistent volume; no HF-cache duplication. ---
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./multilingual_fact_db")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ASR: turbo is multilingual, keeps word timestamps, ~5-8x faster than large-v3.
#     Use the FULL CT2 repo id, not the bare "large-v3-turbo" alias — old faster-whisper builds
#     don't know the alias and treat it as a (nonexistent) repo id -> 404. If this repo ever 404s
#     on the mirror, fall back to the official Systran/faster-whisper-large-v3 (larger, non-turbo). ---
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
# NOTE: int8_float16 can crash faster-whisper's WORD-timestamp alignment (model.align() ->
# std::bad_alloc) on some ctranslate2 builds. float16 is the turbo model's native format (~1.6 GB,
# trivial on a 5090) and avoids the quantization path. Only drop to int8_float16 if VRAM is tight
# AND you've confirmed word_timestamps still work.
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16" if DEVICE == "cuda" else "int8")

# --- encoders (Space A / B / C). Loaded from *_DIR if present (offline, single copy), else pulled
#     from the *_REPO hub id on the fly — so ingest/serve work even before models are fetched. ---
WAV2VEC_DIR = os.path.join(MODELS_DIR, "wav2vec2_xlsr")
DEBERTA_DIR = os.path.join(MODELS_DIR, "mdeberta_v3")
BGE_DIR = os.path.join(MODELS_DIR, "bge_m3")

# canonical hub ids. Confirm WAV2VEC_REPO with whoever trained the alignment brain — the encoder
# features must match the one the brain was trained on, or Space A scores are meaningless.
WAV2VEC_REPO = os.environ.get("WAV2VEC_REPO", "facebook/wav2vec2-large-xlsr-53")
DEBERTA_REPO = os.environ.get("DEBERTA_REPO", "microsoft/mdeberta-v3-base")
BGE_REPO = os.environ.get("BGE_REPO", "BAAI/bge-m3")


def source(local_dir: str, repo_id: str) -> str:
    """Local dir if it exists (offline), else the hub id (auto-download via HF_ENDPOINT)."""
    return local_dir if os.path.isdir(local_dir) else repo_id


WAV2VEC_SRC = source(WAV2VEC_DIR, WAV2VEC_REPO)
DEBERTA_SRC = source(DEBERTA_DIR, DEBERTA_REPO)
BGE_SRC = source(BGE_DIR, BGE_REPO)

# Trained alignment brain — try STAGE1 then the final checkpoint.
ALIGN_WEIGHTS = [
    os.environ.get("ALIGN_WEIGHTS", "./saved_models/alignment_engine_STAGE1.pth"),
    "./saved_models/final_confusion_detector.pth",
]

# --- Space C (fact check) is now LLM-only via the judge — no BGE/Chroma/DB to load. Cheap to
#     leave on; set 0 to skip the fact-check judge call and run disturbance-only. ---
ENABLE_SPACE_C = os.environ.get("ENABLE_SPACE_C", "1") == "1"

# --- the contradiction judge. "local" = 4-bit Qwen on the box; "api" = offload to an
#     OpenAI-compatible endpoint (e.g. the main backend's GLM-4.6) and host NO LLM here. ---
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "local")  # "local" | "api"
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
QWEN_4BIT = os.environ.get("QWEN_4BIT", "1") == "1"       # 3.1 GB fp16 -> ~1 GB in 4-bit

# API-judge settings (only used when JUDGE_BACKEND == "api")
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "https://api.openai.com/v1")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_API_MODEL = os.environ.get("JUDGE_API_MODEL", "gpt-4o-mini")

# --- thresholds (were magic numbers in the notebook) ---
FACT_DISTANCE_THRESHOLD = float(os.environ.get("FACT_DISTANCE_THRESHOLD", "0.45"))
ZSCORE_ANOMALY = float(os.environ.get("ZSCORE_ANOMALY", "2.0"))

# --- ingestion (ground-truth DB build). Slimmed hard for the 20 GB quota. ---
INGEST_MAX_ROWS = int(os.environ.get("INGEST_MAX_ROWS", "3000"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
