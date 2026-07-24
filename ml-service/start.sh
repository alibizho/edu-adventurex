#!/usr/bin/env bash
# Idempotent startup for the confusion GPU service on EPHEMERAL HyperAI instances.
#
# Why this exists: every new HyperAI container resets pip packages AND env vars (that's why you
# keep hitting "No module named faster_whisper" and WHISPER_MODEL reverting). Only the persistent
# volume survives — so models stay put (./models, ./saved_models, ./multilingual_fact_db) but the
# python env and exports must be restored each boot. This script does both, then starts the server.
#
# ONE-TIME on the persistent volume, create ./.env.local (git-ignored) with your secret:
#     export JUDGE_API_KEY=sk-your-deepseek-key
# EVERY new instance, from the dir that contains ./models and server.py:
#     bash start.sh
set -euo pipefail

# 1. secrets / per-box overrides (lives on the persistent volume, not committed)
[ -f .env.local ] && source .env.local

# 2. env defaults for the local, fully-offline model layout.
#    All models live in ./models on the persistent volume -> block HF entirely so a stray hub id
#    fails fast with a clear error instead of hanging on the unreachable mirror.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WHISPER_MODEL="${WHISPER_MODEL:-./models/whisper}"   # local CT2 dir, not a hub alias
export ENABLE_SPACE_C="${ENABLE_SPACE_C:-1}"
# judge = DeepSeek API (OpenAI-compatible) -> hosts NO LLM on the box.
export JUDGE_BACKEND="${JUDGE_BACKEND:-api}"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.deepseek.com}"
export JUDGE_API_MODEL="${JUDGE_API_MODEL:-deepseek-v4-flash}"

# 3. restore python deps only if the container wiped them (fast no-op once present)
if ! python -c "import faster_whisper, fastapi, transformers, FlagEmbedding" 2>/dev/null; then
  echo "[start] fresh instance — installing python deps…"
  pip install -q faster-whisper transformers sentencepiece protobuf accelerate \
      FlagEmbedding chromadb huggingface_hub openai fastapi "uvicorn[standard]" \
      python-multipart pandas pyarrow numpy requests
  [ "${JUDGE_BACKEND}" = "local" ] && pip install -q bitsandbytes
fi

# 4. sanity: is the ASR model actually on disk?
if [ ! -s ./models/whisper/model.bin ]; then
  echo "[start] ERROR: ./models/whisper/model.bin missing/empty. Re-fetch it (resumable):" >&2
  echo "  wget -c -O ./models/whisper/model.bin \\" >&2
  echo "    https://www.modelscope.cn/models/pengzhendong/faster-whisper-large-v3-turbo/resolve/master/model.bin" >&2
  exit 1
fi
if [ "${JUDGE_BACKEND}" = "api" ] && [ -z "${JUDGE_API_KEY:-}" ]; then
  echo "[start] WARN: JUDGE_BACKEND=api but JUDGE_API_KEY is empty — Space B/C will fail per request." >&2
fi

echo "[start] whisper=${WHISPER_MODEL} judge=${JUDGE_BACKEND} space_c=${ENABLE_SPACE_C}"
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8100}"
