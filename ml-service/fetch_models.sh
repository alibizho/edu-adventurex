#!/usr/bin/env bash
# Fetch the encoder models into ./models on the persistent volume. Uses wget -c (resumable,
# single-stream) for the big weight files because the ModelScope CLI restarts-and-fails on big
# files over a flaky link (that's what kept corrupting model.bin). Idempotent: wget -c skips files
# that are already complete. Run once per fresh persistent volume, from the dir that holds ./models.
#
# Only the files transformers actually needs are pulled (skips tf_model.h5, flax, and mDeBERTa's
# 905 MB generator.bin) to respect the 20 GB quota.
set -euo pipefail
MS="https://www.modelscope.cn/models"

fetch() {  # fetch <repo> <dir> <file...>
  local repo="$1" dir="$2"; shift 2
  mkdir -p "$dir"
  for f in "$@"; do
    echo "[fetch] $repo -> $dir/$f"
    wget -c -O "$dir/$f" "$MS/$repo/resolve/master/$f"
  done
}

# Space A audio encoder. MUST match the encoder the alignment brain was trained on — confirm with
# whoever trained it before swapping this id, or Space A scores are meaningless.
fetch facebook/wav2vec2-large-xlsr-53 ./models/wav2vec2_xlsr \
  config.json preprocessor_config.json pytorch_model.bin

# Space A/B text encoder. No tokenizer.json in the repo -> the fast tokenizer is built from
# spm.model (needs sentencepiece + protobuf, already in requirements).
fetch microsoft/mdeberta-v3-base ./models/mdeberta_v3 \
  config.json tokenizer_config.json spm.model pytorch_model.bin

# Space C retriever (BGE-M3). Sentence-transformers layout has subdirs, so let the CLI lay it out,
# then wget -c the 2.1 GB weight file if the CLI left it incomplete.
if [ "${ENABLE_SPACE_C:-1}" = "1" ]; then
  modelscope download --model BAAI/bge-m3 --local_dir ./models/bge_m3 || true
  sz=$(stat -c%s ./models/bge_m3/pytorch_model.bin 2>/dev/null || echo 0)
  if [ "$sz" -lt 2200000000 ]; then
    echo "[fetch] bge pytorch_model.bin incomplete ($sz bytes) -> wget -c"
    wget -c -O ./models/bge_m3/pytorch_model.bin "$MS/BAAI/bge-m3/resolve/master/pytorch_model.bin"
  fi
fi

echo "[fetch] done:"
du -sh ./models/* 2>/dev/null || true
