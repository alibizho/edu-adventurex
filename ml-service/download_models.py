"""Pull every model to ONE place (MODELS_DIR) on the persistent volume, so there is no duplicate
copy in ~/.cache/huggingface eating the 20 GB quota. Run once after redeeming the HyperAI box.

  python download_models.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # fast in CN
os.environ.setdefault("HF_HOME", os.path.abspath("./.hf_cache"))

from huggingface_hub import snapshot_download

import config as C

# repo_id -> local dir. Whisper is pulled by faster-whisper on first run (CT2 format), not here.
REPOS = {
    C.WAV2VEC_REPO: C.WAV2VEC_DIR,   # Space A audio encoder (1024-dim)
    C.DEBERTA_REPO: C.DEBERTA_DIR,   # Space A/B text encoder (768-dim)
}
if C.ENABLE_SPACE_C:
    REPOS[C.BGE_REPO] = C.BGE_DIR    # Space C retriever (multilingual)


def main():
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    for repo, dest in REPOS.items():
        print(f"↓ {repo} -> {dest}")
        snapshot_download(repo_id=repo, local_dir=dest)
    print("\nDone. Now:")
    print("  - faster-whisper will fetch", C.WHISPER_MODEL, "on first server start (int8, ~0.8 GB).")
    if C.JUDGE_BACKEND == "local":
        print("  - the Qwen judge downloads on first /analyze (4-bit ~1 GB). Set JUDGE_BACKEND=api to skip it.")
    print("  - rm -rf ./.hf_cache after downloads to reclaim space.")


if __name__ == "__main__":
    main()
