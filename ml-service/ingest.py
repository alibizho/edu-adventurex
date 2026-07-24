"""Build the Space-C ground-truth vector DB. Slimmed hard for the 20 GB HyperAI quota.

vs the notebook:
  - INGEST_MAX_ROWS 20000 -> 3000, overlap 200 -> 80: ~350k chunks -> ~35k (~5 GB -> ~0.5 GB).
  - fixed the df.iterrows() index bug (index != row position) via reset_index.
  - fp16 embeddings kept; deletes the parquet after ingest to reclaim disk.

Run once (ideally on the RTX PRO 6000 for headroom):  python ingest.py
"""
import os

import config as C            # imported first: sets HF_ENDPOINT before any huggingface_hub import

import chromadb
import pandas as pd
import requests
from FlagEmbedding import BGEM3FlagModel

SHARDS = {
    "en": ("https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00000-of-00041.parquet", "facts_en"),
    "zh": ("https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00000-of-00006.parquet", "facts_zh"),
}


def ingest(bge, client, url, lang, collection_name, max_rows):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    local = f"./wiki_{lang}.parquet"
    print(f"[{lang}] downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    df = pd.read_parquet(local, columns=["text", "title"]).head(max_rows).reset_index(drop=True)
    print(f"[{lang}] embedding {len(df)} articles (chunk={C.CHUNK_SIZE}/{C.CHUNK_OVERLAP})")

    texts, ids = [], []
    for i, row in df.iterrows():                       # i is now 0..N-1 after reset_index
        full = f"Article: {row['title']}. Content: {row['text']}"
        if len(full) < 50:
            continue
        start = 0
        while start < len(full):
            chunk = full[start:start + C.CHUNK_SIZE]
            if len(chunk) > 50:
                texts.append(chunk)
                ids.append(f"{lang}_{i}_{start}")
                if len(texts) >= 64:
                    vecs = bge.encode(texts)["dense_vecs"]
                    collection.add(embeddings=vecs.tolist(), documents=texts, ids=ids)
                    texts, ids = [], []
            start += (C.CHUNK_SIZE - C.CHUNK_OVERLAP)

    if texts:
        vecs = bge.encode(texts)["dense_vecs"]
        collection.add(embeddings=vecs.tolist(), documents=texts, ids=ids)

    print(f"[{lang}] done: {collection.count()} chunks")
    if os.path.exists(local):
        os.remove(local)


def main():
    bge = BGEM3FlagModel(C.BGE_SRC, use_fp16=True)
    client = chromadb.PersistentClient(path=C.CHROMA_PATH)
    for lang, (url, name) in SHARDS.items():
        ingest(bge, client, url, lang, name, C.INGEST_MAX_ROWS)
    print("ground-truth DB built.")


if __name__ == "__main__":
    main()
