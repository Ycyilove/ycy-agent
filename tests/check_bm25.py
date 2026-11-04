import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
from rag.vector_store import FAISSVectorStore


vs = FAISSVectorStore()
vs_cids = {m.get("chunk_id") for m in vs.metadata if m.get("chunk_id")}
cache_dir = "data/bm25_cache"

for fn in os.listdir(cache_dir):
    if not fn.endswith(".meta.json"):
        continue
    with open(os.path.join(cache_dir, fn), encoding="utf-8") as f:
        data = json.load(f)

    bm_cids = {d.get("chunk_id") for d in data if d.get("chunk_id")}
    matched = bm_cids & vs_cids
    print(f"{fn}: bm={len(bm_cids)} vs={len(vs_cids)} matched={len(matched)}")

    hits = [d for d in data if "杨灿垣" in (d.get("text") or "")]
    print(f"  含杨灿垣: {len(hits)}")
    for h in hits[:3]:
        cid = h.get("chunk_id")
        print(
            f"    cid={cid}, in_vs={cid in vs_cids}, "
            f"text[:60]={(h.get('text') or '')[:60]!r}"
        )