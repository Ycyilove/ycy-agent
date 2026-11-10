"""hybrid 策略：vector + BM25 + RRF。

原实现：tools/capability/index.py::CapabilityIndex.search
    - 先按 intent_features 过滤
    - 向量检索 + BM25 检索 → RRF 融合
    - 按 server 聚合

B6 保持原逻辑不变；B7 会把它接进 DiscoveryPipeline。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    index_dir: str,
    *,
    domain: Optional[str] = None,
    top_k: int = 10,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """向量 + BM25 + RRF 混合检索。

    Args:
        query: 查询文本
        index_dir: 索引目录
        domain: 领域过滤（与 records 的 intent_features.domain 匹配）
        top_k: 返回条数

    Returns:
        [{server_name, top_tools, score, intent_features, registry_entry}, ...]
    """
    from tools.semantics.index.encoder import encode_query
    from tools.semantics.index.store import load_index, load_records

    records = load_records(index_dir)
    index = load_index(index_dir)
    if not records or index is None:
        return []

    # 1. 确定性过滤：domain 匹配
    if domain:
        filtered = [
            r for r in records
            if r.get("intent_features", {}).get("domain") == domain
        ]
    else:
        filtered = records
    if not filtered:
        filtered = records

    # 2. 向量检索
    try:
        qv = encode_query(query)
    except Exception:
        logger.exception("[hybrid] encode_query failed")
        return []

    import numpy as np
    qv = np.array([qv], dtype="float32")
    k = min(top_k * 5, len(records))
    dists, idxs = index.search(qv, k)

    filtered_ids = {r.get("record_id") for r in filtered}
    vec_results: List[Dict[str, Any]] = []
    for d, i in zip(dists[0], idxs[0]):
        if i < 0 or i >= len(records):
            continue
        r = dict(records[i])
        if r.get("record_id") not in filtered_ids:
            continue
        r["score"] = float(d)
        r["metadata"] = {"chunk_id": r.get("record_id", f"vec-{i}")}
        vec_results.append(r)

    # 3. BM25
    bm25_results: List[Dict[str, Any]] = []
    try:
        from rag.bm25_store import BM25Store, rrf_fuse
        bm25 = BM25Store()
        bm25.build(filtered)
        bm25_results = bm25.search(query, top_k * 3, lambda: filtered)
        for i, r in enumerate(bm25_results):
            r["metadata"] = {"chunk_id": r.get("record_id", f"bm25-{i}")}
    except Exception:
        logger.exception("[hybrid] bm25 failed, use vector only")

    # 4. RRF 融合
    if bm25_results:
        from rag.bm25_store import rrf_fuse
        fused = rrf_fuse(vec_results, bm25_results, limit=top_k * 2)
    else:
        fused = vec_results[: top_k * 2]

    # 5. 按 server 聚合
    by_server: Dict[str, Dict[str, Any]] = {}
    for item in fused:
        srv = item.get("server_name")
        if not srv:
            continue
        if (
            srv not in by_server
            or item.get("rrf_score", 0) > by_server[srv].get("rrf_score", 0)
        ):
            by_server[srv] = item

    out: List[Dict[str, Any]] = []
    for srv, item in by_server.items():
        out.append({
            "server_name": srv,
            "top_tools": [item.get("tool_name", "")],
            "score": item.get("rrf_score", 0),
            "intent_features": item.get("intent_features", {}),
            "registry_entry": item.get("provenance", {}).get(
                "registry_entry", {}
            ),
        })
    out.sort(key=lambda c: -c["score"])
    return out