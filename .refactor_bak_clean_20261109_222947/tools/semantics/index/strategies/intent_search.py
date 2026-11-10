"""intent_search 策略：goal → top-k server（按 server 聚合）。

原实现：tools/mcp_intent_index.py::search_intents
    - 编码 query → FAISS top_k → 阈值过滤 → 按 server 聚合
    - 每 server 保留最高分的 per_server_cap 条
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_SCORE = 0.5


def search_intents(
    query: str,
    index_dir: str,
    *,
    top_k: int = 20,
    per_server_cap: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
    frame: Any = None,
) -> List[Dict[str, Any]]:
    """语义检索工具索引，返回按 server 聚合的候选。

    Returns:
        [{server_name, top_score, matched_tools: [{tool_name, score, text}]}, ...]
        按 top_score 降序。
    """
    from tools.semantics.index.encoder import encode_query
    from tools.semantics.index.store import load_index, load_records

    records = load_records(index_dir)
    index = load_index(index_dir)
    if not records or index is None:
        return []

    try:
        qv = encode_query(query)
    except Exception:
        logger.exception("[intent_search] encode_query failed")
        return []

    import numpy as np
    qv = np.array([qv], dtype="float32")

    k = min(top_k * 3, len(records))
    dists, idxs = index.search(qv, k)

    hits: List[Dict[str, Any]] = []
    for d, i in zip(dists[0], idxs[0]):
        if i < 0 or i >= len(records):
            continue
        score = float(d)
        if score < min_score:
            continue
        r = records[i]
        hits.append({
            "tool_name": r.get("tool_name", ""),
            "server_name": r.get("server_name", ""),
            "text": r.get("text", ""),
            "score": score,
        })

    if not hits:
        logger.info(
            "[intent_search] no hit above %.2f for %r", min_score, query,
        )
        return []

    # 领域过滤（frame 有 domain 时降权）
    if frame is not None and getattr(frame, "domain", "unknown") != "unknown":
        _domain_kw = {
            "news": ("news", "新闻", "headline"),
            "sports": ("sport", "体育", "match", "score"),
            "file": ("file", "文件", "path", "directory"),
            "web": ("web", "网页", "http", "url", "fetch"),
            "code": ("code", "代码", "function", "script"),
        }.get(frame.domain, ())
        if _domain_kw:
            for h in hits:
                text_low = h["text"].lower()
                if not any(kw in text_low for kw in _domain_kw):
                    h["score"] *= 0.8

    by_server: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        srv = h.get("server_name") or ""
        if not srv:
            continue
        by_server.setdefault(srv, []).append(h)

    out: List[Dict[str, Any]] = []
    for srv, items in by_server.items():
        items.sort(key=lambda x: -x["score"])
        out.append({
            "server_name": srv,
            "top_score": items[0]["score"],
            "matched_tools": items[:per_server_cap],
        })
    out.sort(key=lambda x: -x["top_score"])
    return out