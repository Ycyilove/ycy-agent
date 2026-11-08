"""离线意图索引 + 在线检索。

索引文件：data/capability_index/{faiss.index, records.json}
每条 record 带 intent_features（domain/action/freshness）+ provenance。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_INDEX_DIR = os.path.join(_PROJECT_ROOT, "data", "capability_index")
_INDEX_FILE = os.path.join(_INDEX_DIR, "faiss.index")
_RECORDS_FILE = os.path.join(_INDEX_DIR, "records.json")

_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


class CapabilityIndex:
    """离线索引检索。不调外部网络。"""

    def covers(self, frame, tools: Dict[str, Any]) -> bool:
        """现有工具是否覆盖 frame。用 tools 的 tags 判断。"""
        if frame.domain == "unknown":
            return False
        from tools.semantics.tags import get_intent_tags
        want_action = frame.action
        for name in tools:
            tags = get_intent_tags(name)
            if not tags:
                continue
            # action 直接匹配
            if want_action in tags:
                return True
            # 领域匹配（news/web 类只要有 network_read 就够）
            if frame.domain in ("news", "web") and "network_read" in tags:
                return True
            if frame.domain == "knowledge" and "knowledge_read" in tags:
                return True
        return False

    def search(self, frame, top_k: int = 10) -> List[Any]:
        """混合检索，返回按 server 聚合的 RankedCandidate。"""
        from .types import RankedCandidate
        if not (os.path.exists(_RECORDS_FILE) and os.path.exists(_INDEX_FILE)):
            return []

        with open(_RECORDS_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not records:
            return []

        # 确定性过滤：domain 匹配
        filtered = [
            r for r in records
            if r.get("intent_features", {}).get("domain") == frame.domain
        ]
        if not filtered:
            filtered = records

        # 向量 + BM25 + RRF
        try:
            import faiss
            import numpy as np
            from infra.embedding import get_encoder
            from rag.bm25_store import BM25Store, rrf_fuse
        except Exception:
            logger.exception("[capability] deps missing, fallback to bm25 only")
            return []

        enc = get_encoder(_EMBED_MODEL, local_files_only=True)
        qv = enc.encode([frame.raw], normalize_embeddings=True)[0]
        qv = np.array([qv], dtype="float32")
        idx = faiss.read_index(_INDEX_FILE)

        k = min(top_k * 5, len(records))
        dists, idxs = idx.search(qv, k)

        filtered_ids = {r.get("record_id") for r in filtered}
        vec_results: List[Dict] = []
        for d, i in zip(dists[0], idxs[0]):
            if i < 0 or i >= len(records):
                continue
            r = dict(records[i])
            if r.get("record_id") not in filtered_ids:
                continue
            r["score"] = float(d)
            r["metadata"] = {"chunk_id": r.get("record_id", f"vec-{i}")}
            vec_results.append(r)

        bm25 = BM25Store()
        bm25.build(filtered)
        bm25_results = bm25.search(frame.raw, top_k * 3, lambda: filtered)
        for i, r in enumerate(bm25_results):
            r["metadata"] = {
                "chunk_id": r.get("record_id", f"bm25-{i}")
            }

        fused = rrf_fuse(
            vec_results, bm25_results, limit=top_k * 2,
        ) if bm25_results else vec_results[: top_k * 2]

        # 聚合到 server
        by_server: Dict[str, Dict] = {}
        for item in fused:
            srv = item.get("server_name")
            if not srv:
                continue
            if srv not in by_server or (
                item.get("rrf_score", 0) > by_server[srv].get("rrf_score", 0)
            ):
                by_server[srv] = item

        out: List[RankedCandidate] = []
        for srv, item in by_server.items():
            out.append(
                RankedCandidate(
                    server_name=srv,
                    top_tools=[item.get("tool_name", "")],
                    score=item.get("rrf_score", 0),
                    intent_features=item.get("intent_features", {}),
                    registry_entry=item.get("provenance", {}).get(
                        "registry_entry", {}
                    ),
                )
            )
        out.sort(key=lambda c: -c.score)
        return out