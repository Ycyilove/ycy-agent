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
        """向量 + BM25 + RRF 混合检索。

        B6-c 起委托给 tools.semantics.index.strategies.hybrid.hybrid_search。
        保留原签名（frame + top_k），调用方零改动。
        """
        from tools.semantics.index.strategies.hybrid import (
            hybrid_search as _impl,
        )

        # frame 可能是 IntentFrame 对象，取其 raw / domain
        query = getattr(frame, "raw", "") or ""
        domain = getattr(frame, "domain", None)
        if domain == "unknown":
            domain = None

        raw = _impl(
            query,
            _INDEX_DIR,
            domain=domain,
            top_k=top_k,
        )

        # 转回 RankedCandidate（下游 _pipeline 期望这个类型）
        from .types import RankedCandidate
        out: List[RankedCandidate] = []
        for item in raw:
            out.append(
                RankedCandidate(
                    server_name=item["server_name"],
                    top_tools=item.get("top_tools") or [],
                    score=item.get("score", 0.0),
                    intent_features=item.get("intent_features") or {},
                    registry_entry=item.get("registry_entry") or {},
                )
            )
        return out
