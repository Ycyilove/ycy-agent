"""shortlist 策略：goal → top-k 工具（保持原 ToolShortlister 语义）。

原实现：tools/orchestration/tool_shortlister.py
    - 工具集不变时缓存向量
    - 分数阈值 min_score + 保底 min_keep
    - 返回 dict 子集；失败返回 None（调用方兜底用全量）
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ShortlistStrategy:
    """带工具向量缓存。一个实例可复用多次 shortlist。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 8,
        min_score: float = 0.10,
        min_keep: int = 3,
    ):
        self._model_name = model_name
        self._top_k = top_k
        self._min_score = min_score
        self._min_keep = min_keep

        self._tool_vectors: Dict[str, List[float]] = {}
        self._cache_key: Optional[tuple] = None
        self._lock = threading.Lock()

    def _ensure_index(self, tools: Dict[str, Any]) -> bool:
        key = tuple(sorted(tools.keys()))
        if key == self._cache_key:
            return True

        with self._lock:
            if key == self._cache_key:
                return True

            from tools.semantics.index.encoder import build_tool_text, encode_texts

            names = list(tools.keys())
            texts = [build_tool_text(tools[n]) for n in names]
            try:
                vectors = encode_texts(texts)
            except Exception:
                logger.exception("[shortlist] encode_texts failed")
                return False

            self._tool_vectors = dict(zip(names, vectors))
            self._cache_key = key
            logger.info(
                "[shortlist] indexed %d tools", len(self._tool_vectors),
            )
            return True

    def shortlist(
        self,
        goal: str,
        tools: Dict[str, Any],
        *,
        weight_overrides: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """goal -> top-k 工具。

        weight_overrides: {tool_name: weight}，降权（<1.0）不剔除。
        top_k: 动态 top_k，None 时用 self._top_k。
        """
        if not goal or not tools:
            return None
        if not self._ensure_index(tools):
            return None

        from tools.semantics.index.encoder import encode_query
        try:
            goal_vec = encode_query(goal)
        except Exception:
            logger.exception("[shortlist] encode_query failed")
            return None

        _k = top_k if top_k is not None else self._top_k
        _w = weight_overrides or {}

        scores = []
        for name, vec in self._tool_vectors.items():
            if name not in tools:
                continue
            dot = 0.0
            for a, b in zip(goal_vec, vec):
                dot += a * b
            if name in _w:
                dot = dot * float(_w[name])
            scores.append((name, dot))

        scores.sort(key=lambda x: -x[1])
        kept = [n for n, s in scores if s >= self._min_score][: _k]
        if len(kept) < self._min_keep:
            logger.info(
                "[shortlist] kept=%d < min_keep=%d, fallback to full",
                len(kept), self._min_keep,
            )
            return None

        return {n: tools[n] for n in kept if n in tools}


_default_strategy: Optional[ShortlistStrategy] = None
_default_lock = threading.Lock()


def get_shortlist_strategy() -> ShortlistStrategy:
    global _default_strategy
    if _default_strategy is None:
        with _default_lock:
            if _default_strategy is None:
                _default_strategy = ShortlistStrategy()
    return _default_strategy


def shortlist(
    goal: str, tools: Dict[str, Any], **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    return get_shortlist_strategy().shortlist(goal, tools, **kwargs)