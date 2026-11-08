"""工具筛选器（B6-c 薄壳）：委托给 tools/semantics/index/strategies/shortlist。

保留原类签名 + 方法名（warmup / shortlist），调用方零改动。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ToolShortlister:
    """goal → top-k 工具。委托给 ShortlistStrategy。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 8,
        min_score: float = 0.10,
        min_keep: int = 3,
    ):
        from tools.semantics.index.strategies.shortlist import ShortlistStrategy

        self._impl = ShortlistStrategy(
            model_name=model_name,
            top_k=top_k,
            min_score=min_score,
            min_keep=min_keep,
        )

    def warmup(self) -> None:
        """预热：加载模型 + 编码一个占位串。失败静默。"""
        try:
            from tools.semantics.index.encoder import encode_texts
            encode_texts(["warmup"])
            logger.info("[shortlist] warmed up")
        except Exception:
            logger.exception("[shortlist] warmup failed")

    def shortlist(
        self, goal: str, tools: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        return self._impl.shortlist(goal, tools)


__all__ = ["ToolShortlister"]