"""基于 embedding 的工具筛选器。

设计：
    - 复用 sentence-transformers 的 all-MiniLM-L6-v2（RAG 已在用）
    - 工具集不变时不重编向量
    - 模型不可用时返回全量，由上层兜底
    - 首次调用懒加载模型，之后缓存
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ToolShortlister:
    """goal → top-k 相关工具。

    相似度用归一化向量的点积（= cosine）。
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 15,
        min_score: float = 0.10,
        min_keep: int = 3,
    ):
        self._model_name = model_name
        self._top_k = top_k
        self._min_score = min_score
        self._min_keep = min_keep

        self._model = None
        self._model_ok = False
        self._tool_vectors: Dict[str, list] = {}
        self._cache_key: Optional[tuple] = None
        self._lock = threading.Lock()

    # ── 模型懒加载 ──

    def _ensure_model(self) -> bool:
        if self._model_ok:
            return True
        with self._lock:
            if self._model_ok:
                return True
            if self._model is None:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    self._model = HuggingFaceEmbeddings(
                        model_name=self._model_name,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                    logger.info(
                        "[dsh][shortlist] model loaded: %s", self._model_name
                    )
                except Exception:
                    logger.exception(
                        "[dsh][shortlist] model load failed; "
                        "shortlister disabled"
                    )
                    self._model_ok = False
                    return False
            self._model_ok = True
            return True

    def warmup(self) -> None:
        """显式预热：在 lifespan 里后台调用，避免首次任务卡顿。"""
        self._ensure_model()

    # ── 工具索引 ──

    def _ensure_index(self, tools: Dict[str, Any]) -> bool:
        if not self._ensure_model():
            return False

        key = tuple(sorted(tools.keys()))
        if key == self._cache_key:
            return True

        with self._lock:
            if key == self._cache_key:
                return True

            texts = []
            names = []
            for name, meta in tools.items():
                desc = getattr(meta, "description", "") or ""
                tags = " ".join(
                    str(t) for t in (getattr(meta, "intent_tags", None) or [])
                )
                texts.append(f"{name} {desc} {tags}".strip())
                names.append(name)

            try:
                vectors = self._model.embed_documents(texts)
            except Exception:
                logger.exception("[dsh][shortlist] embed_documents failed")
                return False

            self._tool_vectors = dict(zip(names, vectors))
            self._cache_key = key
            logger.info(
                "[dsh][shortlist] indexed %d tools", len(self._tool_vectors)
            )
            return True

    # ── 公开接口 ──

    def shortlist(
        self, goal: str, tools: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """返回筛选后的工具子集。失败时返回 None（调用方兜底）。"""
        if not goal or not tools:
            return None

        if not self._ensure_index(tools):
            return None

        try:
            goal_vec = self._model.embed_query(goal)
        except Exception:
            logger.exception("[dsh][shortlist] embed_query failed")
            return None

        # 余弦相似度（已归一化，点积即可）
        scores = []
        for name, vec in self._tool_vectors.items():
            if name not in tools:
                continue
            dot = 0.0
            for a, b in zip(goal_vec, vec):
                dot += a * b
            scores.append((name, dot))

        scores.sort(key=lambda x: -x[1])

        kept = [n for n, s in scores if s >= self._min_score][: self._top_k]
        if len(kept) < self._min_keep:
            logger.info(
                "[dsh][shortlist] kept=%d < min_keep=%d, fallback to full",
                len(kept), self._min_keep,
            )
            return None

        result = {n: tools[n] for n in kept if n in tools}
        logger.info(
            "[dsh][shortlist] %d/%d tools kept; top5=%r",
            len(result), len(tools),
            [(n, round(s, 3)) for n, s in scores[:5]],
        )
        return result