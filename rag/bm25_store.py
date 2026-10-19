"""BM25 关键词检索存储。

设计：
    - 从 FAISS 的 metadata.pkl 重建（不额外存文件）
    - 中文分词用 jieba
    - 结果与向量检索用 RRF 融合
    - 懒加载 + 缓存；FAISS 索引变化时通过 reset() 失效
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """中文分词。失败时降级为字符切分。"""
    if not text:
        return []
    try:
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        return list(text)


class BM25Store:
    """BM25 索引 + 检索。

    从 metadata 列表构建（不依赖 FAISS）。
    """

    def __init__(self):
        self._bm25 = None
        self._docs: List[Dict] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        """索引变化时调用，让下次查询重建。"""
        with self._lock:
            self._bm25 = None
            self._docs = []

    def is_built(self) -> bool:
        """索引是否已构建。"""
        with self._lock:
            return self._bm25 is not None and bool(self._docs)

    def build(self, metadata: List[Dict]) -> None:
        """公开的构建入口。入库后调用，避免每次查询重建。"""
        with self._lock:
            if self._bm25 is not None and bool(self._docs):
                return  # 已构建，跳过
            if not metadata:
                return
            self._build(metadata)

    def _build(self, metadata: List[Dict]) -> None:
        """从 metadata 构建 BM25 索引。"""
        from rank_bm25 import BM25Okapi

        self._docs = list(metadata)
        tokenized = [_tokenize(doc.get("text", "")) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "[dsh][bm25] index built: %d docs", len(self._docs)
        )

    def search(
        self,
        query: str,
        top_k: int,
        metadata_provider,
    ) -> List[Dict]:
        """BM25 检索。

        Args:
            query: 查询文本
            top_k: 返回数量
            metadata_provider: 返回当前 metadata 列表的 callable

        Returns:
            与 FAISS search 相同结构的列表
        """
        if not query or not query.strip():
            return []

        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError:
            logger.debug("[dsh][bm25] rank_bm25 未安装，跳过 BM25")
            return []

        with self._lock:
            if self._bm25 is None:
                metadata = metadata_provider()
                if not metadata:
                    return []
                self._build(metadata)

            if self._bm25 is None or not self._docs:
                return []

            query_tokens = _tokenize(query)
            if not query_tokens:
                return []

            scores = self._bm25.get_scores(query_tokens)
            # 取 top_k
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: -x[1])
            top = indexed[:top_k]

            results = []
            for idx, score in top:
                if idx >= len(self._docs):
                    continue
                doc = self._docs[idx]
                results.append({
                    "text": doc.get("text", ""),
                    "filename": doc.get("filename", "unknown"),
                    "score": float(score),
                    "metadata": {
                        k: v for k, v in doc.items() if k != "text"
                    },
                })
            return results

def rrf_fuse(*result_lists, k=60, limit=10, weights=None):
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights 数量必须与 result_lists 一致")

    def _key(item: dict):
        # 优先用 chunk_id：不同 chunk 的前 200 字可能相同
        # （表格重复行、模板文头），用 text[:200] 会误合并。
        cid = (item.get("metadata") or {}).get("chunk_id")
        if cid:
            return ("chunk", cid)
        # 兜底：filename + 完整 text 的 hash
        import hashlib
        text = item.get("text", "")
        digest = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
        return ("text", item.get("filename", ""), digest)

    scores: Dict[tuple, float] = {}
    items: Dict[tuple, dict] = {}
    for results, weight in zip(result_lists, weights):
        for rank, item in enumerate(results):
            key = _key(item)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
            if key not in items:
                items[key] = item

    fused = sorted(scores.items(), key=lambda x: -x[1])[:limit]
    out = []
    for key, rrf_score in fused:
        item = dict(items[key])
        item["rrf_score"] = rrf_score
        out.append(item)
    return out