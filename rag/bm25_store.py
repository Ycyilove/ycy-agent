"""BM25 关键词检索存储。

设计：
    - 从 FAISS 的 metadata.pkl 重建（不额外存文件）
    - 中文分词用 jieba
    - 结果与向量检索用 RRF 融合
    - 懒加载 + 缓存；FAISS 索引变化时通过 reset() 失效
    - 持久化：tokenized docs 落盘，命中缓存时跳过 jieba 分词
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 缓存目录：默认项目根下 data/bm25_cache，可用 RAG_BM25_CACHE_DIR 覆盖
_DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "bm25_cache")
)


def _cache_dir() -> str:
    path = os.getenv("RAG_BM25_CACHE_DIR", _DEFAULT_CACHE_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_text(text: str) -> str:
    """NFKC 规范化：把兼容字符（如 ⼯ U+2F47）映射回标准汉字（工 U+5DE5）。

    同时统一全角/半角、组合字符，避免同一语义在分词和 BM25 匹配时被当成不同词项。
    """
    if not text:
        return ""
    try:
        import unicodedata
        return unicodedata.normalize("NFKC", text)
    except Exception:
        return text


def _tokenize(text: str) -> List[str]:
    """中文分词。失败时降级为字符切分。"""
    if not text:
        return []
    text = _normalize_text(text)
    try:
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        return list(text)


def _docs_fingerprint(metadata: List[Dict]) -> str:
    """对 docs 的 text + filename 做指纹，用于缓存命名和失效判断。"""
    h = hashlib.md5()
    for d in metadata:
        h.update((d.get("text") or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update((d.get("filename") or "").encode("utf-8", "ignore"))
        h.update(b"\x01")
    return h.hexdigest()


def _cache_paths(fingerprint: str) -> Tuple[str, str]:
    base = _cache_dir()
    return (
        os.path.join(base, f"bm25_{fingerprint}.tokens.json"),
        os.path.join(base, f"bm25_{fingerprint}.meta.json"),
    )


def save_bm25_cache(metadata: List[Dict]) -> str:
    """把 tokenized docs + 精简 metadata 落盘。返回 fingerprint。"""
    fp = _docs_fingerprint(metadata)
    tok_path, meta_path = _cache_paths(fp)
    tokenized = [_tokenize(d.get("text", "")) for d in metadata]

    slim: List[Dict] = []
    for d in metadata:
        meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        slim.append({
            "text": d.get("text", ""),
            "filename": d.get("filename", "unknown"),
            "chunk_id": meta.get("chunk_id"),
        })

    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tokenized, f, ensure_ascii=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False)
    return fp


def load_bm25_cache(
    fingerprint: str,
) -> Optional[Tuple[List[List[str]], List[Dict]]]:
    """读缓存。缺失 / 损坏 / 长度不一致时返回 None。"""
    tok_path, meta_path = _cache_paths(fingerprint)
    if not (os.path.exists(tok_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(tok_path, "r", encoding="utf-8") as f:
            tokenized = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        if not isinstance(tokenized, list) or not isinstance(docs, list):
            return None
        if len(tokenized) != len(docs) or not docs:
            return None
        return tokenized, docs
    except Exception:
        logger.exception("[dsh][bm25] load cache failed: %s", fingerprint)
        return None


class BM25Store:
    """BM25 索引 + 检索。

    从 metadata 列表构建（不依赖 FAISS）。
    命中磁盘缓存时跳过 jieba 分词，直接构造 BM25Okapi。
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
        """从 metadata 构建 BM25 索引（优先命中磁盘缓存）。"""
        from rank_bm25 import BM25Okapi

        fp = _docs_fingerprint(metadata)
        cached = load_bm25_cache(fp)
        if cached:
            tokenized, docs = cached
            self._docs = docs
            self._bm25 = BM25Okapi(tokenized)
            logger.info(
                "[dsh][bm25] index loaded from cache: %d docs (fp=%s)",
                len(self._docs), fp[:8],
            )
            return

        self._docs = list(metadata)
        tokenized = [_tokenize(doc.get("text", "")) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        try:
            save_bm25_cache(metadata)
        except Exception:
            logger.exception("[dsh][bm25] save cache failed")
        logger.info(
            "[dsh][bm25] index built: %d docs (fp=%s)",
            len(self._docs), fp[:8],
        )

    def search(
        self,
        query: str,
        top_k: int,
        metadata_provider,
    ) -> List[Dict]:
        """BM25 检索。"""
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
        cid = (item.get("metadata") or {}).get("chunk_id")
        if cid:
            return ("chunk", cid)
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