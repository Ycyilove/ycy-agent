"""检索服务：search / bm25_queries_for / should_use_record / record_search_docs。

B14-3 从 rag/service.py 原样搬（search 264 行 + 3 个辅助）。
SearchService.__init__ 做引用转发（self.xxx = rag.xxx）。
方法名映射（对齐 facade）：
    _bm25_queries_for -> bm25_queries_for
    _should_use_record -> should_use_record
    _record_search -> record_search_docs
"""
from __future__ import annotations

import json
import logging
import re
import time as _time
from typing import Dict, List, Optional

from rag.bm25_store import BM25Store, rrf_fuse
from rag.text_utils import clean_cn_spaces, clean_heading
from rag.search.record_search import record_search

logger = logging.getLogger(__name__)


class SearchService:
    """检索服务。self.xxx 是 rag.xxx 的引用转发。"""

    def __init__(self, rag):
        self.rag = rag
        self.vector_store = rag.vector_store
        self.bm25_store = rag.bm25_store
        self.record_store = rag.record_store
        self.resource_manager = rag.resource_manager
        self._extractor = rag._extractor
        self._schema = rag._schema

        # B14-3 修：search 方法体还调旧名（_should_use_record 等），加别名
        self._bm25_queries_for = self.bm25_queries_for
        self._should_use_record = self.should_use_record
        self._record_search = self.record_search_docs

    def search(
        self,
        query: str,
        top_k: int = 8,
        rewrite: bool = True,
        use_bm25: bool = True,
        mode: str = "auto",
    ) -> List[Dict]:
        """向量检索 + BM25 混合检索 + 可选 query 改写。

        mode:
            - "flat"   ：原 chunk 级混合检索
            - "record" ：记录级检索（表格/列表优先）
            - "auto"   ：有记录且命中记录 BM25 时走 record，否则 flat
        """
        # NFKC：把兼容字符（如 ⼯）映射回标准汉字，避免分词/BM25 对不上
        try:
            import unicodedata
            query = unicodedata.normalize("NFKC", query or "")
        except Exception:
            pass
        query = clean_cn_spaces(query)

        fallback_warning = None
        if mode == "record" or (mode == "auto" and self._should_use_record(query)):
            record_docs = self._record_search(query, top_k)
            if record_docs:
                return record_docs
            logger.info("[dsh][rag] record 模式空召回，fallback 到 flat 检索")
            fallback_warning = {
                "type": "record_empty_fallback",
                "message": "record 模式未命中结构化记录，已降级为 flat 检索",
            }

        # 1. query 改写（带超时降级）
        effective_query = query
        if rewrite:
            try:
                from backend.config import (
                    RAG_QUERY_REWRITE_ENABLED,
                    RAG_QUERY_REWRITE_TIMEOUT,
                )
                if RAG_QUERY_REWRITE_ENABLED:
                    from .query_rewriter import get_query_rewriter
                    import concurrent.futures

                    # 不能用 with：__exit__ 会 shutdown(wait=True)，
                    # 超时后仍会阻塞到改写跑完，超时形同虚设。
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = ex.submit(
                            get_query_rewriter().rewrite, query
                        )
                        effective_query = future.result(
                            timeout=RAG_QUERY_REWRITE_TIMEOUT
                        )
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            "[dsh][rag] query 改写超时(%ss)，用原 query",
                            RAG_QUERY_REWRITE_TIMEOUT,
                        )
                        effective_query = query
                    finally:
                        ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.exception("[dsh][rag] query 改写失败，用原 query")

        import time as _time
        _t_search0 = _time.perf_counter()

        pool_size = max(top_k * 3, top_k + 10)

        # 2. 向量检索：原 query 单路
        vector_results = []
        _t_vec0 = _time.perf_counter()
        try:
            vector_results = self.vector_store.search(query, pool_size)
            if not vector_results and self.vector_store.index is not None:
                logger.warning(
                    "[dsh][rag] 向量检索返回空，可能是索引维度与模型不匹配；"
                    "请检查 vector_store 或重建索引"
                )
        except Exception:
            logger.exception("[dsh][rag] 向量检索失败，降级为纯 BM25")
            vector_results = []
        _vec_ms = (_time.perf_counter() - _t_vec0) * 1000

        # 3. BM25 检索：用实体 query（避免高频意图词稀释）
        bm25_results: List[Dict] = []
        _t_bm0 = _time.perf_counter()
        if use_bm25:
            try:
                from backend.config import RAG_BM25_ENABLED, RAG_RRF_K
                if RAG_BM25_ENABLED:
                    if not self.bm25_store.is_built():
                        logger.info("[dsh][rag] BM25 索引未构建，惰性构建")
                        self.bm25_store.build(self.vector_store.metadata)

                    def _provider():
                        return self.vector_store.metadata

                    queries = self._bm25_queries_for(query)
                    if effective_query != query:
                        queries.extend(self._bm25_queries_for(effective_query))

                    seen = set()
                    uniq = []
                    for q in queries:
                        if q and q not in seen:
                            seen.add(q)
                            uniq.append(q)

                    lists = [
                        self.bm25_store.search(q, pool_size, _provider)
                        for q in uniq
                    ]
                    lists = [r for r in lists if r]
                    if len(lists) == 1:
                        bm25_results = lists[0]
                    elif len(lists) > 1:
                        bm25_results = rrf_fuse(
                            *lists, k=RAG_RRF_K, limit=pool_size,
                        )
            except Exception:
                logger.exception("[dsh][rag] BM25 检索失败，仅用向量结果")
        _bm_ms = (_time.perf_counter() - _t_bm0) * 1000

        # 4. 融合
        _t_fuse0 = _time.perf_counter()
        if bm25_results:
            from backend.config import RAG_RRF_K
            fused = rrf_fuse(
                vector_results,
                bm25_results,
                k=RAG_RRF_K,
                limit=top_k,
                weights=[1.0, 2.0],
            )
        else:
            fused = vector_results[:top_k]
        logger.info(
            "[dsh][rag] vector=%d bm25=%d fused=%d",
            len(vector_results), len(bm25_results), len(fused),
        )
        _fuse_ms = (_time.perf_counter() - _t_fuse0) * 1000

        _total_ms = (_time.perf_counter() - _t_search0) * 1000
        logger.info(
            "[dsh][rag] search timing: vector=%.0fms bm25=%.0fms fuse=%.0fms "
            "total=%.0fms (vec=%d, bm25=%d, fused=%d)",
            _vec_ms, _bm_ms, _fuse_ms, _total_ms,
            len(vector_results), len(bm25_results), len(fused),
        )

        # ── 返回前：从 vector_store.metadata 按 chunk_id 回填完整 metadata ──
        # BM25 缓存版 metadata 只有 text/filename/chunk_id，
        # headings/page_numbers 等需要从 vector_store.metadata 取回。
        chunk_meta_map: Dict[str, Dict] = {}
        try:
            for m in self.vector_store.metadata:
                cid = m.get("chunk_id")
                if cid:
                    chunk_meta_map[cid] = m
        except Exception:
            logger.exception("[dsh][rag] build chunk_meta_map failed")

        enriched = []
        for item in fused:
            meta = item.get("metadata", {}) or {}
            cid = meta.get("chunk_id")
            full_meta = chunk_meta_map.get(cid) if cid else None
            if full_meta:
                # 用完整 metadata 覆盖，保留 BM25 的 score / rrf_score
                merged_meta = dict(full_meta)
                merged_meta.update({
                    k: v for k, v in meta.items()
                    if k not in merged_meta
                })
                item["metadata"] = merged_meta
                meta = merged_meta

            headings = meta.get("headings", []) or []
            indexed_text = item.get("text", "") or ""
            original_text = meta.get("original_text") or ""

            # 优先返回 original_text（索引原文，含所有实体）；
            # enriched（indexed_text）作为补充上下文。
            # 但若 original_text 为空（旧索引），退回 indexed_text。
            if original_text:
                if indexed_text and original_text not in indexed_text:
                    # enriched 和原文不同（如 contextual 摘要）→ 双份
                    indexed_text = (
                        original_text + "\n\n【上下文】\n" + indexed_text
                    )
                else:
                    indexed_text = original_text
            # original_text 为空时保持 indexed_text

            if headings and not indexed_text.startswith(headings[0]):
                prefix = " > ".join(headings) + "\n"
                item["text"] = prefix + indexed_text
            else:
                item["text"] = indexed_text

            item["headings"] = headings
            item["heading_levels"] = meta.get("heading_levels", []) or []
            item["page_numbers"] = meta.get("page_numbers", []) or []
            item["has_contextual"] = meta.get("has_contextual", False)
            enriched.append(item)

        if fallback_warning:
            for item in enriched:
                item.setdefault("warnings", []).append(fallback_warning)
        # 把分段耗时挂到第一条结果的 metadata 上（不污染每条）
        enriched = self.resource_manager.enrich_search_results(enriched)
        if enriched:
            enriched[0].setdefault("metadata", {})["timing_ms"] = {
                "vector": round(_vec_ms, 1),
                "bm25": round(_bm_ms, 1),
                "fuse": round(_fuse_ms, 1),
                "total": round(_total_ms, 1),
            }

        # ── 实体兜底：向量对生僻实体不敏感，BM25 召回的含实体 chunk
        #    在 RRF 融合后可能排后。如果 top_k 里没有含 query 实体的 chunk，
        #    把 BM25 的含实体 chunk 追加到最前。 ──
        ents = [m.surface for m in self._extractor.extract(query)]
        if ents:
            has_entity = any(
                any(e in (r.get("text") or "") for e in ents)
                for r in enriched[:top_k]
            )
            if not has_entity:
                for item in bm25_results:
                    text = item.get("text") or ""
                    if not any(e in text for e in ents):
                        continue
                    cid = (item.get("metadata") or {}).get("chunk_id")
                    full_meta = chunk_meta_map.get(cid) if cid else None
                    if not full_meta:
                        continue
                    new_item = dict(item)
                    new_item["metadata"] = full_meta
                    headings = full_meta.get("headings") or []
                    original_text = full_meta.get("original_text") or ""
                    indexed_text = original_text or text
                    if headings and not indexed_text.startswith(headings[0]):
                        indexed_text = " > ".join(headings) + "\n" + indexed_text
                    new_item["text"] = indexed_text
                    new_item["headings"] = headings
                    new_item["heading_levels"] = full_meta.get("heading_levels") or []
                    new_item["page_numbers"] = full_meta.get("page_numbers") or []
                    new_item["has_contextual"] = full_meta.get("has_contextual", False)
                    new_item["score"] = new_item.get("score", 0.0) or 0.0
                    enriched.insert(0, new_item)
                    logger.info(
                        "[dsh][rag] 实体兜底：追加含 %r 的 chunk cid=%s",
                        ents, cid,
                    )
                    break

        enriched = enriched[:top_k]
        return enriched


    def bm25_queries_for(self, text: str) -> List[str]:
        """返回 [实体串, 全句]，两路分别检索后 RRF。"""
        try:
            import unicodedata
            text = unicodedata.normalize("NFKC", text or "")
        except Exception:
            pass
        ents = [m.surface for m in self._extractor.extract(text)]
        queries: List[str] = []
        if ents:
            queries.append(" ".join(ents))
        queries.append(text)
        return queries

    _RELATION_HINTS = ("属于", "队友", "成员", "团队", "关联", "同队", "所在", "获得")


    def should_use_record(self, query: str) -> bool:
        if not self.record_store.stats().get("total_records"):
            return False
        if any(h in query for h in self._RELATION_HINTS):
            return True
        rec_hits = self.record_store.search(query, top_k=1)
        if not rec_hits:
            return False
        # ── 额外判断：query 里的实体必须出现在 record 的文本里 ──
        # 避免 BM25 单字匹配（如"天梯赛"匹配到含"天"的财务表格）
        ents = [m.surface for m in self._extractor.extract(query)]
        if not ents:
            return False  # 无实体 → 不走 record
        top = rec_hits[0]
        record_text = ""
        if isinstance(top, dict):
            record_text = top.get("text") or ""
            if not record_text and top.get("record") is not None:
                rec = top["record"]
                record_text = getattr(rec, "to_index_text", lambda: "")()
        # 实体必须命中至少一个
        return any(e in record_text for e in ents)


    def record_search_docs(self, query: str, top_k: int) -> List[Dict]:
        """记录级检索 → 转成与 flat search 同构的 dict 列表。"""
        result = record_search(
            query, self.record_store, self._extractor,
            top_k=top_k, schema=self._schema,
        )
        docs: List[Dict] = []
        for record in result["records"]:
            metadata = {
                "chunk_id": record.record_id,
                "record_id": record.record_id,
                "source_filename": record.source_filename,
                "headings": record.headings,
                "page_numbers": record.page_numbers,
                "kind": "record",
            }
            item = {
                "text": record.to_index_text(),
                "filename": record.source_filename,
                "score": 0.0,
                "metadata": metadata,
            }
            enriched = self.resource_manager._enrich_one(item, metadata)
            enriched["paths"] = result["paths"]
            enriched["warnings"] = result["warnings"]
            docs.append(enriched)
        return docs
    
