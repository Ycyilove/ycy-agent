"""RAG 测试：搜索 / RRF 融合 / query 改写超时。

合并自：
    - test_rag_search.py
    - test_rrf_fuse.py
    - test_query_rewrite_timeout.py
"""

from __future__ import annotations

import os
import time

import pytest

from rag.bm25_store import rrf_fuse


# ═════════════════════════════════════════════════════════════
# 1. 搜索（原 test_rag_search.py）
# ═════════════════════════════════════════════════════════════

class TestRagSearch:
    @pytest.mark.skipif(
        not os.path.exists("vector_store/faiss.index"),
        reason="需要已建索引的知识库",
    )
    def test_search_returns_entity(self, rag_service):
        res = rag_service.search(
            "杨灿垣 天梯赛", top_k=5, rewrite=False, mode="flat",
        )
        for i, r in enumerate(res):
            print(f"--- {i} ---")
            print("filename:", r.get("filename"))
            print("text[:200]:", (r.get("text") or "")[:200])
        assert any("杨灿垣" in r["text"] for r in res)


# ═════════════════════════════════════════════════════════════
# 2. RRF 融合（原 test_rrf_fuse.py）
# ═════════════════════════════════════════════════════════════

class TestRrfFuse:
    def test_rrf_uses_chunk_id(self):
        a = [{"text": "x" * 300, "filename": "f", "metadata": {"chunk_id": "c1"}}]
        b = [{"text": "y" * 300, "filename": "f", "metadata": {"chunk_id": "c2"}}]
        fused = rrf_fuse(a, b, limit=10)
        assert len(fused) == 2

    def test_rrf_same_text_different_chunk(self):
        same = "z" * 300
        a = [{"text": same, "filename": "f", "metadata": {"chunk_id": "c1"}}]
        b = [{"text": same, "filename": "f", "metadata": {"chunk_id": "c2"}}]
        fused = rrf_fuse(a, b, limit=10)
        assert len(fused) == 2

    def test_rrf_same_chunk_merges(self):
        a = [{"text": "abc", "filename": "f", "metadata": {"chunk_id": "c1"}}]
        b = [{"text": "abc", "filename": "f", "metadata": {"chunk_id": "c1"}}]
        fused = rrf_fuse(a, b, limit=10)
        assert len(fused) == 1
        assert fused[0]["rrf_score"] > 0


# ═════════════════════════════════════════════════════════════
# 3. query 改写超时（原 test_query_rewrite_timeout.py）
# ═════════════════════════════════════════════════════════════

class TestQueryRewriteTimeout:
    def test_rewrite_timeout_not_blocking(self, rag_service, monkeypatch):
        """验证 query 改写超时能真正取消，不会阻塞到改写跑完。"""
        monkeypatch.setattr("backend.config.RAG_QUERY_REWRITE_TIMEOUT", 0.1)

        def slow_rewrite(*args, **kwargs):
            time.sleep(5)
            return "slow"

        import rag.query_rewriter as qr

        class _FakeRewriter:
            rewrite = staticmethod(slow_rewrite)

        monkeypatch.setattr(qr, "get_query_rewriter", lambda: _FakeRewriter())

        t0 = time.time()
        rag_service.search("test", top_k=3, rewrite=True)
        elapsed = time.time() - t0

        assert elapsed < 2.0, (
            f"超时未生效，实际耗时 {elapsed:.1f}s "
            f"（若接近 5s 说明 RAG_QUERY_REWRITE_TIMEOUT 还是 5，"
            f"检查 monkeypatch 是否 patch 了 backend.config 模块属性）"
        )