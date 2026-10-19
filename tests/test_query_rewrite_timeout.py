import time

import pytest


def test_rewrite_timeout_not_blocking(rag_service, monkeypatch):
    """验证 query 改写超时能真正取消，不会阻塞到改写跑完。

    rag_service 的 session 级 warmup 已经加载了 embedding 模型和
    BM25 索引，所以这里 elapsed 只反映 query 改写 + 检索本身的开销。

    注意：rag/service.py 里是 `from backend.config import RAG_QUERY_REWRITE_TIMEOUT`
    （函数内 import），每次从 backend.config 模块属性读取。所以必须
    monkeypatch.setattr 改模块属性，setenv 改环境变量无效（config 早已读过 .env）。
    """
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

    # 改写 sleep 5，若超时未生效，elapsed 会 >= 5s。
    # 阈值 2.0 留出检索本身的余量（通常 < 1s）。
    assert elapsed < 2.0, (
        f"超时未生效，实际耗时 {elapsed:.1f}s "
        f"（若接近 5s 说明 RAG_QUERY_REWRITE_TIMEOUT 还是 5，"
        f"检查 monkeypatch 是否 patch 了 backend.config 模块属性）"
    )
