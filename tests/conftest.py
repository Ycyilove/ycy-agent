import pytest


@pytest.fixture(scope="session")
def rag_service():
    """session 级 RAG 实例。

    warmup 一次：加载 embedding 模型 + 构建 BM25 索引。
    否则首次用 rag_service 的测试会把 ~100s 初始化开销
    记到自己的 elapsed 里，导致超时类测试误判。
    """
    from rag import LocalRAGService
    svc = LocalRAGService()
    try:
        svc.search("warmup", top_k=1, rewrite=False)
    except Exception:
        # warmup 失败不影响测试；让后续测试自己暴露真实错误
        pass
    return svc
