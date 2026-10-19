import os

import pytest


@pytest.mark.skipif(
    not os.path.exists("vector_store/faiss.index"),
    reason="需要已建索引的知识库",
)
def test_search_returns_entity(rag_service):
    res = rag_service.search("杨灿垣 天梯赛", top_k=5, rewrite=False)
    assert res, "检索结果为空"
    assert any("杨灿垣" in r["text"] for r in res), (
        "索引/返回不一致：enriched 命中但返回文本里没有实体"
    )
