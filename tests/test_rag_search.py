import os
import pytest

@pytest.mark.skipif(
    not os.path.exists("vector_store/faiss.index"),
    reason="需要已建索引的知识库",
)
def test_search_returns_entity(rag_service):
    res = rag_service.search("杨灿垣 天梯赛", top_k=5, rewrite=False, mode="flat")
    for i, r in enumerate(res):
        print(f"--- {i} ---")
        print("filename:", r.get("filename"))
        print("text[:200]:", (r.get("text") or "")[:200])
    assert any("杨灿垣" in r["text"] for r in res)