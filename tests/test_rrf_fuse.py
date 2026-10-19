from rag.bm25_store import rrf_fuse


def test_rrf_uses_chunk_id():
    a = [{"text": "x" * 300, "filename": "f", "metadata": {"chunk_id": "c1"}}]
    b = [{"text": "y" * 300, "filename": "f", "metadata": {"chunk_id": "c2"}}]
    fused = rrf_fuse(a, b, limit=10)
    assert len(fused) == 2


def test_rrf_same_text_different_chunk():
    same = "z" * 300
    a = [{"text": same, "filename": "f", "metadata": {"chunk_id": "c1"}}]
    b = [{"text": same, "filename": "f", "metadata": {"chunk_id": "c2"}}]
    fused = rrf_fuse(a, b, limit=10)
    assert len(fused) == 2


def test_rrf_same_chunk_merges():
    a = [{"text": "abc", "filename": "f", "metadata": {"chunk_id": "c1"}}]
    b = [{"text": "abc", "filename": "f", "metadata": {"chunk_id": "c1"}}]
    fused = rrf_fuse(a, b, limit=10)
    assert len(fused) == 1
    assert fused[0]["rrf_score"] > 0
