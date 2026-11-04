"""验证 read_tool_result 的分页参数拆分。"""
import json
import os

import pytest

from core.task.tool_result_store import save_tool_result
from tools.tools_def.tool_result_reader import read_tool_result


@pytest.fixture
def saved_chunks(tmp_path, monkeypatch):
    """构造一个含 chunks 的落盘结果。"""
    # 把 _RESULT_DIR 指到 tmp_path
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

    payload = {
        "chunks": [
            {"filename": "a.pdf", "text": f"chunk-{i}"} for i in range(10)
        ]
    }
    result_id = save_tool_result(
        task_id="t1", step_index=1, tool_name="search_knowledge", payload=payload,
    )
    return result_id


@pytest.fixture
def saved_text(tmp_path, monkeypatch):
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

    payload = {"content": "A" * 10000}
    result_id = save_tool_result(
        task_id="t1", step_index=1, tool_name="read_text_file", payload=payload,
    )
    return result_id


def test_chunk_pagination(saved_chunks):
    r = read_tool_result(
        result_id=saved_chunks, task_id="t1",
        chunk_index=2, chunk_count=3,
    )
    assert r["success"] is True
    assert r["chunk_index"] == 2
    assert r["returned"] == 3
    assert r["chunks"][0]["text"] == "chunk-2"


def test_char_pagination(saved_text):
    r = read_tool_result(
        result_id=saved_text, task_id="t1",
        char_offset=100, char_limit=500,
    )
    assert r["success"] is True
    assert r["char_offset"] == 100
    assert r["returned_chars"] == 500


def test_legacy_params(saved_chunks):
    r = read_tool_result(
        result_id=saved_chunks, task_id="t1",
        start_index=1, limit=2,
    )
    assert r["success"] is True
    assert r["chunk_index"] == 1
    assert r["returned"] == 2