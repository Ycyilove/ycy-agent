"""上下文预算 + 工具结果读取。

合并自：
    - test_context_exceeded.py
    - test_large_file_handle.py
    - test_tool_result_reader.py
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _inject_fake_config():
    """B2/B9 重构后 ModelScopeLLM.__init__ 从 ports.get_config() 读配置，测试要注入。"""
    from core import ports
    fake = MagicMock()
    fake.DEFAULT_MODEL = "fake-model"
    fake.MODEL_SOURCE = "ONLINE"
    fake.LOCAL_MODEL_CONFIG = {}
    fake.YUNDU_API_KEY = "fake-key"
    fake.YUNDU_BASE_URL = "http://fake"
    fake.MODESCOPE_BASE_URL = "http://fake"
    fake.MODESCOPE_API_KEY = "fake-key"
    fake.RAG_EMBEDDING_MODEL = "fake-embed"
    old = ports.get_config()
    ports.set_config(fake)
    try:
        yield
    finally:
        ports.set_config(old)


import json
from unittest.mock import MagicMock



from core.task.tool_result_store import (
    build_handle_summary,
    load_tool_result,
)
from core.llm.provider import ContextLengthExceeded, ModelScopeLLM
from tools.agent import ToolCallResult
from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState
from tools.tools_def.tool_result_reader import read_tool_result



# ─────────────────────────────────────────────────────────────
# 共享 fixture：把落盘目录指到 tmp_path，避免污染 data/
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_result_dir(tmp_path, monkeypatch):
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path / "tool_results"))


# ─────────────────────────────────────────────────────────────
# 共享夹具：大文件内容
# ─────────────────────────────────────────────────────────────

def _make_large_content(lines: int = 400) -> str:
    parts = []
    for i in range(lines):
        parts.append(f"LINE_{i:04d}: def func_{i}(x): return x + {i}  # marker_{i:04d}")
    return "\n".join(parts)


LARGE_CONTENT = _make_large_content(400)
assert len(LARGE_CONTENT) > 10000


def _make_edit_file_meta():
    meta = MagicMock()
    meta.description = "局部替换文件内容"
    meta.input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["file_path", "old_text", "new_text"],
    }
    return meta


# ═════════════════════════════════════════════════════════════
# 1. ContextLengthExceeded（原 test_context_exceeded.py）
# ═════════════════════════════════════════════════════════════

class TestContextExceeded:
    def _make_llm_with_side_effect(self, exc: Exception) -> ModelScopeLLM:
        llm = ModelScopeLLM()
        llm.use_local = False
        llm.llm = MagicMock()
        llm.llm.invoke.side_effect = exc
        return llm

    def test_generate_raises_context_exceeded_maximum_context(self):
        llm = self._make_llm_with_side_effect(
            Exception("This model's maximum context length is 8192 tokens")
        )
        with pytest.raises(ContextLengthExceeded):
            llm.generate([{"role": "user", "content": "x"}], mode="quick")

    def test_generate_raises_context_exceeded_too_many_tokens(self):
        llm = self._make_llm_with_side_effect(
            Exception("Error code: 400 - too many tokens in the prompt")
        )
        with pytest.raises(ContextLengthExceeded):
            llm.generate([{"role": "user", "content": "x"}], mode="quick")

    def test_generate_passthrough_other_exception(self):
        llm = self._make_llm_with_side_effect(ValueError("network error"))
        with pytest.raises(ValueError):
            llm.generate([{"role": "user", "content": "x"}], mode="quick")


# ═════════════════════════════════════════════════════════════
# 2. 大文件落盘 + 句柄化 + 分页回读（原 test_large_file_handle.py）
# ═════════════════════════════════════════════════════════════

class TestLargeFileHandle:
    def test_large_result_goes_to_handle(self, tmp_path):
        payload = {"content": LARGE_CONTENT, "file_path": "big.py"}
        handle = build_handle_summary(
            task_id="t-large", step_index=1,
            tool_name="read_text_file", payload=payload,
            max_inline_tokens=4000,
        )
        assert handle["inline"] is False
        assert handle["result_id"].startswith("tr-01-")
        assert handle["handle_path"]
        assert handle["estimated_tokens"] > 4000

        saved = load_tool_result("t-large", handle["result_id"])
        assert saved is not None
        assert saved["content"] == LARGE_CONTENT
        assert saved["file_path"] == "big.py"

    def test_small_result_stays_inline(self, tmp_path):
        payload = {"content": "short", "file_path": "small.py"}
        handle = build_handle_summary(
            task_id="t-small", step_index=1,
            tool_name="read_text_file", payload=payload,
        )
        assert handle["inline"] is True
        assert "payload" in handle
        assert "result_id" not in handle

    def test_read_tool_result_char_pagination(self):
        import core.task.tool_result_store as store
        payload = {"content": LARGE_CONTENT}
        result_id = store.save_tool_result(
            task_id="t-char", step_index=1,
            tool_name="read_text_file", payload=payload,
        )

        collected = []
        offset = 0
        for _ in range(10):
            r = read_tool_result(
                result_id=result_id, task_id="t-char",
                char_offset=offset, char_limit=5000,
            )
            assert r["success"] is True
            if not r["content"]:
                break
            collected.append(r["content"])
            offset += r["returned_chars"]
            if offset >= r["total_chars"]:
                break

        full = "".join(collected)
        parsed = json.loads(full)
        assert parsed["content"] == LARGE_CONTENT

    def test_read_tool_result_chunk_pagination(self):
        import core.task.tool_result_store as store
        chunks = [
            {"filename": "a.pdf", "text": f"chunk-{i}"} for i in range(10)
        ]
        result_id = store.save_tool_result(
            task_id="t-chunk", step_index=1,
            tool_name="search_knowledge", payload={"chunks": chunks},
        )
        r = read_tool_result(
            result_id=result_id, task_id="t-chunk",
            chunk_index=3, chunk_count=4,
        )
        assert r["success"] is True
        assert r["chunk_index"] == 3
        assert r["returned"] == 4
        assert [c["text"] for c in r["chunks"]] == [
            "chunk-3", "chunk-4", "chunk-5", "chunk-6",
        ]

    def test_read_tool_result_missing_id(self):
        r = read_tool_result(
            result_id="tr-99-deadbeef", task_id="t-missing",
        )
        assert r["success"] is False
        assert "不存在" in r["error"]

    @pytest.mark.asyncio
    async def test_fill_params_large_file_guides_read_file_range(self):
        class FakeLLM:
            def __init__(self):
                self.last_prompt = ""

            def generate(self, messages, mode="quick", timeout=None):
                self.last_prompt = messages[0]["content"] if messages else ""
                return json.dumps({
                    "file_path": "big.py",
                    "old_text": "LINE_0000: def func_0(x): return x + 0  # marker_0000",
                    "new_text": "LINE_0000: def func_0(x): return x + 0  # marker_0000  # edited",
                }, ensure_ascii=False)

        fake_llm = FakeLLM()
        agent = MagicMock()
        orch = AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: fake_llm,
        )

        tools = {"edit_file": _make_edit_file_meta()}
        state = OrchestratorState(goal="修 big.py")
        state.last_read_content = LARGE_CONTENT
        state.last_read_path = "big.py"

        params = await orch._fill_params(
            goal="修 big.py",
            tool_name="edit_file",
            tools=tools,
            state=state,
        )

        assert "文件较大" in fake_llm.last_prompt
        assert "read_file_range" in fake_llm.last_prompt
        assert "LINE_0000" in fake_llm.last_prompt
        assert "LINE_0399" not in fake_llm.last_prompt
        assert params["file_path"] == "big.py"

    @pytest.mark.asyncio
    async def test_fill_params_small_file_includes_full_content(self):
        class FakeLLM:
            def __init__(self):
                self.last_prompt = ""

            def generate(self, messages, mode="quick", timeout=None):
                self.last_prompt = messages[0]["content"] if messages else ""
                return json.dumps({
                    "file_path": "small.py",
                    "old_text": "x", "new_text": "y",
                }, ensure_ascii=False)

        fake_llm = FakeLLM()
        agent = MagicMock()
        orch = AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: fake_llm,
        )

        small = "def f():\n    return 1\n"
        tools = {"edit_file": _make_edit_file_meta()}
        state = OrchestratorState(goal="修 small.py")
        state.last_read_content = small
        state.last_read_path = "small.py"

        await orch._fill_params(
            goal="修 small.py",
            tool_name="edit_file",
            tools=tools,
            state=state,
        )

        assert "最近读取的文件内容" in fake_llm.last_prompt
        assert "def f():" in fake_llm.last_prompt
        assert "文件较大" not in fake_llm.last_prompt


# ═════════════════════════════════════════════════════════════
# 3. read_tool_result 分页（原 test_tool_result_reader.py）
# ═════════════════════════════════════════════════════════════

class TestToolResultReader:
    @pytest.fixture
    def saved_chunks(self):
        import core.task.tool_result_store as store
        payload = {
            "chunks": [
                {"filename": "a.pdf", "text": f"chunk-{i}"} for i in range(10)
            ]
        }
        return store.save_tool_result(
            task_id="t1", step_index=1,
            tool_name="search_knowledge", payload=payload,
        )

    @pytest.fixture
    def saved_text(self):
        import core.task.tool_result_store as store
        payload = {"content": "A" * 10000}
        return store.save_tool_result(
            task_id="t1", step_index=1,
            tool_name="read_text_file", payload=payload,
        )

    def test_chunk_pagination(self, saved_chunks):
        r = read_tool_result(
            result_id=saved_chunks, task_id="t1",
            chunk_index=2, chunk_count=3,
        )
        assert r["success"] is True
        assert r["chunk_index"] == 2
        assert r["returned"] == 3
        assert r["chunks"][0]["text"] == "chunk-2"

    def test_char_pagination(self, saved_text):
        r = read_tool_result(
            result_id=saved_text, task_id="t1",
            char_offset=100, char_limit=500,
        )
        assert r["success"] is True
        assert r["char_offset"] == 100
        assert r["returned_chars"] == 500

    def test_legacy_params(self, saved_chunks):
        r = read_tool_result(
            result_id=saved_chunks, task_id="t1",
            start_index=1, limit=2,
        )
        assert r["success"] is True
        assert r["chunk_index"] == 1
        assert r["returned"] == 2