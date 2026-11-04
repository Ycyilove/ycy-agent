"""真实调用测试：大文件落盘 → 句柄化 → 分段回读 → 不超限。

不 mock tool_result_store / tool_result_reader，走真实落盘和回读。
LLM 用 fake，按预设决策驱动。
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from core.task.tool_result_store import (
    build_handle_summary,
    load_tool_result,
    _RESULT_DIR as STORE_ROOT,
)
from tools.agent import ToolCallResult
from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState
from tools.tools_def.tool_result_reader import read_tool_result


# ─────────────────────────────────────────────────────────────
# 夹具：大文件内容
# ─────────────────────────────────────────────────────────────

# 约 12000 字符，含可定位标记，模拟"大文件"
def _make_large_content(lines: int = 400) -> str:
    parts = []
    for i in range(lines):
        parts.append(f"LINE_{i:04d}: def func_{i}(x): return x + {i}  # marker_{i:04d}")
    return "\n".join(parts)


LARGE_CONTENT = _make_large_content(400)
assert len(LARGE_CONTENT) > 10000, f"content too short: {len(LARGE_CONTENT)}"


# ─────────────────────────────────────────────────────────────
# 1. 落盘 + 句柄化：build_handle_summary
# ─────────────────────────────────────────────────────────────

def test_large_result_goes_to_handle(tmp_path, monkeypatch):
    """大结果应落盘，返回 inline=False + result_id + handle_path。"""
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

    payload = {"content": LARGE_CONTENT, "file_path": "big.py"}
    handle = build_handle_summary(
        task_id="t-large",
        step_index=1,
        tool_name="read_text_file",
        payload=payload,
        max_inline_tokens=4000,
    )

    assert handle["inline"] is False
    assert handle["result_id"].startswith("tr-01-")
    assert handle["handle_path"]
    # 跨盘符时应退化为绝对路径；同盘符时是相对路径
    # 两种都可接受，只要非空且指向真实文件
    assert handle["estimated_tokens"] > 4000

    # 落盘文件真实存在
    saved = load_tool_result("t-large", handle["result_id"])
    assert saved is not None
    assert saved["content"] == LARGE_CONTENT
    assert saved["file_path"] == "big.py"


def test_small_result_stays_inline(tmp_path, monkeypatch):
    """小结果应 inline，不落盘。"""
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

    payload = {"content": "short", "file_path": "small.py"}
    handle = build_handle_summary(
        task_id="t-small", step_index=1,
        tool_name="read_text_file", payload=payload,
    )
    assert handle["inline"] is True
    assert "payload" in handle
    assert "result_id" not in handle


# ─────────────────────────────────────────────────────────────
# 2. 分段回读：chunk 分页 + 字符分页
# ─────────────────────────────────────────────────────────────

def test_read_tool_result_char_pagination(tmp_path, monkeypatch):
    """按 char_offset/char_limit 分页回读，能拼回完整内容。"""
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

    payload = {"content": LARGE_CONTENT}
    result_id = store.save_tool_result(
        task_id="t-char", step_index=1,
        tool_name="read_text_file", payload=payload,
    )

    # 分 3 段读，每段 5000 字符
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
    # 落盘的是 {"content": LARGE_CONTENT} 的 JSON 序列化文本，
    # 回读拿到的是 JSON 文本片段，不是原始 content。
    # 断言：完整 JSON 能被解析回原 payload
    parsed = json.loads(full)
    assert parsed["content"] == LARGE_CONTENT


def test_read_tool_result_chunk_pagination(tmp_path, monkeypatch):
    """chunks 场景按 chunk_index/chunk_count 分页。"""
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

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


def test_read_tool_result_missing_id(tmp_path, monkeypatch):
    """不存在的 result_id 返回 success=False。"""
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

    r = read_tool_result(
        result_id="tr-99-deadbeef", task_id="t-missing",
    )
    assert r["success"] is False
    assert "不存在" in r["error"]


# ─────────────────────────────────────────────────────────────
# 3. _fill_params 大文件走"引导 read_file_range"
# ─────────────────────────────────────────────────────────────

class FakeLLMForFill:
    """只用于 fill_params：返回固定参数。"""

    def __init__(self):
        self.last_prompt = ""

    def generate(self, messages, mode="quick", timeout=None):
        self.last_prompt = messages[0]["content"] if messages else ""
        # 给 edit_file 返回参数
        return json.dumps({
            "file_path": "big.py",
            "old_text": "LINE_0000: def func_0(x): return x + 0  # marker_0000",
            "new_text": "LINE_0000: def func_0(x): return x + 0  # marker_0000  # edited",
        }, ensure_ascii=False)


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


@pytest.mark.asyncio
async def test_fill_params_large_file_guides_read_file_range():
    """state.last_read_content > 4000 时，_fill_params 应引导 read_file_range，
    而不是给截断内容。"""
    fake_llm = FakeLLMForFill()
    agent = MagicMock()
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: fake_llm,
    )

    tools = {"edit_file": _make_edit_file_meta()}
    state = OrchestratorState(goal="修 big.py")
    state.last_read_content = LARGE_CONTENT   # 12000 字符
    state.last_read_path = "big.py"

    params = await orch._fill_params(
        goal="修 big.py",
        tool_name="edit_file",
        tools=tools,
        state=state,
    )

    # prompt 应包含引导语，而不是完整截断内容
    assert "文件较大" in fake_llm.last_prompt
    assert "read_file_range" in fake_llm.last_prompt
    # 预览只给前 2000 字符
    assert "LINE_0000" in fake_llm.last_prompt
    # 不应包含完整 12000 字符（LINE_0399 在尾部，不应出现）
    assert "LINE_0399" not in fake_llm.last_prompt
    # 参数照常返回
    assert params["file_path"] == "big.py"


@pytest.mark.asyncio
async def test_fill_params_small_file_includes_full_content():
    """小文件（< 4000）应给完整内容，不引导。"""
    fake_llm = FakeLLMForFill()
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


# ─────────────────────────────────────────────────────────────
# 4. 端到端：大文件 read → 落盘 → 下一步 decide 不超预算
# ─────────────────────────────────────────────────────────────

class FakeLLME2E:
    """端到端：第 1 步 read，第 2 步 edit，第 3 步 done。"""

    def __init__(self, decisions):
        self._it = iter(decisions)
        self.decide_prompts = []

    def generate(self, messages, mode="quick", timeout=None):
        prompt = messages[0]["content"] if messages else ""
        if "多步工具编排助手" in prompt:
            self.decide_prompts.append(prompt)
            try:
                d = next(self._it)
            except StopIteration:
                d = {"thought": "stop", "done": True}
            return json.dumps(d, ensure_ascii=False)
        # fill_params
        if "`read_text_file`" in prompt:
            return json.dumps({"file_path": "big.py", "encoding": "utf-8"})
        if "`edit_file`" in prompt:
            return json.dumps({
                "file_path": "big.py",
                "old_text": "LINE_0000: def func_0(x): return x + 0  # marker_0000",
                "new_text": "LINE_0000: def func_0(x): return x + 0  # marker_0000  # edited",
            })
        if "`run_tests`" in prompt:
            return json.dumps({"project_dir": "project"})
        return "{}"


def _make_agent_for_e2e():
    agent = MagicMock()
    exec_calls = []

    async def _exec(tc):
        exec_calls.append((tc.tool_name, dict(tc.parameters)))
        if tc.tool_name == "run_tests":
            return ToolCallResult("run_tests", True, result={"success": True})
        if tc.tool_name == "read_text_file":
            return ToolCallResult(
                "read_text_file", True,
                result={
                    "content": LARGE_CONTENT,
                    "file_path": tc.parameters.get("file_path", ""),
                    "size": len(LARGE_CONTENT),
                    "line_count": LARGE_CONTENT.count("\n") + 1,
                    "success": True,
                },
            )
        if tc.tool_name in ("edit_file", "write_file"):
            return ToolCallResult(
                tc.tool_name, True,
                result={"success": True, "file_path": tc.parameters.get("file_path", "")},
            )
        return ToolCallResult(tc.tool_name, True, result={"success": True})

    agent.execute_tool_async = _exec
    agent.format_tool_result = lambda r: f"OK {r.tool_name}"
    agent.register_dynamic_tool = lambda *a, **k: None
    agent.unregister_dynamic_tool = lambda *a, **k: None

    def _list_all_tools(allowed_danger_levels=None):
        def m(desc, schema):
            meta = MagicMock()
            meta.description = desc
            meta.input_schema = schema
            return meta
        return {
            "read_text_file": m("读取文本文件", {
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "encoding": {"type": "string"}},
                "required": ["file_path"],
            }),
            "edit_file": m("局部替换", {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["file_path", "old_text", "new_text"],
            }),
            "run_tests": m("运行测试", {
                "type": "object",
                "properties": {"project_dir": {"type": "string"}},
                "required": ["project_dir"],
            }),
            "search_and_connect_mcp": m("discover", {
                "type": "object",
                "properties": {"user_query": {"type": "string"}, "capability": {"type": "string"}},
                "required": ["user_query"],
            }),
        }
    agent.list_all_tools = _list_all_tools
    return agent, exec_calls


@pytest.mark.asyncio
async def test_e2e_large_file_decide_prompt_within_budget(tmp_path, monkeypatch):
    """端到端：大文件 read 后，下一步 decide 的 prompt 不因大文件超预算。"""
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path))

    # 绕过 mem0/skills/reset_tried
    try:
        from core.memory import mem0_layer
        async def _no_wait(*a, **k): return False
        monkeypatch.setattr(mem0_layer, "wait_until_ready", _no_wait)
    except Exception:
        pass
    try:
        from tools.skills import session_skills
        monkeypatch.setattr(session_skills, "build_skills_prompt_block", lambda *a, **k: "")
    except Exception:
        pass
    try:
        from tools import mcp_discovery
        monkeypatch.setattr(mcp_discovery, "reset_tried_servers", lambda: None)
    except Exception:
        pass

    agent, exec_calls = _make_agent_for_e2e()
    fake_llm = FakeLLME2E([
        {"thought": "读大文件", "done": False, "tool_name": "read_text_file"},
        {"thought": "改文件", "done": False, "tool_name": "edit_file"},
        {"thought": "完成", "done": True},
    ])
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: fake_llm,
        max_steps=6,
        enable_intent_router=False,
    )
    # 压预算，强制触发裁剪
    orch._prompt_budget_chars = 8000

    async for _ in orch.run(
        "修复 big.py 的 bug，并确保 tests 通过",
        allowed_danger_levels={"safe", "medium", "high"},
        allow_mcp_discovery=False,
        workspace_paths=["project"],
        session_id="e2e", task_id="e2e-task",
    ):
        pass

    # 断言：所有 decide prompt 都不超过预算（软上限，允许 10% 误差）
    for i, p in enumerate(fake_llm.decide_prompts):
        assert len(p) <= 8000 * 1.1, (
            f"decide prompt #{i} 超预算: {len(p)} chars"
        )
    # 至少调过 read + edit
    tools_called = [n for n, _ in exec_calls]
    assert "read_text_file" in tools_called
    assert "edit_file" in tools_called


# ─────────────────────────────────────────────────────────────
# 5. 清理：确保测试不留落盘文件
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_result_dir(tmp_path, monkeypatch):
    """所有测试默认把落盘目录指到 tmp_path。"""
    import core.task.tool_result_store as store
    monkeypatch.setattr(store, "_RESULT_DIR", str(tmp_path / "tool_results"))