import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from tools.orchestration.core import AgentOrchestrator
from tools.agent import ToolCallResult

import pytest

@pytest.fixture(autouse=True)
def _bypass_security_gateway(monkeypatch):
    """测试里绕过安全网关：edit_file 等写操作不触发审批。"""
    import tools.security.gateway as gw

    # 1. is_write_operation 是静态/类方法，patch 到类上
    _WRITE_TOOLS = {
        "edit_file", "write_file", "delete_file", "move_file",
        "rename_file", "create_directory",
    }
    monkeypatch.setattr(
        gw.SecurityGateway, "is_write_operation",
        staticmethod(lambda name: name in _WRITE_TOOLS),
        raising=False,
    )

    # 2. get_gateway().check(...) 永远返回 allowed=True
    class _AllowDecision:
        allowed = True
        needs_approval = False
        reason = ""

    class _AllowGateway:
        def check(self, *args, **kwargs):
            return _AllowDecision()

    monkeypatch.setattr(gw, "get_gateway", lambda: _AllowGateway())

    # 3. 清掉可能已设置的安全上下文，避免 sec_ctx 走审批分支
    try:
        from tools.security.gateway import set_security_context, SecurityContext
        set_security_context(SecurityContext(
            allowed_danger_levels={"safe", "medium", "high"},
            workspace_roots=None,
            workspace_trusted=True,
            session_id="test",
        ))
    except Exception:
        pass

GOAL = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"


def make_agent():
    agent = MagicMock()
    calls = []

    async def _exec(tc):
        calls.append((tc.tool_name, dict(tc.parameters)))
        print("EXEC ->", tc.tool_name, tc.parameters)
        if tc.tool_name == "run_tests":
            return ToolCallResult("run_tests", True, result={"success": True})
        return ToolCallResult(tc.tool_name, True, result={
            "content": "ok", "success": True,
            "file_path": tc.parameters.get("file_path", ""),
        })

    agent.execute_tool_async = _exec
    agent.list_all_tools = lambda allowed_danger_levels=None: {
        "read_text_file": MagicMock(description="read"),
        "edit_file": MagicMock(description="edit"),
        "write_file": MagicMock(description="write"),
        "run_tests": MagicMock(description="run tests"),
        "search_and_connect_mcp": MagicMock(description="discover"),
    }
    agent.format_tool_result = lambda r: f"OK {r.tool_name}"
    agent.register_dynamic_tool = lambda *a, **k: None
    agent.unregister_dynamic_tool = lambda *a, **k: None
    return agent, calls


def make_llm(decisions):
    it = iter(decisions)
    llm = MagicMock()

    def _generate(messages, mode="quick", timeout=None):
        prompt = messages[0]["content"]
        if "多步工具编排助手" in prompt:
            try:
                d = next(it)
            except StopIteration:
                print("DECIDE -> [STOP] 决策已耗尽，返回 done")
                d = {"thought": "stop", "done": True}
            print("DECIDE ->", d)
            return json.dumps(d, ensure_ascii=False)
        if "`read_text_file`" in prompt:
            print("FILL read_text_file")
            return json.dumps({"file_path": "calculator/core.py"})
        if "`edit_file`" in prompt:
            print("FILL edit_file")
            return json.dumps({
                "file_path": "calculator/core.py",
                "old_text": "a/b", "new_text": "a/b if b else 0",
            })
        if "`run_tests`" in prompt:
            print("FILL run_tests")
            return json.dumps({"project_dir": "project"})
        print("FILL unknown")
        return "{}"

    llm.generate = _generate
    return llm


@pytest.mark.asyncio
async def test_write_triggers_verification():
    agent, calls = make_agent()
    decisions = [
        {"thought": "read", "done": False, "tool_name": "read_text_file"},
        {"thought": "edit", "done": False, "tool_name": "edit_file"},
        # 新设计：写后返回验证提示，LLM 自己选 run_tests
        {"thought": "跑测试", "done": False, "tool_name": "run_tests"},
        {"thought": "done", "done": True},
    ]
    shared_llm = make_llm(decisions)
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: shared_llm,
        max_steps=8,
    )

    events = []
    async for evt in orch.run(
        GOAL,
        allowed_danger_levels={"safe", "medium", "high"},
        allow_mcp_discovery=False,
        workspace_paths=["project"],
        session_id="s", task_id="t",
    ):
        events.append(evt)
        if evt.get("type") in ("done", "max_steps", "error"):
            break

    assert any(name == "run_tests" for name, _ in calls), f"写完未自动验证：{calls}"
    done_evt = next((e for e in events if e.get("type") == "done"), None)
    assert done_evt is not None