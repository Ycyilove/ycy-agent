from unittest.mock import AsyncMock, MagicMock
import pytest

from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState
from tools.orchestration.task_types import GapKind
from tools.agent import ToolCallResult


def make_orch():
    agent = MagicMock()
    agent.execute_tool_async = AsyncMock()
    llm = MagicMock()
    return AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
    )


def test_goal_needs_verification():
    o = make_orch()
    assert o._goal_needs_verification("修复代码并确保 tests 通过") is True
    assert o._goal_needs_verification("运行测试验证") is True
    assert o._goal_needs_verification("看看这个文件") is False
    assert o._goal_needs_verification("") is False


def test_find_verify_tool_from_tools():
    o = make_orch()
    tools = {
        "read_text_file": MagicMock(),
        "edit_file": MagicMock(),
        "run_tests": MagicMock(),
    }
    assert o._find_verify_tool(tools) == "run_tests"

    tools2 = {"read_text_file": MagicMock()}
    assert o._find_verify_tool(tools2) is None


@pytest.mark.asyncio
async def test_collect_verification_records_evidence():
    o = make_orch()
    agent = o._agent_provider()
    agent.execute_tool_async.return_value = ToolCallResult(
        tool_name="run_tests", success=True,
        result={"success": True, "passed": 5, "failed": 0},
    )
    st = OrchestratorState(goal="修复并跑 tests")
    st.workspace_roots = ["project"]
    tools = {"run_tests": MagicMock()}

    res = await o._collect_verification(st.goal, st, tools)
    assert res is not None
    # 新设计：不自动执行，返回 MISSING_VERIFICATION 让 LLM 决策
    assert res.sufficient is False
    assert res.gap_kind.value == "missing_verification"
    assert "run_tests" in (res.gap + res.next_hint)
    assert st.verification_evidence is None
    assert st.verification_evidence["success"] is True


@pytest.mark.asyncio
async def test_collect_verification_failure_gap_kind():
    o = make_orch()
    agent = o._agent_provider()
    agent.execute_tool_async.return_value = ToolCallResult(
        tool_name="run_tests", success=True,
        result={"success": False, "failed": 2, "failures": [
            {"test": "test_div", "message": "ZeroDivisionError"},
        ]},
    )
    st = OrchestratorState(goal="修复并跑 tests")
    st.workspace_roots = ["project"]
    tools = {"run_tests": MagicMock()}

    res = await o._collect_verification(st.goal, st, tools)
    assert res is not None
    assert res.sufficient is False
    assert res.gap_kind == GapKind.MISSING_VERIFICATION