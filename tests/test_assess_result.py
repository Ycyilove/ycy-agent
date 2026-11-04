import pytest
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_assess_read_success_not_done_when_modify_missing():
    orch = make_orch()
    goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
    st = OrchestratorState(goal=goal)
    result = ToolCallResult(
        tool_name="read_text_file", success=True,
        result={"content": "def divide(): ...", "file_path": "calculator/core.py"},
    )
    res = await orch._assess_result(
        goal, "read_text_file",
        {"file_path": "calculator/core.py"}, result, st,
    )
    assert res is not None
    assert res.sufficient is False
    assert res.gap_kind == GapKind.MISSING_ACTION


@pytest.mark.asyncio
async def test_assess_write_success_marks_written():
    orch = make_orch()
    goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
    st = OrchestratorState(goal=goal)
    result = ToolCallResult(
        tool_name="edit_file", success=True,
        result={"success": True, "file_path": "calculator/core.py"},
    )
    await orch._assess_result(
        goal, "edit_file",
        {"file_path": "calculator/core.py"}, result, st,
    )
    assert "calculator/core.py" in st.written_paths


@pytest.mark.asyncio
async def test_assess_chitchat_short_circuit():
    orch = make_orch()
    st = OrchestratorState(goal="你好")
    result = ToolCallResult(tool_name="noop", success=True, result={})
    res = await orch._assess_result("你好", "noop", {}, result, st)
    assert res is not None and res.sufficient is True