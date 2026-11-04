from unittest.mock import MagicMock
from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState


def make_orch():
    return AgentOrchestrator(
        agent_provider=lambda: MagicMock(),
        llm_provider=lambda: MagicMock(),
    )


def test_validate_done_blocks_read_only():
    orch = make_orch()
    goal = "修复 calculator/core.py，并确保 tests 通过"
    st = OrchestratorState(goal=goal)
    st.read_paths = {"calculator/core.py"}
    st.succeeded_tool_names = {"read_text_file": 1}
    reason = orch._validate_done(goal, st)
    assert reason is not None
    assert "MISSING_ACTION" in reason or "写" in reason


def test_validate_done_allows_after_write_and_verify():
    orch = make_orch()
    goal = "修复 calculator/core.py，并确保 tests 通过"
    st = OrchestratorState(goal=goal)
    st.read_paths = {"calculator/core.py"}
    st.written_paths = {"calculator/core.py"}
    st.succeeded_tool_names = {"read_text_file": 1, "edit_file": 2, "run_tests": 3}
    st.verification_evidence = {"success": True}
    assert orch._validate_done(goal, st) is None


def test_validate_done_chitchat_passthrough():
    orch = make_orch()
    st = OrchestratorState(goal="你好")
    assert orch._validate_done("你好", st) is None