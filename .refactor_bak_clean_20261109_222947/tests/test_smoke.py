"""冒烟测试：跨域端到端，验证核心链路不崩。"""

import pytest

from tools.orchestration.prompt import ORCHESTRATOR_PROMPT
from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState
from unittest.mock import MagicMock


def test_prompt_has_tool_chain():
    """ORCHESTRATOR_PROMPT 必须包含工具链约束（防 prompt 回退）。"""
    assert "convert_to_pdf" in ORCHESTRATOR_PROMPT
    assert (
        "需要前置产物" in ORCHESTRATOR_PROMPT
        or "前置条件" in ORCHESTRATOR_PROMPT
    )
    assert (
        "失败 2 次" in ORCHESTRATOR_PROMPT
        or "禁止再选" in ORCHESTRATOR_PROMPT
    )


def test_prompt_has_path_rule():
    """路径规则必须禁止从 goal 抄路径。"""
    assert "绝对不要从" in ORCHESTRATOR_PROMPT
    assert "任务" in ORCHESTRATOR_PROMPT


def test_prompt_has_statement_vs_instruction():
    """必须区分陈述 vs 指令（防止“为什么删除文件”触发 delete_file）。"""
    assert "陈述" in ORCHESTRATOR_PROMPT
    assert "指令" in ORCHESTRATOR_PROMPT


async def test_smoke_decide_empty_history():
    """_decide 在空 history 下能返回决策，不崩。"""
    agent = MagicMock()
    agent.list_all_tools = lambda **kw: {
        "read_text_file": MagicMock(
            description="read",
            input_schema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        ),
    }
    llm = MagicMock()
    llm.generate = lambda msgs, **kw: '{"thought": "done", "done": true}'
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
        enable_intent_router=False,
    )
    state = OrchestratorState(goal="测试", session_id="s")
    decision = await orch._decide("测试", state, agent.list_all_tools())
    assert decision is not None
    assert decision.done is True