"""排查 _fill_params 必填参数缺失时是否重试。

当前行为：第一次返回不合法 → 过滤后剩 {} → 直接执行（失败）
期望行为：第一次返回不合法 → 把"缺 X"反馈给 LLM 重试一次 → 仍缺才跳过
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tools.agent import ToolCallResult
from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState


def _meta(desc, schema):
    m = MagicMock()
    m.description = desc
    m.input_schema = schema
    return m


@pytest.mark.asyncio
async def test_fill_params_retries_on_missing_required():
    """第一次返回缺 pattern，第二次应被要求补上。

    期望：_fill_params 内部重试，最终返回含 pattern 的参数。
    """
    agent = MagicMock()
    tools = {
        "search_files": _meta("搜索文件", {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        }),
    }

    calls = []
    responses = iter([
        # 第一次：抄了上一轮 run_tests 的参数，缺 pattern
        json.dumps({
            "project_dir": "D:/x", "test_path": "tests/", "timeout": 120,
        }, ensure_ascii=False),
        # 第二次：被反馈"缺 pattern"，改对
        json.dumps({
            "pattern": "*.py", "path": "D:/x",
        }, ensure_ascii=False),
    ])

    def _gen(msgs, **kw):
        prompt = msgs[0]["content"]
        calls.append(prompt)
        return next(responses)

    llm = MagicMock()
    llm.generate = _gen
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
    )
    state = OrchestratorState(goal="找配置文件")

    params = await orch._fill_params(
        "找一下项目里所有 .py 文件",
        "search_files",
        tools,
        state=state,
    )

    # 断言 1：LLM 被调了 2 次（第一次 + 重试）
    assert len(calls) == 2, (
        f"期望重试 1 次（共 2 次调用），实际 {len(calls)} 次。"
        f"当前 _fill_params 没实现重试。"
    )
    # 断言 2：最终参数含 pattern
    assert params.get("pattern") == "*.py", (
        f"重试后仍缺 pattern：{params}"
    )
    # 断言 3：第二次 prompt 含"缺 pattern"的反馈
    assert "pattern" in calls[1] and (
        "缺少" in calls[1] or "required" in calls[1].lower()
    ), "第二次 prompt 应包含缺失参数的反馈"


@pytest.mark.asyncio
async def test_fill_params_gives_up_after_one_retry():
    """两次都缺必填 → 返回 {}，不无限重试。"""
    agent = MagicMock()
    tools = {
        "search_files": _meta("搜索文件", {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        }),
    }
    calls = []
    def _gen(msgs, **kw):
        calls.append(msgs[0]["content"])
        return json.dumps({"path": "x"}, ensure_ascii=False)  # 永远缺 pattern

    llm = MagicMock()
    llm.generate = _gen
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
    )
    state = OrchestratorState(goal="x")

    params = await orch._fill_params("找文件", "search_files", tools, state=state)

    # 断言：最多 2 次调用（1 次初始 + 1 次重试），不死循环
    assert len(calls) <= 2, f"重试不应超过 1 次，实际 {len(calls)} 次"
    assert "pattern" not in params or not params.get("pattern"), (
        f"两次都缺，应返回不含 pattern 的参数：{params}"
    )


@pytest.mark.asyncio
async def test_fill_params_does_not_echo_previous_tool_params():
    """_fill_params 不应把历史里其他工具的参数抄给当前工具。"""
    agent = MagicMock()
    tools = {
        "search_files": _meta("搜索文件", {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        }),
    }
    prompts = []
    def _gen(msgs, **kw):
        prompts.append(msgs[0]["content"])
        return json.dumps({"pattern": "*.py"}, ensure_ascii=False)

    llm = MagicMock()
    llm.generate = _gen
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
    )
    state = OrchestratorState(goal="找文件")
    # 模拟历史里有 run_tests 的参数
    state.history.append(
        "Step 2: run_tests({'project_dir': 'D:/x', 'test_path': 'tests/'}) → FAIL"
    )

    await orch._fill_params("找文件", "search_files", tools, state=state)

    # 断言：prompt 里应明确当前工具是 search_files，参数表只有 pattern
    p = prompts[0]
    assert "`search_files`" in p, "prompt 应明确当前工具名"
    assert "`pattern`" in p, "prompt 应含当前工具的参数表"
    # 不应把 run_tests 的 project_dir 作为"可抄的示例"突出
    # （历史里出现可以，但不应被 LLM 误认为当前工具参数）