"""排查验证能力是否被特化。

当前设计：
    - _VERIFY_TOOL_HINTS 硬编码 run_tests/pytest/lint/typecheck
    - _collect_verification 自动执行 run_tests
    - _goal_needs_verification 只匹配"测试/tests/验证/确保通过"

问题：
    1. JS 项目跑 npm test → run_python 跑不了
    2. 文档生成验证 → 需要读回文件，不是 run_tests
    3. API 验证 → 需要 http_get，不是 run_tests
    4. 非修改任务也要验证（"查 X"→必须实际抓取）
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState
from tools.orchestration.task_types import GapKind
from tools.orchestration.audit import run_audit, collect_evidence, extract_requirements


def _make_orch():
    return AgentOrchestrator(
        agent_provider=lambda: MagicMock(),
        llm_provider=lambda: MagicMock(),
    )


# ─────────────────────────────────────────────────────────────
# 1. 验证场景覆盖度
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("goal,expected_verify", [
    # Python 项目
    ("修复 calculator/core.py 的 bug，确保 tests 通过", True),
    # JS 项目
    ("修复 src/index.js，确保 npm test 通过", True),
    # 文档生成
    ("生成 report.md，并检查内容包含所有章节", True),
    # 数据处理
    ("处理 data.csv，验证行数不超过 1000", True),
    # API 调用
    ("调用 /api/users 接口，确认返回 200", True),
    # 数据库
    ("插入用户记录后，查询确认数据已入库", True),
    # 配置修改
    ("改 config.py 的 DATA_DIR，启动服务确认不报错", True),
    # 纯查询（非修改也要验证）
    ("查一下维基百科上维尼修斯的最新状态", True),
    # 纯闲聊
    ("你好", False),
])
def test_goal_needs_verification_coverage(goal, expected_verify):
    """_goal_needs_verification 应覆盖多种验证场景。

    当前实现只匹配"测试/tests/验证/确保通过"，
    JS/文档/API/数据库场景会漏。
    """
    orch = _make_orch()
    actual = orch._goal_needs_verification(goal)
    assert actual == expected_verify, (
        f"goal={goal!r}\n"
        f"期望 verify={expected_verify}，实际 {actual}\n"
        f"当前 _goal_needs_verification 的正则："
        f"r'(测试|tests?|验证|确保.*通过|make.*pass|run.*test|检查.*通过)'"
    )


# ─────────────────────────────────────────────────────────────
# 2. 验证工具发现
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tools,expected_tool", [
    # Python
    ({"run_python": MagicMock(), "read_text_file": MagicMock()}, "run_python"),
    ({"run_tests": MagicMock()}, "run_tests"),
    # 通用 shell
    ({"run_shell": MagicMock()}, "run_shell"),
    ({"bash": MagicMock()}, "bash"),
    # 没有执行类工具
    ({"read_text_file": MagicMock(), "search_content": MagicMock()}, None),
])
def test_find_verify_tool_generality(tools, expected_tool):
    """_find_verify_tool 应能找到多种执行类工具，不只 run_tests。

    当前 _VERIFY_TOOL_HINTS = ("run_test", "pytest", "test_runner",
                               "run_check", "lint", "typecheck")
    找不到 run_python/run_shell/bash。
    """
    orch = _make_orch()
    actual = orch._find_verify_tool(tools)
    assert actual == expected_tool, (
        f"tools={list(tools)}\n"
        f"期望 {expected_tool}，实际 {actual}"
    )


# ─────────────────────────────────────────────────────────────
# 3. 非修改任务的验证
# ─────────────────────────────────────────────────────────────

def test_non_modify_task_requires_fetch():
    """纯查询任务（"查 X"）也需要验证：必须实际抓取，不能凭记忆。"""
    goal = "查一下今天 A 股大盘指数"
    reqs = extract_requirements(goal)
    kinds = [r.kind for r in reqs]
    # 应抽出 fetch 类要求
    assert "fetch" in kinds, (
        f"纯查询任务应抽出 fetch 要求，实际 {kinds}。\n"
        f"当前 audit 的 fetch 正则可能漏了'查'这个动词。"
    )

    # 有 fetch 要求 + 无 fetch 证据 → 不满足
    st = OrchestratorState(goal=goal)
    audit = run_audit(goal, st, evidence=collect_evidence(st))
    assert audit.satisfied is False
    assert audit.gap_kind == GapKind.MISSING_CONTENT


def test_modify_task_without_verification_blocks():
    """修改任务有 verify 要求但无验证证据 → 不满足。"""
    goal = "修复 bug，确保 tests 通过"
    st = OrchestratorState(goal=goal)
    st.written_paths = {"f.py"}
    st.succeeded_tool_names = {"read_text_file": 1, "edit_file": 2}
    audit = run_audit(goal, st, evidence=collect_evidence(st))
    assert audit.satisfied is False
    assert audit.gap_kind == GapKind.MISSING_VERIFICATION


def test_modify_task_with_generic_verification_passes():
    """修改任务用 run_python 跑命令验证 → 应满足 verify。"""
    goal = "修复 bug，确保 tests 通过"
    st = OrchestratorState(goal=goal)
    st.written_paths = {"f.py"}
    st.succeeded_tool_names = {
        "read_text_file": 1, "edit_file": 2, "run_python": 3,
    }
    # 关键：verification_evidence 应能接受任意执行类工具的结果
    st.verification_evidence = {
        "tool": "run_python",  # 不是 run_tests
        "success": True,
        "payload": {"exit_code": 0, "stdout": "5 passed"},
    }
    audit = run_audit(goal, st, evidence=collect_evidence(st))
    assert audit.satisfied is True, (
        f"用 run_python 验证成功仍判不满足：{audit.gap}\n"
        f"当前 audit 的 verify 分支可能硬编码了 run_tests。"
    )


# ─────────────────────────────────────────────────────────────
# 4. _collect_verification 不应自动执行，应提示 LLM
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collect_verification_does_not_hardcode_run_tests():
    """_collect_verification 不应硬编码 run_tests，应找通用执行工具。"""
    agent = MagicMock()
    exec_calls = []

    async def _exec(tc):
        exec_calls.append(tc.tool_name)
        return MagicMock(success=True, result={"success": True})

    agent.execute_tool_async = _exec
    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: MagicMock(),
    )

    # 只有 run_python，没有 run_tests
    tools = {"run_python": MagicMock()}
    st = OrchestratorState(goal="修复 bug，确保 tests 通过")
    st.workspace_roots = ["project"]

    result = await orch._collect_verification(
        "修复 bug，确保 tests 通过", st, tools,
    )
    # 新设计：不自动执行，返回提示含 run_python
    assert result is not None
    assert result.sufficient is False
    assert result.gap_kind.value == "missing_verification"
    assert "run_python" in (result.gap + result.next_hint), (
        f"提示里没有 run_python：gap={result.gap!r}, hint={result.next_hint!r}"
    )
    # 确认没有自动执行
    assert "run_python" not in exec_calls, (
        f"_collect_verification 不应自动执行，实际执行了：{exec_calls}"
    )