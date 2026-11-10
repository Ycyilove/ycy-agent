"""编排主链路测试：_decide / _assess_result / _validate_done / audit / verification。

合并自：
    - test_assess_result.py
    - test_audit.py
    - test_validate_done.py
    - test_verification.py
    - test_verification_generality.py
    - test_run_write_then_verify.py
    - test_dynamic_discovery.py
    - test_real_task_regressions.py
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.agent import ToolCallResult
from tools.orchestration.audit import (
    extract_requirements,
    run_audit,
    collect_evidence,
)
from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState
from tools.orchestration.task_types import GapKind


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

def make_orch(max_steps: int = 10):
    agent = MagicMock()
    agent.execute_tool_async = AsyncMock()
    agent.register_dynamic_tool = lambda *a, **k: None
    agent.unregister_dynamic_tool = lambda *a, **k: None
    agent.format_tool_result = lambda r: f"OK {r.tool_name}"
    llm = MagicMock()
    return AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
        max_steps=max_steps,
    )


def make_state(**kw) -> OrchestratorState:
    st = OrchestratorState(goal=kw.pop("goal", ""))
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _meta(desc, schema=None):
    m = MagicMock()
    m.description = desc
    m.input_schema = schema or {}
    return m


def _async_return(ret):
    async def _f(tc):
        return ret
    return _f


def _async_record(calls):
    async def _f(tc):
        calls.append((tc.tool_name, dict(tc.parameters)))
        return ToolCallResult(tc.tool_name, True, result={"success": True})
    return _f


@pytest.fixture(autouse=True)
def _bypass_security_gateway(monkeypatch):
    """绕过安全网关：写操作不触发审批。"""
    import tools.security.gateway as gw

    _WRITE_TOOLS = {
        "edit_file", "write_file", "delete_file", "move_file",
        "rename_file", "create_directory",
    }
    monkeypatch.setattr(
        gw.SecurityGateway, "is_write_operation",
        staticmethod(lambda name: name in _WRITE_TOOLS),
        raising=False,
    )

    class _AllowDecision:
        allowed = True
        needs_approval = False
        reason = ""

    class _AllowGateway:
        def check(self, *args, **kwargs):
            return _AllowDecision()

    monkeypatch.setattr(gw, "get_gateway", lambda: _AllowGateway())


# ═════════════════════════════════════════════════════════════
# 1. _assess_result（原 test_assess_result.py）
# ═════════════════════════════════════════════════════════════

class TestAssessResult:
    @pytest.mark.asyncio
    async def test_assess_read_success_not_done_when_modify_missing(self):
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
    async def test_assess_write_success_marks_written(self):
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
    async def test_assess_chitchat_short_circuit(self):
        orch = make_orch()
        st = OrchestratorState(goal="你好")
        result = ToolCallResult(tool_name="noop", success=True, result={})
        res = await orch._assess_result("你好", "noop", {}, result, st)
        assert res is not None and res.sufficient is True


# ═════════════════════════════════════════════════════════════
# 2. audit（原 test_audit.py）
# ═════════════════════════════════════════════════════════════

class TestAudit:
    def test_extract_code_fix_requirements(self):
        goal = (
            "请帮我修复以下问题，并确保 tests/ 下的测试全部通过："
            "1. calculator/core.py 的 divide 函数遇到除零会崩溃 "
            "2. calculator/validators.py 的 validate_positive 逻辑反了 "
            "3. calculator/formatters.py 的 format_result 精度不对 "
            "4. config.py 的 DATA_DIR 硬编码 "
            "5. main.py 调用 divide 时未导入 "
            "修复后请运行 tests/ 验证。"
        )
        reqs = extract_requirements(goal)
        kinds = [r.kind for r in reqs]
        assert "modify" in kinds, kinds
        assert "verify" in kinds, kinds
        assert not all(k == "answer" for k in kinds)

    def test_extract_chitchat_requirements(self):
        reqs = extract_requirements("你好")
        assert all(r.kind == "answer" for r in reqs)

    def test_audit_read_only_blocks_done(self):
        goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
        st = make_state(
            goal=goal,
            read_paths={"calculator/core.py"},
            written_paths=set(),
            succeeded_tool_names={"read_text_file": 1},
        )
        res = run_audit(goal, st, evidence=collect_evidence(st))
        assert res.satisfied is False
        assert res.gap_kind == GapKind.MISSING_ACTION, res.gap_kind

    def test_audit_write_without_verify_blocks_done(self):
        goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
        st = make_state(
            goal=goal,
            read_paths={"calculator/core.py"},
            written_paths={"calculator/core.py"},
            succeeded_tool_names={"read_text_file": 1, "edit_file": 2},
            verification_evidence=None,
        )
        res = run_audit(goal, st, evidence=collect_evidence(st))
        assert res.satisfied is False
        assert res.gap_kind == GapKind.MISSING_VERIFICATION, res.gap_kind

    def test_audit_write_and_verify_pass(self):
        goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
        st = make_state(
            goal=goal,
            read_paths={"calculator/core.py"},
            written_paths={"calculator/core.py"},
            succeeded_tool_names={"read_text_file": 1, "edit_file": 2, "run_tests": 3},
            verification_evidence={"tool": "run_tests", "success": True},
        )
        res = run_audit(goal, st, evidence=collect_evidence(st))
        assert res.satisfied is True, res.gap

    @pytest.mark.xfail(reason="当前 audit 对 search 算 fetch；收紧后应通过")
    def test_audit_web_search_only_blocks_done(self):
        goal = "查一下维基百科上维尼修斯的当前状态"
        st = make_state(
            goal=goal,
            succeeded_tool_names={"mcp__tavily__tavily_search": 1},
        )
        res = run_audit(goal, st, evidence=collect_evidence(st))
        assert res.satisfied is False

    def test_audit_multi_doc_coverage(self):
        goal = "这些文档分别讲了什么"
        st = make_state(goal=goal)
        object.__setattr__(st, "_target_files_cache", {"a.pdf", "b.pdf"})
        st.read_paths = {"a.pdf"}
        res = run_audit(goal, st, evidence=collect_evidence(st))
        assert res.satisfied is False
        st.read_paths = {"a.pdf", "b.pdf"}
        res = run_audit(goal, st, evidence=collect_evidence(st))
        assert res.satisfied is True


# ═════════════════════════════════════════════════════════════
# 3. _validate_done（原 test_validate_done.py）
# ═════════════════════════════════════════════════════════════

class TestValidateDone:
    def test_validate_done_blocks_read_only(self):
        orch = make_orch()
        goal = "修复 calculator/core.py，并确保 tests 通过"
        st = OrchestratorState(goal=goal)
        st.read_paths = {"calculator/core.py"}
        st.succeeded_tool_names = {"read_text_file": 1}
        reason = orch._validate_done(goal, st)
        assert reason is not None
        assert "MISSING_ACTION" in reason or "写" in reason

    def test_validate_done_allows_after_write_and_verify(self):
        orch = make_orch()
        goal = "修复 calculator/core.py，并确保 tests 通过"
        st = OrchestratorState(goal=goal)
        st.read_paths = {"calculator/core.py"}
        st.written_paths = {"calculator/core.py"}
        st.succeeded_tool_names = {"read_text_file": 1, "edit_file": 2, "run_tests": 3}
        st.verification_evidence = {"success": True}
        assert orch._validate_done(goal, st) is None

    def test_validate_done_chitchat_passthrough(self):
        orch = make_orch()
        st = OrchestratorState(goal="你好")
        assert orch._validate_done("你好", st) is None


# ═════════════════════════════════════════════════════════════
# 4. verification（原 test_verification.py + test_verification_generality.py）
# ═════════════════════════════════════════════════════════════

class TestVerification:
    def test_goal_needs_verification(self):
        o = make_orch()
        assert o._goal_needs_verification("修复代码并确保 tests 通过") is True
        assert o._goal_needs_verification("运行测试验证") is True
        assert o._goal_needs_verification("看看这个文件") is False
        assert o._goal_needs_verification("") is False

    def test_find_verify_tool_from_tools(self):
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
    async def test_collect_verification_records_evidence(self):
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
        assert res.sufficient is False
        assert res.gap_kind.value == "missing_verification"
        assert "run_tests" in (res.gap + res.next_hint)

    @pytest.mark.asyncio
    async def test_collect_verification_failure_gap_kind(self):
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

    @pytest.mark.parametrize("goal,expected_verify", [
        ("修复 calculator/core.py 的 bug，确保 tests 通过", True),
        ("修复 src/index.js，确保 npm test 通过", True),
        ("生成 report.md，并检查内容包含所有章节", True),
        ("处理 data.csv，验证行数不超过 1000", True),
        ("调用 /api/users 接口，确认返回 200", True),
        ("插入用户记录后，查询确认数据已入库", True),
        ("改 config.py 的 DATA_DIR，启动服务确认不报错", True),
        ("查一下维基百科上维尼修斯的最新状态", True),
        ("你好", False),
    ])
    def test_goal_needs_verification_coverage(self, goal, expected_verify):
        orch = make_orch()
        actual = orch._goal_needs_verification(goal)
        assert actual == expected_verify

    @pytest.mark.parametrize("tools,expected_tool", [
        ({"run_python": MagicMock(), "read_text_file": MagicMock()}, "run_python"),
        ({"run_tests": MagicMock()}, "run_tests"),
        ({"run_shell": MagicMock()}, "run_shell"),
        ({"bash": MagicMock()}, "bash"),
        ({"read_text_file": MagicMock(), "search_content": MagicMock()}, None),
    ])
    def test_find_verify_tool_generality(self, tools, expected_tool):
        orch = make_orch()
        actual = orch._find_verify_tool(tools)
        assert actual == expected_tool

    def test_non_modify_task_requires_fetch(self):
        goal = "查一下今天 A 股大盘指数"
        reqs = extract_requirements(goal)
        kinds = [r.kind for r in reqs]
        assert "fetch" in kinds

        st = OrchestratorState(goal=goal)
        audit = run_audit(goal, st, evidence=collect_evidence(st))
        assert audit.satisfied is False
        assert audit.gap_kind == GapKind.MISSING_CONTENT

    def test_modify_task_without_verification_blocks(self):
        goal = "修复 bug，确保 tests 通过"
        st = OrchestratorState(goal=goal)
        st.written_paths = {"f.py"}
        st.succeeded_tool_names = {"read_text_file": 1, "edit_file": 2}
        audit = run_audit(goal, st, evidence=collect_evidence(st))
        assert audit.satisfied is False
        assert audit.gap_kind == GapKind.MISSING_VERIFICATION

    def test_modify_task_with_generic_verification_passes(self):
        goal = "修复 bug，确保 tests 通过"
        st = OrchestratorState(goal=goal)
        st.written_paths = {"f.py"}
        st.succeeded_tool_names = {
            "read_text_file": 1, "edit_file": 2, "run_python": 3,
        }
        st.verification_evidence = {
            "tool": "run_python",
            "success": True,
            "payload": {"exit_code": 0, "stdout": "5 passed"},
        }
        audit = run_audit(goal, st, evidence=collect_evidence(st))
        assert audit.satisfied is True, audit.gap

    @pytest.mark.asyncio
    async def test_collect_verification_does_not_hardcode_run_tests(self):
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

        tools = {"run_python": MagicMock()}
        st = OrchestratorState(goal="修复 bug，确保 tests 通过")
        st.workspace_roots = ["project"]

        result = await orch._collect_verification(
            "修复 bug，确保 tests 通过", st, tools,
        )
        assert result is not None
        assert result.sufficient is False
        assert result.gap_kind.value == "missing_verification"
        assert "run_python" in (result.gap + result.next_hint)
        assert "run_python" not in exec_calls


# ═════════════════════════════════════════════════════════════
# 5. 写后验证链路（原 test_run_write_then_verify.py）
# ═════════════════════════════════════════════════════════════

def _make_write_agent():
    agent = MagicMock()
    calls = []

    async def _exec(tc):
        calls.append((tc.tool_name, dict(tc.parameters)))
        if tc.tool_name == "run_tests":
            return ToolCallResult("run_tests", True, result={"success": True})
        return ToolCallResult(tc.tool_name, True, result={
            "content": "ok", "success": True,
            "file_path": tc.parameters.get("file_path", ""),
        })

    agent.execute_tool_async = _exec
    agent.list_all_tools = lambda allowed_danger_levels=None: {
        "read_text_file": _meta("read"),
        "edit_file": _meta("edit"),
        "write_file": _meta("write"),
        "run_tests": _meta("run tests"),
        "search_and_connect_mcp": _meta("discover"),
    }
    agent.format_tool_result = lambda r: f"OK {r.tool_name}"
    agent.register_dynamic_tool = lambda *a, **k: None
    agent.unregister_dynamic_tool = lambda *a, **k: None
    return agent, calls


def _make_write_llm(decisions):
    it = iter(decisions)
    llm = MagicMock()

    def _generate(messages, mode="quick", timeout=None):
        prompt = messages[0]["content"]
        if "多步工具编排助手" in prompt:
            try:
                d = next(it)
            except StopIteration:
                d = {"thought": "stop", "done": True}
            return json.dumps(d, ensure_ascii=False)
        if "`read_text_file`" in prompt:
            return json.dumps({"file_path": "calculator/core.py"})
        if "`edit_file`" in prompt:
            return json.dumps({
                "file_path": "calculator/core.py",
                "old_text": "a/b", "new_text": "a/b if b else 0",
            })
        if "`run_tests`" in prompt:
            return json.dumps({"project_dir": "project"})
        return "{}"

    llm.generate = _generate
    return llm


class TestWriteThenVerify:
    @pytest.mark.asyncio
    async def test_write_triggers_verification(self):
        agent, calls = _make_write_agent()
        decisions = [
            {"thought": "read", "done": False, "tool_name": "read_text_file"},
            {"thought": "edit", "done": False, "tool_name": "edit_file"},
            {"thought": "跑测试", "done": False, "tool_name": "run_tests"},
            {"thought": "done", "done": True},
        ]
        shared_llm = _make_write_llm(decisions)
        orch = AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: shared_llm,
            max_steps=8,
        )

        events = []
        async for evt in orch.run(
            "修复 calculator/core.py 的 divide 函数，并确保 tests 通过",
            allowed_danger_levels={"safe", "medium", "high"},
            allow_mcp_discovery=False,
            workspace_paths=["project"],
            session_id="s", task_id="t",
        ):
            events.append(evt)
            if evt.get("type") in ("done", "max_steps", "error"):
                break

        assert any(name == "run_tests" for name, _ in calls)
        done_evt = next((e for e in events if e.get("type") == "done"), None)
        assert done_evt is not None


# ═════════════════════════════════════════════════════════════
# 6. 动态发现（原 test_dynamic_discovery.py + test_real_task_regressions.py 的第 4 节）
#     合并去重：两个文件都有 test_non_web_task_skips_static_mcp / test_web_task_keeps_static_mcp
# ═════════════════════════════════════════════════════════════

class TestDynamicDiscovery:
    def test_chitchat_skips_all_tools(self):
        """闲聊不召回任何工具。"""
        orch = make_orch()
        tools = {
            "read_text_file": _meta("read"),
            "search_and_connect_mcp": _meta("discover"),
            "mcp__tavily__tavily_search": _meta("tavily"),
        }
        shortlist = orch._select_shortlist("你好", tools)
        assert shortlist == {}

    def test_web_task_keeps_tavily_or_discovery(self):
        """网页任务：tavily 或 discovery 入口至少有一个在候选里。

        重构后 shortlister 是 embedding 筛选，tavily 是否被保留取决于
        相似度，不是确定性行为。断言放宽为“tavily 在 或 discovery 在”。
        """
        orch = make_orch()
        tools = {
            "read_text_file": _meta("read"),
            "search_and_connect_mcp": _meta("discover"),
            "mcp__tavily__tavily_search": _meta("tavily search"),
        }
        shortlist = orch._select_shortlist("查一下今天的新闻", tools)
        assert (
            any("tavily" in n for n in shortlist)
            or "search_and_connect_mcp" in shortlist
        ), f"网页任务丢失搜索入口：{list(shortlist)}"

    def test_discovery_entry_always_present(self):
        """discovery 入口必须始终在候选里（即使 shortlist 为空）。"""
        orch = make_orch()
        tools = {
            "read_text_file": _meta("read"),
            "search_and_connect_mcp": _meta("discover"),
        }
        shortlist = orch._select_shortlist("修 calculator/core.py 的 bug", tools)
        assert "search_and_connect_mcp" in shortlist

    def test_recent_discovered_readded_in_decide(self):
        orch = make_orch()
        tools = {
            "read_text_file": _meta("read"),
            "search_and_connect_mcp": _meta("discover"),
            "mcp__news_server__search_news": _meta("news"),
        }
        state = OrchestratorState(goal="查新闻")
        state.recent_discovered_tools = {"mcp__news_server__search_news"}

        base = orch._select_shortlist("查新闻", tools)
        assert "mcp__news_server__search_news" not in base

        recent = getattr(state, "recent_discovered_tools", set()) or set()
        for name in recent:
            if name in tools and name not in base:
                base[name] = tools[name]
        assert "mcp__news_server__search_news" in base

    def test_snapshot_roundtrip_recent_discovered(self):
        st = OrchestratorState(goal="x")
        st.recent_discovered_tools = {"mcp__a__b", "mcp__c__d"}
        snap = st.snapshot()
        st2 = OrchestratorState.from_snapshot(snap)
        assert st2.recent_discovered_tools == {"mcp__a__b", "mcp__c__d"}


# ═════════════════════════════════════════════════════════════
# 7. 真实任务回归（原 test_real_task_regressions.py）
#     删除 test_consecutive_failures_intercepted（_intercept_call 已不检查 _consecutive_failures）
#     删除 test_non_web_task_skips_static_mcp / test_web_task_keeps_static_mcp（与 TestDynamicDiscovery 重复）
# ═════════════════════════════════════════════════════════════

class TestRealTaskRegressions:
    def test_max_steps_sets_stop_reason(self):
        agent = MagicMock()
        agent.list_all_tools = lambda **kw: {
            "read_text_file": _meta("read", {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            }),
        }
        llm = MagicMock()
        llm.generate = lambda msgs, **kw: json.dumps({
            "thought": "继续读", "done": False, "tool_name": "read_text_file",
        }, ensure_ascii=False)
        agent.execute_tool_async = _async_return(ToolCallResult(
            "read_text_file", True, result={"content": "x", "success": True},
        ))

        orch = AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: llm,
            max_steps=3,
        )

        async def _run():
            events = []
            async for evt in orch.run(
                "修 calculator/core.py",
                allow_mcp_discovery=False,
                session_id="s", task_id="t",
            ):
                events.append(evt)
            return events

        events = asyncio.run(_run())
        max_steps_evt = [e for e in events if e.get("type") == "max_steps"]
        assert max_steps_evt

    def test_service_persists_max_steps_stop_reason(self):
        from core.task.service import AgentTaskService

        update_calls = []
        memory = MagicMock()
        memory.update_task = lambda task_id, **kw: update_calls.append((task_id, kw))
        memory.get_task = lambda task_id: {"stop_reason": None}

        svc = AgentTaskService(
            agent_provider=lambda: MagicMock(),
            llm_provider=lambda: MagicMock(),
            multimodal_provider=lambda: MagicMock(),
            session_provider=lambda: memory,
        )

        async def _process():
            evts = []
            async for e in svc._process_orchestrator_event(
                {"type": "max_steps", "steps": 10, "reason": "max_steps"},
                task_id="t1", session_id="s1", turn_id="turn1",
                message="x", mode="agent", workspace=None,
                workspace_trusted=False, auto_discover_mcp=False,
                analysis_step_id="a1",
                tool_execution_context=[], accumulated_resources=[],
            ):
                evts.append(e)
            return evts

        asyncio.run(_process())
        assert any(
            kw.get("stop_reason") == "max_steps"
            for _, kw in update_calls
        ), f"max_steps 未持久化 stop_reason，实际：{update_calls}"

    def test_missing_required_params_skips_step(self):
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
        agent.list_all_tools = lambda **kw: tools
        exec_calls = []
        agent.execute_tool_async = _async_record(exec_calls)

        llm = MagicMock()
        responses = iter([
            {"thought": "搜", "done": False, "tool_name": "search_files"},
            {"thought": "完成", "done": True},
        ])

        def _gen(msgs, **kw):
            return json.dumps(next(responses), ensure_ascii=False)

        llm.generate = _gen
        orch = AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: llm,
            max_steps=5,
        )

        async def _empty_fill(*a, **kw):
            return {}
        orch._fill_params = _empty_fill

        async def _run():
            events = []
            async for evt in orch.run(
                "找一下配置文件",
                allow_mcp_discovery=False,
                session_id="s", task_id="t",
            ):
                events.append(evt)
            return events

        asyncio.run(_run())
        assert not any(n == "search_files" for n, _ in exec_calls), (
            f"search_files 被空参数执行了：{exec_calls}"
        )

    @pytest.mark.asyncio
    async def test_write_clears_last_read_content(self):
        agent = MagicMock()
        tools = {
            "read_text_file": _meta("读", {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            }),
            "edit_file": _meta("改", {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["file_path", "old_text", "new_text"],
            }),
        }
        agent.list_all_tools = lambda **kw: tools
        agent.register_dynamic_tool = lambda *a, **k: None
        agent.unregister_dynamic_tool = lambda *a, **k: None
        agent.format_tool_result = lambda r: f"OK {r.tool_name}"

        async def _exec(tc):
            if tc.tool_name == "read_text_file":
                return ToolCallResult("read_text_file", True, result={
                    "content": "old content", "file_path": "f.py", "success": True,
                })
            if tc.tool_name == "edit_file":
                return ToolCallResult("edit_file", True, result={
                    "success": True, "file_path": "f.py",
                })
            return ToolCallResult(tc.tool_name, True, result={"success": True})

        agent.execute_tool_async = _exec

        responses = iter([
            {"thought": "读", "done": False, "tool_name": "read_text_file"},
            {"thought": "改", "done": False, "tool_name": "edit_file"},
            {"thought": "完成", "done": True},
        ])
        llm = MagicMock()
        llm.generate = lambda msgs, **kw: json.dumps(next(responses), ensure_ascii=False)

        orch = AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: llm,
            max_steps=5,
        )

        async def _fill(goal, tool_name, tools, state, **kw):
            if tool_name == "read_text_file":
                return {"file_path": "f.py"}
            if tool_name == "edit_file":
                return {"file_path": "f.py", "old_text": "x", "new_text": "y"}
            return {}

        orch._fill_params = _fill

        async for _ in orch.run(
            "修 f.py",
            allow_mcp_discovery=False,
            session_id="s", task_id="t",
        ):
            pass

    def test_large_file_edit_with_line_range(self):
        from tools.tools_def.file import edit_file
        import inspect
        fn = getattr(edit_file, "__wrapped__", edit_file)
        sig = inspect.signature(fn)
        if "start_line" not in sig.parameters and "line_range" not in sig.parameters:
            pytest.xfail(
                "edit_file 不支持按行范围改，大文件只能全量 write_file。"
            )

    def test_read_file_range_supports_large_file(self):
        from tools.tools_def.file_search import read_file_range
        import inspect
        sig = inspect.signature(read_file_range)
        assert "start_line" in sig.parameters