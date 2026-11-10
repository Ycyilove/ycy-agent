"""AgentOrchestrator 的评估 Mixin：结果评估 / 完成度校验 / 验证收集。

从 core.py 按行号提取（B8-c-1），方法体零改动。
依赖：
    - _goal_needs_verification（仍留在 core.py，run 里调用）
    - _read_tool_schema / _format_discovery_schema_brief（core.py）
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .state import OrchestratorState
from tools.types import ToolCallResult
from .helpers import _is_read_like
from .task_types import AssessResult, GapKind, StopReason
from .audit import run_audit, collect_evidence
from .validator import requires_tool_call

logger = logging.getLogger(__name__)
class AssessMixin:
    """评估相关方法。self 是 AgentOrchestrator。"""
    def _validate_done(
        self,
        goal: str,
        state: OrchestratorState,
    ) -> Optional[str]:
        """如果 done=true 不合理，返回拒绝原因；合理则返回 None。"""
        if not requires_tool_call(goal):
            return None

        if not state.succeeded_tool_names:
            logger.info(
                "[dsh][orchestrator] done rejected: requires_tool_call=True "
                "but no tool succeeded yet (goal=%r)",
                goal[:80],
            )
            return (
                "⛔ 校验失败 —— 这个请求必须调用工具才能完成，"
                "但你还没有成功执行任何工具。\n"
                "**下一步：从工具目录里选一个能完成目标的工具并调用**，"
                "不要只描述「可以用 X 完成」。\n"
                "**知道 ≠ 做过。**只有工具实际执行并返回结果后，才允许 done=true。"
            )

        audit = run_audit(goal, state, evidence=collect_evidence(state))
        if audit.satisfied:
            return None

        logger.info(
            "[dsh][orchestrator] done rejected by audit: kind=%s gap=%r",
            audit.gap_kind.value, audit.gap[:120],
        )
        kind_hint = {
            GapKind.MISSING_CONTENT: "**下一步：读取/检索缺失内容**",
            GapKind.MISSING_ACTION: "**下一步：执行写/改操作**",
            GapKind.MISSING_VERIFICATION: "**下一步：运行验证（测试/检查）**",
            GapKind.BLOCKED: "**外部阻塞，需 done=true 并如实告知用户**",
            GapKind.NONE: "",
        }.get(audit.gap_kind, "")
        return (
            f"⛔ 校验失败 —— 目标尚未完成。\n"
            f"缺失要求：{audit.gap}\n"
            f"{kind_hint}\n"
            f"{audit.next_hint}"
        )


    def _find_verify_tool(cls, tools: Dict[str, Any]) -> Optional[str]:
        for name in tools:
            low = name.lower()
            if any(k in low for k in cls._VERIFY_TOOL_HINTS):
                return name
        return None

    _VERIFY_PATTERN = (
        r"(测试|tests?|验证|确保.{0,8}通过|make.{0,8}pass|run.{0,8}test|"
        r"检查.{0,8}(通过|内容|包含|是否|结果)|"
        r"确认.{0,8}(返回|成功|通过|不报错|入库|存在|结果)|"
        r"启动.{0,8}确认|查询.{0,8}确认|"
        r"npm\s+test|jest|vitest|mocha|"
        r"verify|validate|confirm|check.{0,8}(pass|success|200|ok)|"
        r"查一下|查询一下|搜一下|最新状态|最新情况|当前状态)"
    )


    async def _collect_verification(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
    ) -> Optional[AssessResult]:
        verify_tool = self._find_verify_tool(tools)
        if not verify_tool:
            return AssessResult(
                sufficient=False,
                stop_reason=StopReason.NEEDS_USER,
                gap=(
                    "goal 要求验证，但当前工具集里没有可用的执行类工具"
                    "（run_python / run_shell / bash / pytest 等）"
                ),
                gap_kind=GapKind.MISSING_VERIFICATION,
                next_hint=(
                    "**下一步**：如果可用工具确实无法验证，done=true 并如实告知用户"
                    "「已修改但无法自动验证」；否则换一个能执行命令的工具。"
                ),
            )

        exec_tools = [
            name for name in tools
            if any(k in name.lower() for k in self._VERIFY_TOOL_HINTS)
        ]
        exec_hint = "、".join(f"`{n}`" for n in exec_tools) or f"`{verify_tool}`"

        return AssessResult(
            sufficient=False,
            stop_reason=StopReason.NEEDS_USER,
            gap=(
                f"goal 要求验证，但尚未执行任何验证动作。"
                f"可用执行类工具：{exec_hint}"
            ),
            gap_kind=GapKind.MISSING_VERIFICATION,
            next_hint=(
                f"**下一步**：用执行类工具（优先 {exec_hint}）"
                f"跑验证命令。例如：\n"
                f"- Python 项目：`pytest` / `python -m unittest`\n"
                f"- JS 项目：`npm test` / `jest`\n"
                f"- API：`curl -s -o /dev/null -w '%{{http_code}}' <url>`\n"
                f"- 文档/数据：读回文件检查内容\n"
                f"验证通过后再 done=true。"
            ),
        )

    async def _assess_result(
        self,
        goal: str,
        tool_name: str,
        params: Dict[str, Any],
        result: ToolCallResult,
        state: OrchestratorState,
    ) -> Optional[AssessResult]:
        if not requires_tool_call(goal):
            return AssessResult(sufficient=True, stop_reason=StopReason.DONE)

        from tools.security.gateway import SecurityGateway

        if not result.success:
            return AssessResult(
                sufficient=False,
                stop_reason=StopReason.NEEDS_USER,
                gap=f"工具 {tool_name} 失败：{result.error or ''}",
                gap_kind=GapKind.MISSING_ACTION,
                next_hint=(
                    "**下一步**：读错误原因。如果是「未安装/not installed/"
                    "not found in PATH」，该工具在当前环境不可用，"
                    "换工具或 done=true。其他错误可换参数重试一次。"
                ),
            )

        if SecurityGateway.is_write_operation(tool_name, None, params):
            # 产物记录已由 run.py 统一处理，这里不再重复添加
            pass
            audit = run_audit(goal, state, evidence=collect_evidence(state))
            if audit.satisfied:
                return AssessResult(
                    sufficient=True, stop_reason=StopReason.DONE,
                )
            return AssessResult(
                sufficient=False,
                stop_reason=StopReason.NEEDS_USER,
                gap=audit.gap,
                gap_kind=audit.gap_kind,
                next_hint=audit.next_hint,
            )

        if any(k in tool_name.lower() for k in self._VERIFY_TOOL_HINTS):
            try:
                payload = result.result
                passed = bool(
                    isinstance(payload, dict)
                    and payload.get("success")
                )
                state.verification_evidence = {
                    "tool": tool_name,
                    "success": passed,
                    "payload": payload if isinstance(payload, dict) else {},
                }
                logger.info(
                    "[dsh][orchestrator] verification evidence recorded: "
                    "tool=%s success=%r",
                    tool_name, passed,
                )
            except Exception:
                logger.exception("record verification evidence failed")

        if result.success:
            if "read" in (tool_name or "").lower() or _is_read_like(tool_name):
                p = self._extract_path_from_params(params)
                if p:
                    state.read_paths.add(p)
            audit = run_audit(goal, state, evidence=collect_evidence(state))
            if audit.satisfied:
                return AssessResult(
                    sufficient=True, stop_reason=StopReason.DONE,
                )
            return AssessResult(
                sufficient=False,
                stop_reason=StopReason.NEEDS_USER,
                gap=audit.gap,
                gap_kind=audit.gap_kind,
                next_hint=audit.next_hint,
            )

        return None

    _OUTPUT_PATH_KEYS = (
        "file_path", "path", "output_path", "new_path", "old_path",
        "filename", "file", "output", "save_path", "target_path",
        "destination", "dest", "out_path",
    )


    @classmethod
    def _goal_needs_verification(cls, goal: str) -> bool:
        if not goal:
            return False
        import re as _re
        return bool(_re.search(cls._VERIFY_PATTERN, goal, _re.I))


