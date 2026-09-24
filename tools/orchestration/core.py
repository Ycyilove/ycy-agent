"""AgentOrchestrator 主类。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from tools.agent import (
    ToolAgent,
    ToolCallRequest,
    ToolCallResult,
)
from tools.idempotency import classify_tool_idempotency
from tools.semantics import (
    get_intent_tags,
    get_intent_tag_description,
    INTENT_TAG_DESCRIPTIONS,
)

from .prompt import ORCHESTRATOR_PROMPT
from .state import OrchestratorState, params_key
from .validator import requires_tool_call, classify_request
from .intent_router import route_intent


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorStep:
    step_index: int
    thought: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    done: bool = False


class AgentOrchestrator:
    """Multi-step tool orchestration loop with strategies A/B/D/E."""

    def __init__(
        self,
        agent_provider: Callable[[], ToolAgent],
        llm_provider: Callable[[], Any],
        max_steps: int = 10,
        enable_intent_router: bool = True,
    ):
        self._agent_provider = agent_provider
        self._llm_provider = llm_provider
        self._max_steps = max_steps
        self._enable_intent_router = enable_intent_router

    # ── 工具格式化（策略 A：带用途标签） ──

    def _format_tools(self, tools: Dict[str, Any]) -> str:
        if not tools:
            return "(无可用工具)"

        lines: List[str] = []

        # 顶部：用途标签索引
        tag_index: Dict[str, List[str]] = {}
        for name in tools:
            for tag in get_intent_tags(name):
                tag_index.setdefault(tag, []).append(name)
        if tag_index:
            lines.append("## 按用途标签索引")
            for tag in sorted(tag_index):
                desc = INTENT_TAG_DESCRIPTIONS.get(tag, tag)
                tools_for_tag = tag_index[tag][:5]
                lines.append(f"- **{desc}** ({tag}): {', '.join(tools_for_tag)}")
            lines.append("")

        # 逐个工具详情
        lines.append("## 工具详情")
        for name, meta in tools.items():
            lines.append("")
            lines.append(f"### {name}")
            lines.append(f"描述: {meta.description}")

            tags = get_intent_tags(name)
            if tags:
                tag_descs = [get_intent_tag_description(t) for t in tags]
                lines.append(f"🏷️ 用途标签: {'、'.join(tag_descs)}")

            schema = getattr(meta, "input_schema", None) or {}
            required = set(schema.get("required") or [])
            properties = schema.get("properties") or getattr(meta, "parameters", None) or {}

            if not properties:
                lines.append("参数: 无")
                continue

            lines.append("参数:")
            for pname, pinfo in properties.items():
                if not isinstance(pinfo, dict):
                    lines.append(f"  - {pname}: (任意类型)")
                    continue

                ptype = pinfo.get("type", "any")
                mark = "必填" if pname in required else "可选"
                pdesc = pinfo.get("description", "")
                part = f"  - {pname} ({ptype}, {mark})"
                if pdesc:
                    part += f": {pdesc}"
                if "enum" in pinfo and isinstance(pinfo["enum"], list):
                    part += f" [可选值: {pinfo['enum']}]"
                if "default" in pinfo:
                    part += f" [默认: {pinfo['default']}]"
                lines.append(part)

                nested = pinfo.get("properties")
                if isinstance(nested, dict) and nested:
                    nested_required = set(pinfo.get("required") or [])
                    for nname, ninfo in nested.items():
                        if not isinstance(ninfo, dict):
                            continue
                        ntype = ninfo.get("type", "any")
                        nmark = "必填" if nname in nested_required else "可选"
                        ndesc = ninfo.get("description", "")
                        sub = f"      - {nname} ({ntype}, {nmark})"
                        if ndesc:
                            sub += f": {ndesc}"
                        lines.append(sub)

                items = pinfo.get("items")
                if isinstance(items, dict):
                    itype = items.get("type", "any")
                    lines.append(f"      (数组元素类型: {itype})")

            required_params = [p for p in properties if p in required]
            if required_params:
                lines.append(
                    f"  ⚠️ **必填参数**：{', '.join(required_params)}"
                )

        return "\n".join(lines)

    def _format_history(self, history: List[str], goal: str, state: OrchestratorState) -> str:
        reminder = (
            f"## 提醒\n"
            f"你的当前目标是：{goal}\n"
            f"如果某条路径明显偏离这个目标，不要继续沿着它走。\n\n"
            f"## 最近已尝试\n"
            f"{state.describe_recent_attempts()}"
        )
        if not history:
            return f"(尚无步骤)\n\n{reminder}"
        body = "\n".join(history)
        return f"{body}\n\n{reminder}"

    # ── 解析 JSON ──

    def _parse_json(self, text: str) -> Optional[dict]:
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(clean[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

    # ── 快速路径（策略 D） ──

    def _try_fast_path(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
    ) -> Optional[OrchestratorStep]:
        if not self._enable_intent_router:
            return None
        if state.history:
            # 只在第一步走快速路径
            return None
        if not requires_tool_call(goal):
            return None

        tool_name = route_intent(goal, tools)
        if not tool_name:
            return None

        logger.info("[dsh][orchestrator] fast-path: intent matched → %s", tool_name)
        return OrchestratorStep(
            step_index=1,
            thought=f"[快速路径] 意图匹配，直接选 {tool_name}",
            tool_name=tool_name,
            parameters={},  # 参数由 LLM 或后续规则填
            done=False,
        )

    # ── LLM 决策 ──

    async def _decide(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
    ) -> Optional[OrchestratorStep]:
        prompt = ORCHESTRATOR_PROMPT.format(
            tools=self._format_tools(tools),
            history=self._format_history(state.history, goal, state),
            goal=goal,
            max_steps=self._max_steps,
        )
        try:
            response = await asyncio.to_thread(
                self._llm_provider().generate,
                [{"role": "user", "content": prompt}],
            )
            logger.info("[dsh][orchestrator] raw LLM response: %r", response)
        except Exception:
            logger.exception("Orchestrator LLM call failed")
            return None

        data = self._parse_json(response)
        if data is None:
            logger.warning("Orchestrator returned non-JSON: %s", response[:200])
            return None

        return OrchestratorStep(
            step_index=len(state.history) + 1,
            thought=str(data.get("thought", "")),
            tool_name=data.get("tool_name"),
            parameters=data.get("parameters") or {},
            done=bool(data.get("done", False)),
        )

    # ── 决策校验（策略 B） ──

    def _validate_done(
        self,
        goal: str,
        state: OrchestratorState,
    ) -> Optional[str]:
        """如果 done=true 不合理，返回拒绝原因（字符串）；合理则返回 None。"""
        if not requires_tool_call(goal):
            return None

        # 请求需要工具，但还没成功调用过任何工具
        if not state.succeeded_tool_names:
            return (
                "⛔ 校验失败 —— 请求涉及副作用/文件读取，但你的决策是 done=true。"
                "**必须调用对应工具**。查看可用工具列表，选择匹配的工具执行操作。"
                "不要在回答里演示如何做，而是实际调用工具。"
            )
        return None

    # ── 调用拦截（策略 E） ──

    def _intercept_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        state: OrchestratorState,
        tools: Dict[str, Any],
    ) -> Optional[str]:
        """返回拦截原因（字符串）；允许则返回 None。"""
        metadata = tools.get(tool_name)
        idem = classify_tool_idempotency(tool_name, metadata)

        # 拦截 1：非幂等 + 相同参数 + 之前成功过
        if idem == "none" and state.has_succeeded(tool_name, params):
            prev_step = state.successful_calls[(tool_name, params_key(params))]
            return (
                f"⛔ 非幂等工具 {tool_name} 已在 Step {prev_step} 成功调用过"
                f"（参数完全相同），禁止重复调用。请换策略或 done=true。"
            )

        # 拦截 2：条件幂等 + 相邻相同参数
        if (
            idem == "conditional"
            and state.last_successful_call_key is not None
            and (tool_name, params_key(params)) == state.last_successful_call_key
        ):
            return (
                f"⚠️ 条件幂等工具 {tool_name} 与上一步成功调用完全相同"
                f"（参数相同，中间无其他成功调用），已跳过。使用上一步的结果。"
            )

        return None

    # ── 状态标记 ──

    @staticmethod
    def _classify_status(log: str, success: bool) -> str:
        if "⛔" in log or "❌ 工具调用被 MCP 服务器拒绝" in log:
            return "MANDATORY_NEXT"
        if "💡 MCP 服务器附带了一个可选建议" in log:
            return "OPTIONAL_HINT"
        return "OK" if success else "FAIL"

    # ── 主循环 ──

    # orchestrator 里通过名字识别"需要网络发现"的工具。
    # 关闭自动发现时，这些工具会从可用工具集里被剔除，
    # 模型看不到它们，也就不会触发外部 MCP 注册。
    _MCP_DISCOVERY_TOOLS = frozenset({"search_and_connect_mcp"})

    async def run(
        self,
        goal: str,
        allowed_danger_levels: Optional[set] = None,
        allowed_tool_names: Optional[set] = None,
        allow_mcp_discovery: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        # 每个任务开始重置"已试过的 MCP 服务器"
        try:
            from tools.mcp_discovery import reset_tried_servers
            reset_tried_servers()
        except Exception:
            logger.exception("failed to reset tried MCP servers")

        agent = self._agent_provider()
        tools = agent.list_all_tools(allowed_danger_levels=allowed_danger_levels)

        # 关闭自动发现时，隐藏所有 MCP discovery 工具
        if not allow_mcp_discovery:
            tools = {
                name: meta for name, meta in tools.items()
                if name not in self._MCP_DISCOVERY_TOOLS
            }
            logger.info(
                "[dsh][orchestrator] MCP discovery disabled; "
                "hid tools=%r",
                sorted(self._MCP_DISCOVERY_TOOLS),
            )

        if not tools:
            yield {"type": "error", "message": "当前模式下没有可用的工具"}
            return

        state = OrchestratorState(goal=goal)

        for step_index in range(1, self._max_steps + 1):
            # ── 策略 D：先尝试快速路径 ──
            decision = self._try_fast_path(goal, state, tools)
            if decision is None:
                decision = await self._decide(goal, state, tools)

            if decision is None:
                yield {"type": "error", "message": "编排决策失败"}
                return

            yield {
                "type": "thinking",
                "step": step_index,
                "thought": decision.thought,
            }

            if decision.done:
                # ── 策略 B：校验 done 是否合理 ──
                reject = self._validate_done(goal, state)
                if reject:
                    state.add_history(f"Step {step_index}: {reject}")
                    yield {
                        "type": "thinking",
                        "step": step_index,
                        "thought": "[客户端校验] 拒绝 done=true",
                    }
                    continue
                yield {"type": "done", "steps": step_index - 1}
                return

            tool_name = decision.tool_name
            if not tool_name:
                state.add_history(f"Step {step_index}: 模型未给出工具名，已忽略")
                continue

            is_mcp_tool = (
                tool_name.startswith("mcp__")
                and "__" in tool_name[5:]
            )
            if tool_name not in tools and not is_mcp_tool:
                available = ", ".join(list(tools.keys())[:10])
                state.add_history(
                    f"Step {step_index}: 尝试调用无效工具 {tool_name}，已忽略。"
                    f"可用工具（前 10 个）：{available}..."
                )
                continue

            params = decision.parameters or {}

            # ── 策略 E：拦截重复调用 ──
            intercept = self._intercept_call(tool_name, params, state, tools)
            if intercept:
                state.add_history(f"Step {step_index}: {intercept}")
                logger.info("[dsh][orchestrator] intercepted: %s", intercept)
                yield {
                    "type": "thinking",
                    "step": step_index,
                    "thought": f"[客户端拦截] {intercept}",
                }
                continue

            state.mark_attempt()

            tool_call = ToolCallRequest(
                tool_name=tool_name,
                parameters=params,
                confidence=1.0,
                reasoning=decision.thought,
                user_confirmation_needed=False,
            )
            yield {
                "type": "tool_start",
                "step": step_index,
                "tool_call": tool_call,
            }

            try:
                result = await agent.execute_tool_async(tool_call)
            except Exception as error:
                logger.exception("Tool execution failed in orchestrator")
                result = ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(error),
                )

            log = agent.format_tool_result(result)
            status = self._classify_status(log, result.success)
            state.add_history(
                f"Step {step_index}: {tool_name}({params}) "
                f"→ {status}: {log[:400]}"
            )

            if result.success:
                state.mark_success(tool_name, params, step_index)
            else:
                state.mark_failure(tool_name, params, result.error or "")

            yield {
                "type": "tool_result",
                "step": step_index,
                "tool_call": tool_call,
                "result": result,
                "log": log,
            }

        yield {"type": "max_steps", "steps": self._max_steps}