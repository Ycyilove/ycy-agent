"""AgentOrchestrator 主类。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

    # ── 精简工具格式（降低 prompt 长度） ──

    def _format_tools_compact(self, tools: Dict[str, Any]) -> str:
        """一行一个工具 + 必填参数名，省去嵌套 schema、enum、examples。"""
        if not tools:
            return "(无可用工具)"

        lines = ["## 可用工具", ""]
        for name, meta in tools.items():
            desc = (getattr(meta, "description", "") or "").split("\n")[0].strip()
            if len(desc) > 90:
                desc = desc[:87] + "..."

            schema = getattr(meta, "input_schema", None) or {}
            properties = schema.get("properties") or getattr(meta, "parameters", None) or {}
            required = list(schema.get("required") or [])
            if not required and properties:
                # 有些工具没标 required，把全部参数名作为提示
                required = list(properties.keys())

            req_str = ", ".join(required[:6]) if required else "(无)"
            if len(required) > 6:
                req_str += f" …(+{len(required) - 6})"

            lines.append(f"- `{name}` — {desc}  [参数: {req_str}]")

        return "\n".join(lines)

    # ── 工具预筛：从 goal 粗筛相关工具 ──

    # 关键词组 → 匹配的工具名/描述关键词
    _INTENT_TO_TOOL_HINTS: List[tuple] = [
        (
            ("创建", "新建", "写入", "写文件", "保存", "生成文件",
             "create", "write", "save", "append"),
            ("write", "create", "edit", "save", "append"),
        ),
        (
            ("读取", "查看", "打开", "read", "open", "cat"),
            ("read", "list", "get", "load", "open"),
        ),
        (
            ("删除", "移除", "delete", "remove", "unlink"),
            ("delete", "remove", "move", "trash"),
        ),
        (
            ("列出", "目录", "文件夹", "list", "dir", "ls"),
            ("list", "dir", "tree", "walk"),
        ),
        (
            ("搜索", "查找", "search", "find", "grep"),
            ("search", "find", "list", "grep"),
        ),
        (
            ("重命名", "改名", "rename", "move", "移动"),
            ("rename", "move", "edit"),
        ),
        (
            ("执行", "运行", "跑", "run", "execute", "python"),
            ("run", "execute", "python", "shell", "command"),
        ),
        (
            ("时间", "日期", "time", "date", "时区"),
            ("time", "date", "timestamp", "clock"),
        ),
        (
            ("http", "url", "网页", "接口", "api", "请求", "抓取", "fetch"),
            ("http", "fetch", "url", "web", "json"),
        ),
        (
            ("csv", "excel", "表格", "spreadsheet"),
            ("csv", "excel", "table", "spreadsheet"),
        ),
        (
            ("mysql", "数据库", "sql", "database"),
            ("mysql", "sql", "query", "database"),
        ),
        (
            ("知识库", "rag", "检索", "knowledge", "文档库"),
            ("knowledge", "rag", "search", "retrieve"),
        ),
        (
            ("mcp", "外部", "发现", "连接", "connect", "discover"),
            ("mcp", "search_and_connect"),
        ),
    ]

    def _shortlist_tools(self, goal: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        """按 goal 关键词粗筛工具。筛不到相关就返回全量。"""
        MIN_KEEP = 8
        lower = goal.lower()

        matched_hints: set = set()
        for triggers, hints in self._INTENT_TO_TOOL_HINTS:
            if any(t in lower for t in triggers):
                matched_hints.update(hints)

        if not matched_hints:
            return tools

        filtered: Dict[str, Any] = {}
        for name, meta in tools.items():
            name_lower = name.lower()
            desc_lower = (getattr(meta, "description", "") or "").lower()
            hay = f"{name_lower} {desc_lower}"
            if any(h in hay for h in matched_hints):
                filtered[name] = meta

        if len(filtered) < MIN_KEEP:
            return tools

        return filtered

    # ── 工具名解析（容错 LLM 幻觉出的简写） ──

    @staticmethod
    def _resolve_tool_name(requested: str, available: Dict[str, Any]) -> Optional[str]:
        """把 LLM 幻觉出的简写工具名映射到实际存在的工具。

        例：write_file → mcp__filesystem__write_file
        唯一匹配才返回；多个匹配返回 None（不猜）。
        """
        if not requested:
            return None
        if requested in available:
            return requested

        # 后缀匹配 __<requested>
        suffix = f"__{requested}"
        matches = [n for n in available if n.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]

        # 包含匹配（唯一才用）
        contains = [n for n in available if requested in n]
        if len(contains) == 1:
            return contains[0]

        return None

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

    def _try_fast_path(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
    ) -> Optional[OrchestratorStep]:
        if not self._enable_intent_router:
            return None
        if state.history:
            # 只在第一步走快速路径；后续步骤状态已经变了，不能套模板
            return None
        if not requires_tool_call(goal):
            return None

        tool_name = route_intent(goal, tools)
        if not tool_name:
            return None

        logger.info(
            "[dsh][orchestrator] fast-path intent → %s", tool_name
        )
        return OrchestratorStep(
            step_index=1,
            thought=f"[快速路径] 意图匹配：{tool_name}",
            tool_name=tool_name,
            parameters={},   # 待 _fill_params 补
            done=False,
        )

    # ── LLM 决策 ──

    async def _decide(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
    ) -> Optional[OrchestratorStep]:
        # 预筛：只把相关工具塞进 prompt
        shortlist = self._shortlist_tools(goal, tools)

        prompt = (
            ORCHESTRATOR_PROMPT
            .replace("{tools}", self._format_tools(shortlist))
            .replace("{history}", self._format_history(state.history, goal, state))
            .replace("{goal}", goal)
            .replace("{max_steps}", str(self._max_steps))
        )
        try:
            logger.info(
                "[dsh][orchestrator] calling LLM, prompt length=%d chars, "
                "tools_in_prompt=%d (total=%d)",
                len(prompt), len(shortlist), len(tools),
            )
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

        # 工具名容错：把 write_file 解析到 mcp__filesystem__write_file
        raw_name = data.get("tool_name")
        resolved_name = raw_name
        if raw_name and raw_name not in tools:
            resolved = self._resolve_tool_name(raw_name, tools)
            if resolved:
                resolved_name = resolved
                logger.info(
                    "[dsh][orchestrator] resolved tool name: %r -> %r",
                    raw_name, resolved_name,
                )

        return OrchestratorStep(
            step_index=len(state.history) + 1,
            thought=str(data.get("thought", "")),
            tool_name=resolved_name,
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

        kind = classify_request(goal)

        if kind == "action":
            # 涉及副作用的请求：必须有写类工具成功执行
            write_succeeded = any(
                self._is_write_operation(name)
                for name in state.succeeded_tool_names
            )
            if not write_succeeded:
                succeeded = ", ".join(state.succeeded_tool_names.keys()) or "(无)"
                return (
                    f"⛔ 校验失败 —— 请求是「修改类」（action），但没有任何写操作成功。\n"
                    f"已成功调用：{succeeded}\n"
                    f"**必须使用 write_file / edit_file / create_directory / "
                    f"move_file / rename_file 等写类工具实际修改文件**。\n"
                    f"不要在回答里告诉用户「你自己改」，而是你直接调用工具改。"
                )

        elif kind == "query":
            if not state.succeeded_tool_names:
                return (
                    "⛔ 校验失败 —— 请求涉及文件读取（query），"
                    "但还没有成功调用任何工具。"
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

        if idem == "none" and state.has_succeeded(tool_name, params):
            prev_step = state.successful_calls[(tool_name, params_key(params))]
            return (
                f"⛔ 非幂等工具 {tool_name} 已在 Step {prev_step} 成功调用过"
                f"（参数完全相同），禁止重复调用。请换策略或 done=true。"
            )

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

    _MCP_DISCOVERY_TOOLS = frozenset({"search_and_connect_mcp"})

    _WORKSPACE_PATH_KEYS = frozenset({
        "directory", "file_path", "old_path", "output_path", "path",
    })

    # 只有写类操作才触发工作区审批；读类操作直接放行
    _WRITE_TOOL_HINTS = (
        "write", "edit", "create", "delete", "remove",
        "move", "rename", "mkdir", "append", "insert",
        "update", "patch", "save", "overwrite",
    )

    @classmethod
    def _is_write_operation(cls, tool_name: str) -> bool:
        lower = tool_name.lower()
        return any(h in lower for h in cls._WRITE_TOOL_HINTS)

    @classmethod
    def _is_path_out_of_workspace(
        cls,
        tool_name: str,
        params: dict,
        workspace: List[str],
    ) -> bool:
        import os
        for key, value in params.items():
            if key not in cls._WORKSPACE_PATH_KEYS:
                continue
            if not isinstance(value, str) or not value:
                continue
            absolute = os.path.abspath(value)
            in_workspace = any(
                absolute == root or absolute.startswith(root + os.sep)
                for root in workspace
            )
            if not in_workspace:
                return True
        return False

    async def _fill_params(
        self,
        goal: str,
        tool_name: str,
        tools: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM 只填参数，不再选工具。

        prompt 只放这一个工具的 schema，省 token。
        """
        meta = tools.get(tool_name)
        if meta is None:
            return {}

        schema = getattr(meta, "input_schema", None) or {}
        properties = schema.get("properties") or getattr(meta, "parameters", None) or {}
        required = list(schema.get("required") or [])
        if not required and properties:
            required = list(properties.keys())

        # 精简 schema：只给参数名 + 类型 + 描述
        param_lines = []
        for pname, pinfo in properties.items():
            if not isinstance(pinfo, dict):
                param_lines.append(f"- {pname}: (任意类型)")
                continue
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            mark = "必填" if pname in required else "可选"
            line = f"- {pname} ({ptype}, {mark})"
            if pdesc:
                line += f": {pdesc}"
            if "enum" in pinfo:
                line += f" [可选值: {pinfo['enum']}]"
            param_lines.append(line)

        schema_text = "\n".join(param_lines) or "(无参数)"

        prompt = (
            f"任务：{goal}\n\n"
            f"你要调用的工具是 `{tool_name}`。\n"
            f"工具描述：{getattr(meta, 'description', '')}\n\n"
            f"参数表：\n{schema_text}\n\n"
            f"请只输出这个工具的 JSON 参数对象，不要任何解释，不要包裹在 markdown 里。\n"
            f"示例：{{\"file_path\": \"D:/a.txt\", \"content\": \"hello\"}}\n\n"
            f"只输出 JSON："
        )

        try:
            logger.info(
                "[dsh][orchestrator] filling params for %s, prompt length=%d",
                tool_name, len(prompt),
            )
            raw = await asyncio.to_thread(
                self._llm_provider().generate,
                [{"role": "user", "content": prompt}],
            )
            logger.info(
                "[dsh][orchestrator] params raw response: %r", raw
            )
        except Exception:
            logger.exception("Param filling LLM call failed")
            return {}

        data = self._parse_json(raw)
        if not isinstance(data, dict):
            logger.warning("Param filling returned non-JSON: %s", raw[:200])
            return {}
        return data
    
    async def run(
        self,
        goal: str,
        allowed_danger_levels: Optional[set] = None,
        allowed_tool_names: Optional[set] = None,
        allow_mcp_discovery: bool = True,
        workspace_paths: Optional[List[str]] = None,
        plan_mode: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        # 每个任务开始重置"已试过的 MCP 服务器"
        try:
            from tools.mcp_discovery import reset_tried_servers
            reset_tried_servers()
        except Exception:
            logger.exception("failed to reset tried MCP servers")

        agent = self._agent_provider()
        tools = agent.list_all_tools(allowed_danger_levels=allowed_danger_levels)
        logger.info(
            "[dsh][orchestrator] allowed_dangers=%r, filtered tools count=%d, "
            "has write_file=%r",
            allowed_danger_levels,
            len(tools),
            "write_file" in tools,
        )

        # 关闭自动发现时，隐藏所有 MCP discovery 工具
        if not allow_mcp_discovery:
            tools = {
                name: meta for name, meta in tools.items()
                if name not in self._MCP_DISCOVERY_TOOLS
            }
            logger.info(
                "[dsh][orchestrator] MCP discovery disabled; hid tools=%r",
                sorted(self._MCP_DISCOVERY_TOOLS),
            )

        if not tools:
            yield {"type": "error", "message": "当前模式下没有可用的工具"}
            return

        state = OrchestratorState(goal=goal)

        for step_index in range(1, self._max_steps + 1):
            decision = self._try_fast_path(goal, state, tools)
            if decision is None:
                decision = await self._decide(goal, state, tools)

            if decision is None:
                yield {"type": "error", "message": "编排决策失败"}
                return

            # 快速路径选出了工具但没给参数 → 补一次
            if (
                decision.tool_name
                and not decision.parameters
                and not decision.done
            ):
                decision.parameters = await self._fill_params(
                    goal, decision.tool_name, tools
                )
                logger.info(
                    "[dsh][orchestrator] filled params for %s: %r",
                    decision.tool_name, decision.parameters,
                )

            yield {
                "type": "thinking",
                "step": step_index,
                "thought": decision.thought,
                "log": decision.thought,
            }

            if decision.done:
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
                yield {
                    "type": "thinking",
                    "step": step_index,
                    "thought": "[决策错误] 模型未给出工具名，本步忽略",
                }
                continue

            if tool_name not in tools:
                available = ", ".join(list(tools.keys())[:10])
                state.add_history(
                    f"Step {step_index}: 尝试调用无效工具 {tool_name}，已忽略。"
                    f"可用工具（前 10 个）：{available}..."
                )
                logger.warning(
                    "[dsh][orchestrator] unresolved tool name %r; available=%r",
                    tool_name, list(tools.keys()),
                )
                yield {
                    "type": "thinking",
                    "step": step_index,
                    "thought": (
                        f"[决策错误] 模型选择了不存在的工具 {tool_name}，已忽略。"
                        f"可用工具：{available}"
                    ),
                }
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

            # plan 模式下，路径越界也要审批
            needs_workspace_approval = False
            if (
                plan_mode
                and workspace_paths
                and self._is_write_operation(tool_name)
            ):
                needs_workspace_approval = self._is_path_out_of_workspace(
                    tool_name, params, workspace_paths
                )
                logger.info(
                    "[dsh][orchestrator] workspace check: tool=%s params=%r "
                    "workspace=%r → needs_approval=%r",
                    tool_name, params, workspace_paths, needs_workspace_approval,
                )

            if needs_workspace_approval:
                tool_call = ToolCallRequest(
                    tool_name=tool_name,
                    parameters=params,
                    confidence=1.0,
                    reasoning=decision.thought,
                    user_confirmation_needed=True,
                )
                yield {
                    "type": "approval_required",
                    "step": step_index,
                    "tool_call": tool_call,
                    "reason": "路径不在工作区内",
                }
                return

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