"""AgentOrchestrator 主类。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from backend.services.llm_debug_log import dump_llm_call

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
        shortlister_provider: Optional[Callable[[], Any]] = None,
    ):
        self._agent_provider = agent_provider
        self._llm_provider = llm_provider
        self._max_steps = max_steps
        self._enable_intent_router = enable_intent_router
        self._shortlister_provider = shortlister_provider

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

    def _format_tool_directory(self, tools: Dict[str, Any]) -> str:
        """只给工具名 + 一句描述 + tags，用于「选工具」阶段。

        不带参数 schema，避免干扰 LLM 的语义匹配。
        """
        if not tools:
            return "(无可用工具)"

        lines = []
        for name, meta in tools.items():
            desc = (getattr(meta, "description", "") or "").split("\n")[0].strip()
            if len(desc) > 70:
                desc = desc[:67] + "..."

            tags = getattr(meta, "intent_tags", None) or []
            tag_str = ""
            if tags:
                tag_str = f"  [tags: {','.join(str(t) for t in tags[:4])}]"

            # 内置工具打 ⭐，让 LLM 倾向选内置
            marker = "" if name.startswith("mcp__") else "⭐ "

            lines.append(f"- {marker}`{name}`{tag_str}\n    {desc}")

        return "\n".join(lines)

    def _format_single_tool_schema(self, meta: Any) -> str:
        """给单个工具的完整 schema，用于「填参数」阶段。

        包括嵌套结构、enum、默认值、必填标记。
        """
        schema = getattr(meta, "input_schema", None) or {}
        properties = (
            schema.get("properties")
            or getattr(meta, "parameters", None)
            or {}
        )
        required = set(schema.get("required") or [])

        if not properties:
            return "(该工具无参数)"

        lines = []
        for pname, pinfo in properties.items():
            if not isinstance(pinfo, dict):
                lines.append(f"- {pname}: (任意类型)")
                continue

            ptype = pinfo.get("type", "any")
            mark = "必填" if pname in required else "可选"
            pdesc = pinfo.get("description", "")

            line = f"- `{pname}` ({ptype}, {mark})"
            if pdesc:
                line += f": {pdesc}"
            if "enum" in pinfo and isinstance(pinfo["enum"], list):
                line += f"\n    可选值: {pinfo['enum']}"
            if "default" in pinfo:
                line += f"\n    默认: {pinfo['default']}"
            lines.append(line)

            # 嵌套对象：展示 1 层
            nested = pinfo.get("properties")
            if isinstance(nested, dict) and nested:
                nested_required = set(pinfo.get("required") or [])
                for nname, ninfo in nested.items():
                    if not isinstance(ninfo, dict):
                        continue
                    ntype = ninfo.get("type", "any")
                    nmark = "必填" if nname in nested_required else "可选"
                    ndesc = ninfo.get("description", "")
                    sub = f"    - `{nname}` ({ntype}, {nmark})"
                    if ndesc:
                        sub += f": {ndesc}"
                    lines.append(sub)

            # 数组元素
            items = pinfo.get("items")
            if isinstance(items, dict):
                itype = items.get("type", "any")
                inested = items.get("properties")
                lines.append(f"    数组元素类型: {itype}")
                if isinstance(inested, dict) and inested:
                    for iname, iinfo in inested.items():
                        if not isinstance(iinfo, dict):
                            continue
                        itype2 = iinfo.get("type", "any")
                        idesc = iinfo.get("description", "")
                        sub = f"      - `{iname}` ({itype2})"
                        if idesc:
                            sub += f": {idesc}"
                        lines.append(sub)

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

    # 内置工具覆盖列表：内置版本存在时，同名 MCP 版本可能参数不同
    _BUILTIN_EDIT_TOOLS = frozenset({"edit_file"})

    @staticmethod
    def _adapt_params_for_tool(
        tool_name: str,
        params: dict,
        available: Dict[str, Any],
    ) -> dict:
        """按工具 schema 适配参数名。

        处理 LLM 混淆「内置 edit_file」和「MCP filesystem edit_file」的情况：
        LLM 选了内置 `edit_file`，但传了 MCP 格式的 `path` + `edits`。
        这里自动转成内置格式 `file_path` + `old_text` + `new_text`。
        """
        if not isinstance(params, dict):
            return params

        meta = available.get(tool_name)
        if meta is None:
            return params

        schema = getattr(meta, "input_schema", None) or {}
        allowed = set((schema.get("properties") or {}).keys())
        if not allowed:
            return params

        # 参数已全部合法：不动
        if set(params.keys()).issubset(allowed):
            return params

        # ── 特化规则：内置 edit_file 收到 MCP 格式 ──
        if tool_name in AgentOrchestrator._BUILTIN_EDIT_TOOLS:
            if "path" in params and "edits" in params:
                edits = params.get("edits") or []
                if isinstance(edits, list) and edits:
                    first = edits[0]
                    if isinstance(first, dict):
                        old_text = (
                            first.get("oldText")
                            or first.get("old_text")
                            or ""
                        )
                        new_text = (
                            first.get("newText")
                            or first.get("new_text")
                            or ""
                        )
                        return {
                            "file_path": params.get("path"),
                            "old_text": old_text,
                            "new_text": new_text,
                        }

        # ── 通用规则：path → file_path ──
        result = dict(params)
        if "file_path" in allowed and "path" in result:
            result["file_path"] = result.pop("path")

        # ── 通用规则：oldText/newText → old_text/new_text ──
        if "old_text" in allowed and "oldText" in result:
            result["old_text"] = result.pop("oldText")
        if "new_text" in allowed and "newText" in result:
            result["new_text"] = result.pop("newText")

        return result

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

    def _select_shortlist(
        self, goal: str, tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """优先用 embedding 筛，失败退到关键词。"""
        # 1. embedding
        if self._shortlister_provider is not None:
            try:
                sl = self._shortlister_provider()
                if sl is not None:
                    picked = sl.shortlist(goal, tools)
                    if picked is not None:
                        return picked
            except Exception:
                logger.exception(
                    "[dsh][orchestrator] shortlister raised, fallback to keyword"
                )

        # 2. 关键词
        return self._shortlist_tools(goal, tools)

    async def _decide(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
        prior_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[OrchestratorStep]:
        # 预筛：优先 embedding，退关键词
        shortlist = self._select_shortlist(goal, tools)

        # 对话历史（最近 N 条）——让 LLM 知道用户前面说了什么
        conversation_block = ""
        if prior_messages:
            recent = prior_messages[-6:]
            lines = []
            for m in recent:
                role = m.get("role", "user")
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                # 截断单条消息，避免 prompt 爆
                if len(content) > 400:
                    content = content[:400] + "..."
                lines.append(f"- {role}: {content}")
            if lines:
                conversation_block = (
                    "## 对话历史（最近几条）\n" + "\n".join(lines) + "\n\n"
                )

        # 如果之前读过文件，把它放进 prompt 的 history 部分
        history_text = self._format_history(state.history, goal, state)
        if state.last_read_content:
            path = state.last_read_path or "(未知)"
            content = state.last_read_content
            if len(content) > 4000:
                content = content[:4000] + "\n... (truncated)"
            history_text += (
                f"\n\n## 最近读取的文件（已缓存）\n"
                f"路径：`{path}`\n"
                f"内容：\n```\n{content}\n```\n"
            )

        prompt = (
            ORCHESTRATOR_PROMPT
            .replace("{tool_directory}", self._format_tool_directory(shortlist))
            .replace("{history}", history_text)
            .replace("{goal}", goal)
            .replace("{max_steps}", str(self._max_steps))
        )
        # 把对话历史插入 prompt 头部（ORCHESTRATOR_PROMPT 里没有占位符）
        if conversation_block:
            prompt = conversation_block + prompt
        import time as _time
        _t0 = _time.time()

        try:
            logger.info(
                "[dsh][orchestrator] _decide calling LLM, prompt length=%d chars, "
                "tools_in_directory=%d (total=%d)",
                len(prompt), len(shortlist), len(tools),
            )
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._llm_provider().generate,
                    [{"role": "user", "content": prompt}],
                ),
                timeout=120.0,
            )
            duration_ms = (_time.time() - _t0) * 1000
            logger.info("[dsh][orchestrator] raw LLM response: %r", response)

            # 落盘
            dump_llm_call(
                kind="decide",
                prompt=prompt,
                response=response,
                model=getattr(self._llm_provider(), "model", None),
                duration_ms=duration_ms,
                extra={
                    "tools_in_prompt": list(shortlist.keys()),
                    "tools_total": len(tools),
                },
            )
        except asyncio.TimeoutError:
            duration_ms = (_time.time() - _t0) * 1000
            logger.warning("[dsh][orchestrator] _decide LLM timed out after 120s")
            dump_llm_call(
                kind="decide",
                prompt=prompt,
                response=None,
                duration_ms=duration_ms,
                error="timeout(120s)",
                extra={"tools_in_prompt": list(shortlist.keys())},
            )
            return None
        except asyncio.CancelledError:
            logger.warning("[dsh][orchestrator] _decide cancelled")
            raise
        except Exception as exc:
            duration_ms = (_time.time() - _t0) * 1000
            logger.exception("Orchestrator LLM call failed")
            dump_llm_call(
                kind="decide",
                prompt=prompt,
                response=None,
                duration_ms=duration_ms,
                error=str(exc),
            )
            return None

        data = self._parse_json(response)
        if data is None:
            logger.warning("_decide non-JSON: %s", response[:200])
            return None

        # 工具名容错
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
            parameters={},   # ← 参数一律交给 _fill_params
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
        state: Optional[OrchestratorState] = None,
        prior_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """阶段 2：给单个工具填参数。

        prompt 里只放这一个工具的完整 schema；带 history 让 LLM 知道上下文。
        """
        meta = tools.get(tool_name)
        if meta is None:
            logger.warning(
                "[dsh][orchestrator] _fill_params: %s not in tools", tool_name
            )
            return {}

        schema_text = self._format_single_tool_schema(meta)
        tool_desc = getattr(meta, "description", "") or ""

        # history
        history_text = ""
        if state is not None and state.history:
            history_text = (
                "## 已执行步骤\n"
                + "\n".join(state.history[-5:])
                + "\n\n"
            )

        # ── 编辑类工具：把最近读到的文件内容拼进 prompt ──
        NEEDS_FILE_CONTENT_TOOLS = {
            "edit_file", "write_file",
            "mcp__filesystem__edit_file", "mcp__filesystem__write_file",
        }
        file_content_block = ""
        if (
            tool_name in NEEDS_FILE_CONTENT_TOOLS
            and state is not None
            and state.last_read_content
        ):
            content = state.last_read_content
            if len(content) > 4000:
                content = content[:4000] + "\n... (truncated)"
            path = state.last_read_path or "(未知)"
            file_content_block = (
                f"## 最近读取的文件内容\n"
                f"路径：`{path}`\n"
                f"内容（**`old_text` 必须逐字匹配这里的内容**）：\n"
                f"```\n{content}\n```\n\n"
            )

        # 对话历史（最近 6 条）——关键：让 LLM 知道用户上句说的文件在哪
        conversation_block = ""
        if prior_messages:
            recent = prior_messages[-6:]
            lines = []
            for m in recent:
                role = m.get("role", "user")
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if len(content) > 400:
                    content = content[:400] + "..."
                lines.append(f"- {role}: {content}")
            if lines:
                conversation_block = (
                    "## 对话历史（最近几条）\n"
                    + "\n".join(lines)
                    + "\n\n"
                )

        # ── Mem0 检索：本会话已知文件路径 ──
        mem0_block = ""
        if state is not None and getattr(state, "session_id", None):
            try:
                from backend.services.memory_layer import recall_file_paths
                known_paths = recall_file_paths(state.session_id, limit=5)
                if known_paths:
                    mem0_block = (
                        "## 已知文件路径（本会话历史）\n"
                        + "\n".join(f"- `{p}`" for p in known_paths)
                        + "\n\n"
                    )
                    logger.info(
                        "[dsh][orchestrator] mem0 recalled %d paths for session %s",
                        len(known_paths), state.session_id,
                    )
            except Exception:
                logger.exception("recall_file_paths 失败")

        prompt = (
            f"## 任务\n{goal}\n\n"
            f"{conversation_block}"
            f"{mem0_block}"
            f"{history_text}"
            f"{file_content_block}"
            f"## 你要调用的工具\n"
            f"`{tool_name}`：{tool_desc}\n\n"
            f"## 参数表\n{schema_text}\n\n"
            f"## 路径参数规则（重要）\n"
            f"1. **如果任务里给的路径是相对路径或不完整，"
            f"从「对话历史」里找最近一次明确的完整路径**。\n"
            f"2. **绝对不要编造路径**（不要用 `./data`、`D:/data` 等占位）。\n"
            f"3. **优先复用对话历史里出现过的完整路径**。\n"
            f"4. 对话历史里的 `user` 是用户原话，`assistant` 是 Agent 回复——"
            f"两边都可能包含真实路径。\n\n"
            f"## 输出要求\n"
            f"只输出该工具的 JSON 参数对象，不要任何解释，"
            f"不要包裹在 markdown 代码块里。\n"
            f"确保 JSON 合法、字段名与参数表完全一致。\n"
            f"**`old_text` 的值必须严格来自上面的文件内容**，不要编造。\n\n"
            f"示例格式：{{\"file_path\": \"D:/a.txt\", \"content\": \"hello\"}}\n\n"
            f"只输出 JSON："
        )

        import time as _time
        _t0 = _time.time()

        try:
            logger.info(
                "[dsh][orchestrator] _fill_params: calling LLM for %s, "
                "prompt length=%d",
                tool_name, len(prompt),
            )
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    self._llm_provider().generate,
                    [{"role": "user", "content": prompt}],
                ),
                timeout=120.0,
            )
            duration_ms = (_time.time() - _t0) * 1000
            logger.info(
                "[dsh][orchestrator] _fill_params raw response for %s: %r",
                tool_name, raw,
            )

            # 落盘
            dump_llm_call(
                kind="fill_params",
                prompt=prompt,
                response=raw,
                model=getattr(self._llm_provider(), "model", None),
                duration_ms=duration_ms,
                tool_name=tool_name,
                extra={"goal": goal[:200]},
            )
        except asyncio.TimeoutError:
            duration_ms = (_time.time() - _t0) * 1000
            logger.warning(
                "[dsh][orchestrator] _fill_params timed out (120s) for %s",
                tool_name,
            )
            dump_llm_call(
                kind="fill_params",
                prompt=prompt,
                response=None,
                duration_ms=duration_ms,
                tool_name=tool_name,
                error="timeout(120s)",
            )
            return {}
        except asyncio.CancelledError:
            logger.warning("[dsh][orchestrator] _fill_params cancelled")
            raise
        except Exception as exc:
            duration_ms = (_time.time() - _t0) * 1000
            logger.exception(
                "[dsh][orchestrator] _fill_params LLM failed for %s", tool_name
            )
            dump_llm_call(
                kind="fill_params",
                prompt=prompt,
                response=None,
                duration_ms=duration_ms,
                tool_name=tool_name,
                error=str(exc),
            )
            return {}

        data = self._parse_json(raw)
        if not isinstance(data, dict):
            logger.warning(
                "[dsh][orchestrator] _fill_params non-JSON for %s: %s",
                tool_name, raw[:200],
            )
            return {}

        # 参数名自适应
        adapted = self._adapt_params_for_tool(tool_name, data, tools)
        if adapted != data:
            logger.info(
                "[dsh][orchestrator] _fill_params params adapted for %s: "
                "%r → %r",
                tool_name, data, adapted,
            )
        return adapted
    
    async def run(
        self,
        goal: str,
        allowed_danger_levels: Optional[set] = None,
        allowed_tool_names: Optional[set] = None,
        allow_mcp_discovery: bool = True,
        workspace_paths: Optional[List[str]] = None,
        plan_mode: bool = False,
        resume_state: Optional[OrchestratorState] = None,
        resume_from_step: int = 1,
        prior_messages: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
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

        # ── 初始化 state（可能从快照恢复） ──
        if resume_state is not None:
            state = resume_state
            # 旧快照没有 session_id → 用本次传入的补上
            if state.session_id is None:
                state.session_id = session_id
            start_step = max(1, resume_from_step)
            logger.info(
                "[dsh][orchestrator] resuming from step %d, history=%d entries, "
                "session_id=%r",
                start_step, len(state.history), state.session_id,
            )
        else:
            state = OrchestratorState(goal=goal, session_id=session_id)
            start_step = 1

        for step_index in range(start_step, self._max_steps + 1):
            decision = self._try_fast_path(goal, state, tools)
            if decision is None:
                decision = await self._decide(
                    goal, state, tools, prior_messages=prior_messages,
                )

            if decision is None:
                yield {"type": "error", "message": "编排决策失败"}
                return

            # 决策要调工具 → 必填参数（渐进式披露第 2 阶段）
            if decision.tool_name and not decision.done:
                decision.parameters = await self._fill_params(
                    goal, decision.tool_name, tools, state,
                    prior_messages=prior_messages,
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
                    "state_snapshot": state.snapshot(),
                    "next_step": step_index + 1,
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

            logger.info(
                "[dsh][orchestrator] executing tool: %s params=%r",
                tool_name, params,
            )

            try:
                result = await agent.execute_tool_async(tool_call)
            except Exception as error:
                logger.exception("Tool execution failed in orchestrator")
                result = ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(error),
                )

            logger.info(
                "[dsh][orchestrator] tool result: %s success=%r error=%r "
                "result_type=%s result_preview=%r",
                tool_name,
                result.success,
                result.error,
                type(result.result).__name__,
                (
                    str(result.result)[:300]
                    if result.result is not None
                    else None
                ),
            )

            log = agent.format_tool_result(result)
            status = self._classify_status(log, result.success)
            state.add_history(
                f"Step {step_index}: {tool_name}({params}) "
                f"→ {status}: {log[:400]}"
            )

            if result.success:
                state.mark_success(tool_name, params, step_index)

                # ── Mem0 写入：工具成功后提取结构化事实 ──
                try:
                    from backend.services.memory_layer import (
                        remember_tool_success,
                    )
                    remember_tool_success(
                        session_id=state.session_id or "",
                        tool_name=tool_name,
                        params=params,
                        result=result.result,
                    )
                except Exception:
                    logger.exception("remember_tool_success 失败")

                # 记录最近一次读到的文件内容，供 _fill_params 用
                if "read" in tool_name.lower():
                    try:
                        payload = result.result
                        content = None
                        if isinstance(payload, dict):
                            content = payload.get("content")
                        elif isinstance(payload, str):
                            content = payload
                        if content:
                            state.last_read_content = str(content)
                            state.last_read_path = (
                                params.get("file_path")
                                or params.get("path")
                                or params.get("directory")
                            )
                            logger.info(
                                "[dsh][orchestrator] cached read content: "
                                "path=%r len=%d",
                                state.last_read_path,
                                len(state.last_read_content),
                            )
                    except Exception:
                        logger.exception("记录读取内容失败")
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