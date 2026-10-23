"""AgentOrchestrator 主类。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from infra.debug_log.llm import dump_llm_call

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
from .validator import (
    requires_tool_call,
    classify_request,
    is_web_query,
    check_web_query_completion,
)
from .intent_router import route_intent


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorStep:
    step_index: int
    thought: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    done: bool = False


# ─────────────────────────────────────────────────────────────
# 结构化历史（跨 turn 上下文）
# ─────────────────────────────────────────────────────────────

def _format_relative_time(iso_ts: Optional[str], now_ts: float) -> str:
    """把 ISO 时间戳转成「刚刚」/「3 分钟前」/「昨天」这类相对时间。

    空输入 / 解析失败 → 返回空字符串。
    """
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(iso_ts)
        delta = now_ts - dt.timestamp()
    except Exception:
        return ""

    if delta < 0:
        return "刚刚"
    if delta < 60:
        return "刚刚"
    if delta < 3600:
        return f"{int(delta // 60)} 分钟前"
    if delta < 86400:
        return f"{int(delta // 3600)} 小时前"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)} 天前"
    return dt.strftime("%Y-%m-%d")


def _build_structured_history(
    messages: List[Dict[str, Any]],
    max_items: int = 6,
    now_ts: Optional[float] = None,
) -> str:
    """把原始消息列表压成结构化历史块。

    设计意图：
        - **用户消息**：保留内容（截断到 300 字符）——这是用户意图
        - **assistant 消息**：只标记「已回复」，**不带原文**
          —— 避免历史里的绝对化结论（如"无法渲染"）污染当前决策
        - 每条消息带**相对时间**，让 LLM 区分"过去"和"现在"
    """
    if not messages:
        return ""
    import time as _time
    now_ts = now_ts if now_ts is not None else _time.time()

    recent = messages[-max_items:]
    lines: List[str] = []
    for m in recent:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        ts = m.get("timestamp") or ""
        rel = _format_relative_time(ts, now_ts)

        if role == "user":
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if len(content) > 300:
                content = content[:300] + "..."
            prefix = f"[{rel}] " if rel else ""
            lines.append(f"- {prefix}用户：{content}")
        elif role == "assistant":
            prefix = f"[{rel}] " if rel else ""
            lines.append(f"- {prefix}Agent 已回复（内容省略）")
        # 其他 role 忽略

    return "\n".join(lines)

class _SucceededSeqView:
    """把 state.succeeded_tool_names（dict）适配成 list 序列。

    validator.check_web_query_completion 遍历它找 tags；
    每次访问都从 state 实时读，不缓存。
    """

    def __init__(self, state):
        self._state = state

    def __iter__(self):
        names = getattr(self._state, "succeeded_tool_names", {}) or {}
        return iter(list(names.keys()))

    def __len__(self):
        names = getattr(self._state, "succeeded_tool_names", {}) or {}
        return len(names)

    def __bool__(self):
        return len(self) > 0
        
class AgentOrchestrator:
    _MCP_DISCOVERY_TOOLS = frozenset({"search_and_connect_mcp"})

    # 参与工作区越界判断的路径参数名
    _WORKSPACE_PATH_KEYS: frozenset[str] = frozenset({
        "path", "file_path", "directory", "dir",
        "source", "destination", "src", "dst",
        "target", "target_path", "output_path",
    })

    # 写类工具名关键词（小写子串匹配）
    _WRITE_TOOL_HINTS: tuple[str, ...] = (
        "write", "create", "delete", "remove", "update", "modify",
        "move", "rename", "mkdir", "rmdir", "append", "truncate",
        "upload", "put", "edit", "insert", "drop", "copy",
    )

    # 按 token 预算而非字符数（中文 1 token ≈ 1.5 字符）
    _HISTORY_TOKEN_BUDGET = 4000

    _HISTORY_LIMITS = {
        "search_knowledge": 6000,
        "read_pdf_text": 4000,
        "read_docx_text": 4000,
        "read_csv": 2000,
        "search_content": 2000,
        "fetch_url": 4000,
        "http_get": 4000,
    }

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


    def _format_tool_directory(self, tools: Dict[str, Any]) -> str:
        """只给工具名 + 一句描述 + tags，用于「选工具」阶段。

        不带参数 schema，避免干扰 LLM 的语义匹配。
        """
        if not tools:
            return "(无可用工具)"

        # discovery 类工具：完整保留描述，避免截断掉关键引导句
        _FULL_DESC_TOOLS = frozenset({"search_and_connect_mcp"})

        lines = []
        for name, meta in tools.items():
            desc_raw = (getattr(meta, "description", "") or "").strip()
            if name in _FULL_DESC_TOOLS:
                # 取前 3 行（保留 "Use this when..." 这类引导）
                desc = "\n    ".join(desc_raw.split("\n")[:3]).strip()
            else:
                desc = desc_raw.split("\n")[0].strip()
                if len(desc) > 70:
                    desc = desc[:67] + "..."

            tags = get_intent_tags(name)
            tag_str = ""
            if tags:
                tag_str = f"  [tags: {','.join(str(t) for t in tags[:4])}]"

            # 内置工具打 ⭐，让 LLM 倾向选内置
            marker = "" if name.startswith("mcp__") else "⭐ "

            # MCP 工具：附上 server 名，让 LLM 知道来源
            server_note = ""
            if name.startswith("mcp__"):
                parts = name.split("__", 2)
                if len(parts) == 3:
                    server_note = f"  (MCP server: {parts[1]})"
            lines.append(f"- {marker}`{name}`{server_note}{tag_str}\n    {desc}")

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

        # ── 通用规则：常见别名 → schema 字段名 ──
        _ALIASES = {
            "limit": ("max_results", "count", "top_k", "size"),
            "max_results": ("limit", "count", "top_k"),
            "query": ("q", "keyword", "search_query"),
            "days": ("time_range", "since_days", "within_days"),
        }
        for target, aliases in _ALIASES.items():
            if target in allowed:
                continue
            for alias in aliases:
                if alias in result and target not in result:
                    result[target] = result.pop(alias)
                    break

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
        """MCP 工具走意图索引，内置工具走 embedding shortlist。

        - MCP 工具：用 tools/mcp_intent_index.search_intents 检索，
          比 embedding 直接匹配 tool name 准得多。
        - 内置工具：沿用 shortlister（已换成 bge-small-zh）。
        - 任一步失败都退回全量，不阻塞。
        """
        picked: Dict[str, Any] = {}

        builtin_tools = {
            n: m for n, m in tools.items() if not n.startswith("mcp__")
        }
        mcp_tools = {
            n: m for n, m in tools.items() if n.startswith("mcp__")
        }

        # ── 1. 内置工具：embedding shortlist ──
        if builtin_tools:
            if self._shortlister_provider is not None:
                try:
                    sl = self._shortlister_provider()
                    if sl is not None:
                        sub = sl.shortlist(goal, builtin_tools)
                        if sub:
                            picked.update(sub)
                except Exception:
                    logger.exception(
                        "[dsh][orchestrator] shortlister raised for builtin"
                    )
            if not picked:
                picked.update(builtin_tools)

        # ── 2. MCP 工具：意图索引检索 ──
        if mcp_tools:
            try:
                from tools.mcp_intent_index import search_intents
                hits = search_intents(goal, top_k=15)
                keep_mcp = set()
                for h in hits:
                    for t in h.get("matched_tools", []):
                        name = t.get("tool_name")
                        if name in mcp_tools:
                            keep_mcp.add(name)
                if keep_mcp:
                    picked.update({n: mcp_tools[n] for n in keep_mcp})
                    logger.info(
                        "[dsh][orchestrator] intent index kept %d/%d MCP tools",
                        len(keep_mcp), len(mcp_tools),
                    )
                else:
                    # 意图索引无命中 → 全量 MCP 交给 LLM（不要筛掉）
                    logger.info(
                        "[dsh][orchestrator] intent index no hit, "
                        "pass all %d MCP tools",
                        len(mcp_tools),
                    )
                    picked.update(mcp_tools)
            except Exception:
                logger.exception(
                    "[dsh][orchestrator] intent index failed, pass all MCP"
                )
                picked.update(mcp_tools)

        if not picked:
            picked = dict(tools)

        # ── 3. 强制保留关键工具 ──
        force_keep = self._MCP_DISCOVERY_TOOLS | {
            "fetch_url",
            "search_knowledge",
            "knowledge_stats",
            "list_knowledge_sources",
            "mcp__tavily__tavily_search",
            "mcp__tavily__tavily_extract",
            "mcp__tavily__tavily_crawl",
            "mcp__tavily__tavily_research",
        }
        added = []
        for name in force_keep:
            if name in tools and name not in picked:
                picked[name] = tools[name]
                added.append(name)
        if added:
            logger.info(
                "[dsh][orchestrator] force-kept tools: %s", added
            )

        return picked

    async def _decide(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
        prior_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[OrchestratorStep]:
        # 预筛：优先 embedding，退关键词
        shortlist = self._select_shortlist(goal, tools)

        # 对话历史（结构化）——只保留用户意图，不保留 assistant 结论
        conversation_block = ""
        if prior_messages:
            structured = _build_structured_history(prior_messages, max_items=6)
            if structured:
                conversation_block = (
                    "## 对话历史（结构化）\n"
                    + structured
                    + "\n\n**注意**："
                    "`Agent 已回复（内容省略）` 只表示上一轮已响应，"
                    "**不代表当前状态**。历史结论可能已过时——"
                    "重新审视当前工具目录再决策。\n\n"
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

        # ── 网页检索优先检查（正交于 action / query） ──
        if is_web_query(goal):
            web_reject = check_web_query_completion(state)
            if web_reject:
                logger.info(
                    "[dsh][orchestrator] web query done rejected: %s",
                    web_reject[:120],
                )
                return web_reject

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
        # ── discovery 总次数上限 ──
        if tool_name in self._MCP_DISCOVERY_TOOLS:
            count = getattr(state, "discovery_count", 0)
            if count >= 5:
                return (
                    f"⛔ 本任务已调用 discovery {count} 次，达到上限。\n"
                    f"**下一步**：用已连接的工具，或 done=true 告诉用户"
                    f"找不到合适的外部服务。"
                )

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

        # ── 零参数工具：直接返回空 dict，不让 LLM 编字段 ──
        _schema = getattr(meta, "input_schema", None) or {}
        _props = _schema.get("properties") or {}
        _required = _schema.get("required") or []
        _is_mcp = tool_name.startswith("mcp__")
        _is_discovery = tool_name in self._MCP_DISCOVERY_TOOLS

        # 零参数早退：discovery 工具和 MCP 工具除外
        # （它们的 input_schema 可能来自 manager.discover 的 spec，
        #   register_dynamic_tool 没透传到 meta.input_schema，
        #   直接 return {} 会导致参数全空、调用失败）
        if (
            not _props and not _required
            and not _is_discovery and not _is_mcp
        ):
            logger.info(
                "[dsh][orchestrator] _fill_params: %s has no params, "
                "skip LLM call",
                tool_name,
            )
            return {}

        # discovery 工具 / MCP 工具：schema 读不到时用 parameters 兜底
        if (_is_discovery or _is_mcp) and not _props:
            _props = getattr(meta, "parameters", None) or {}
            # 如果 parameters 里有必填信息，也兜底
            if not _required and hasattr(meta, "required"):
                _required = list(getattr(meta, "required", []) or [])
            logger.info(
                "[dsh][orchestrator] _fill_params: %s schema empty, "
                "fallback to parameters: %r, required: %r",
                tool_name, list(_props.keys()), _required,
            )

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
        # 【安全】discovery 工具不注入任何历史/记忆：
        #   - prior_messages 含上一轮实体，会污染 capability
        #   - mem0_block 含本会话文件路径，属于用户私有上下文，
        #     不应随发现请求发给外部 Registry / MCP server
        conversation_block = ""
        if prior_messages and tool_name not in self._MCP_DISCOVERY_TOOLS:
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
        # 【安全】discovery 工具不注入 mem0：文件路径属于用户私有上下文
        mem0_block = ""
        if (
            tool_name not in self._MCP_DISCOVERY_TOOLS
            and state is not None
            and getattr(state, "session_id", None)
        ):
            try:
                from core.memory.mem0_layer import recall_file_paths
                known_paths = recall_file_paths(state.session_id, limit=5)
                # 过滤根路径/盘符，避免 LLM 拿 "/" 当 PDF 路径
                known_paths = [
                    p for p in known_paths
                    if isinstance(p, str)
                    and p.strip()
                    and p.strip() not in {"/", "\\"}
                    and not re.fullmatch(r"[A-Za-z]:[\\/]?", p.strip())
                ]
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

        # discovery 工具：强制把原始 goal 塞进 user_query
        # 【重要】只对 discovery 工具拼这段，否则会污染 search_knowledge /
        # read_pdf_text / mcp__*__search_news 等工具的 _fill_params prompt，
        # 导致 LLM 给这些工具编造 user_query / capability 字段。
        discovery_block = ""
        if tool_name in self._MCP_DISCOVERY_TOOLS:
            goal_json = json.dumps(goal, ensure_ascii=False)
            failed = getattr(state, "failed_capabilities", set()) if state else set()
            failed_line = ""
            if failed:
                failed_line = (
                    f"- **已失败的关键词（不要再用）**："
                    f"{', '.join(sorted(failed))}\n"
                )
            discovery_block = (
                "## discovery 工具填参规则（强制，必须同时填两个字段）\n"
                f"- `user_query` = {goal_json}（原样，不翻译、不改写）\n"
                "- `capability` = 英文关键词，逗号分隔，2~4 个。\n"
                "  用 MCP 作者会在包名里用的通用词根。\n"
                f"{failed_line}"
                "- **两个字段都必须出现在 JSON 里**，缺一不可。\n"
                "- 输出示例：\n"
                f"  {{\"user_query\": {goal_json}, "
                "\"capability\": \"news,sports\", \"limit\": 100}}\n\n"
            )

        prompt = (
            f"## 任务\n{goal}\n\n"
            f"{conversation_block}"
            f"{mem0_block}"
            f"{history_text}"
            f"{file_content_block}"
            f"## 你要调用的工具\n"
            f"`{tool_name}`：{tool_desc}\n\n"
            f"{discovery_block}"
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
            f"确保 JSON 合法、字段名与**参数表完全一致**——"
            f"**只输出参数表里列出的字段**，不要凭经验添加其他字段。\n"
            + (
                f"如果参数表显示「该工具无参数」，输出 `{{}}`。\n"
                if not _props
                else (
                    f"示例（字段名仅供参考格式，**必须用参数表里的真实字段**）：\n"
                    f"```\n"
                    f"{{\"<参数名1>\": <值1>, \"<参数名2>\": <值2>}}\n"
                    f"```\n\n"
                )
            )
            + f"只输出 JSON："
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
            # 兜底：从文本里抓第一个 {...}
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if m:
                data = self._parse_json(m.group(0))
            if not isinstance(data, dict):
                return {}

        # 参数名自适应
        adapted = self._adapt_params_for_tool(tool_name, data, tools)
        if adapted != data:
            logger.info(
                "[dsh][orchestrator] _fill_params params adapted for %s: "
                "%r → %r",
                tool_name, data, adapted,
            )

        # ── 参数名校验：LLM 可能编造 schema 外的字段 ──
        allowed = set(_props.keys())
        if allowed:
            unknown = [k for k in adapted.keys() if k not in allowed]
            if unknown:
                logger.warning(
                    "[dsh][orchestrator] _fill_params: %s got unknown params "
                    "%r (allowed=%r), dropping",
                    tool_name, unknown, sorted(allowed),
                )
                adapted = {k: v for k, v in adapted.items() if k in allowed}
            missing = [k for k in _required if k not in adapted]
            if missing:
                logger.warning(
                    "[dsh][orchestrator] _fill_params: %s missing required %r",
                    tool_name, missing,
                )

        # ── discovery 工具兜底：user_query 空时用 goal 填 ──
        if _is_discovery:
            if not adapted.get("user_query"):
                adapted["user_query"] = goal
                logger.info(
                    "[dsh][orchestrator] _fill_params: %s user_query empty, "
                    "filled with goal", tool_name,
                )
            # 过滤已失败的关键词（LLM 不一定遵守 prompt 里的负面指令）
            failed = getattr(state, "failed_capabilities", set()) if state else set()
            if failed:
                cap = adapted.get("capability", "")
                if cap:
                    kws = [
                        kw.strip() for kw in cap.split(",")
                        if kw.strip() and kw.strip().lower() not in failed
                    ]
                    if kws:
                        adapted["capability"] = ",".join(kws)
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "filtered failed keywords, remaining: %r",
                            tool_name, kws,
                        )
                    else:
                        adapted.pop("capability", None)
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "all keywords failed, cleared capability",
                            tool_name,
                        )

        # ── MCP 工具兜底：必填参数空时，用 goal / 实体填 ──
        if _is_mcp and _required:
            missing = [k for k in _required if not adapted.get(k)]
            if missing:
                # 常见必填参数名 → 用 goal 填
                for k in missing:
                    if k in ("query", "q", "keyword", "search_query"):
                        adapted[k] = goal
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "required %r empty, filled with goal",
                            tool_name, k,
                        )
                    elif k in ("path", "file_path", "url"):
                        # 从 goal 里抽路径/URL
                        import re as _re
                        m = _re.search(r'https?://[^\s"\']+', goal)
                        if m:
                            adapted[k] = m.group(0)
                    # 其他必填参数不猜，留给工具层报错

        return adapted

    # ── 结果达标评估 ──

    # 检索类工具：返回非空且包含 query 实体即视为充分，不走 LLM
    _RETRIEVAL_TOOLS = frozenset({
        "search_knowledge",
        "knowledge_stats",
        "list_knowledge_sources",
        "search_content",
        "search_files",
        "read_pdf_text",
        "read_pdf_metadata",
        "read_docx_text",
        "read_docx_tables",
        "read_csv",
        "read_text_file",
        "read_file_range",
        "fetch_url",
        "http_get",
        "fetch_json",
    })

    @staticmethod
    def _extract_query_terms(goal: str) -> List[str]:
        """从 goal 里提取 2-4 字的中文实体片段，用于确定性匹配。"""
        terms: List[str] = []
        for m in re.finditer(r"[\u4e00-\u9fff]{2,4}", goal or ""):
            terms.append(m.group(0))
        seen = set()
        out = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:8]

    def _deterministic_assess(
        self,
        goal: str,
        tool_name: str,
        preview: str,
    ) -> Optional[Dict[str, Any]]:
        """确定性充分性判断。返回 None 表示需要 LLM 兜底。"""
        if not preview:
            if tool_name in ("search_knowledge", "knowledge_stats", "list_knowledge_sources"):
                hint = (
                    "知识库可能为空或 mode 选错；"
                    "先调用 knowledge_stats 检查，再换 query 或 mode='flat' 重试"
                )
            elif tool_name in ("fetch_url", "http_get", "fetch_json"):
                hint = "换一个 URL 或换一个搜索词重试"
            elif tool_name in ("search_content", "search_files"):
                hint = "换一个关键词或扩大搜索范围"
            else:
                hint = "换一种策略重试"
            return {
                "sufficient": False,
                "gap": "工具返回内容为空",
                "next_hint": hint,
            }

        if tool_name not in self._RETRIEVAL_TOOLS:
            return None

        # search_knowledge：检查覆盖的文档数
        if tool_name == "search_knowledge":
            multi_doc_hints = (
                "这些文档", "所有文档", "各个文档", "每个文档",
                "几个文档", "多少文档", "分别",
            )
            wants_multi = any(h in goal for h in multi_doc_hints)
            if wants_multi:
                files_in_preview = set(
                    re.findall(r'"filename":\s*"([^"]+)"', preview)
                )
                if len(files_in_preview) < 2:
                    return {
                        "sufficient": False,
                        "gap": (
                            f"只覆盖了 {len(files_in_preview)} 个文档"
                            f"（{files_in_preview or '无'}），用户要看多个文档"
                        ),
                        "next_hint": (
                            "改用 mode='flat' 或 mode='auto' 重新检索，"
                            "或分别对每个文档 search_knowledge"
                        ),
                    }
                    
        # 检索类工具：preview 里出现任意 query 实体片段 → 充分
        terms = self._extract_query_terms(goal)
        if terms and any(t in preview for t in terms):
            return {
                "sufficient": True,
                "gap": "",
                "next_hint": "",
            }

        # 没有实体片段但 preview 很长（> 500 字），视为有内容
        if len(preview) > 500:
            return {
                "sufficient": True,
                "gap": "",
                "next_hint": "",
            }

        return None

    async def _assess_result(
        self,
        goal: str,
        tool_name: str,
        params: Dict[str, Any],
        result: ToolCallResult,
        state: OrchestratorState,
    ) -> Optional[Dict[str, Any]]:
        """评估最近一次工具结果是否足以回答用户目标。

        返回 {"sufficient": bool, "gap": str, "next_hint": str}；
        评估失败返回 None（上层视为通过，避免卡死）。
        """
        # 只评估信息获取类工具；写类工具成功即视为完成
        if self._is_write_operation(tool_name):
            return None

        # 取结果文本
        payload = result.result
        preview = ""
        if isinstance(payload, dict):
            # MCP 风格：content 数组
            blocks = payload.get("content") or []
            if isinstance(blocks, list):
                texts = [
                    b.get("text", "") for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                preview = "\n".join(t for t in texts if t)

            # 知识库风格：chunks 数组
            if not preview:
                chunks = payload.get("chunks") or []
                if isinstance(chunks, list):
                    texts = [
                        c.get("text", "") for c in chunks
                        if isinstance(c, dict)
                    ]
                    preview = "\n".join(t for t in texts if t)

                if not preview:
                    _ECHO_KEYS = frozenset({
                        "success", "query", "mode", "top_k", "error",
                        "paths", "warnings", "resources", "chunks",
                    })
                    flat = [
                        f"{k}: {v}" for k, v in payload.items()
                        if k not in _ECHO_KEYS
                        and isinstance(v, (str, int, float))
                        and not isinstance(v, bool)   # True/False 不算内容
                        and v
                    ]
                    preview = "\n".join(flat)

        elif isinstance(payload, str):
            preview = payload
        else:
            preview = str(payload) if payload is not None else ""

        # ── 确定性优先：检索类工具命中实体即充分，不走 LLM ──
        deterministic = self._deterministic_assess(goal, tool_name, preview)
        if deterministic is not None:
            logger.info(
                "[dsh][orchestrator] deterministic assess: %s → sufficient=%r",
                tool_name, deterministic["sufficient"],
            )
            return deterministic

        # preview 放宽到 12000，避免实体被截断
        if len(preview) > 12000:
            preview = preview[:12000] + "\n... (truncated)"

        prompt = (
            "你是结果评估助手。判断最近一次工具返回是否足以回答用户目标。\n\n"
            f"## 用户目标\n{goal}\n\n"
            f"## 刚刚执行\n工具：{tool_name}\n参数：{params}\n\n"
            f"## 返回内容摘要\n{preview}\n\n"
            "## 判断标准\n"
            "- 用户问「某个页面上的具体内容」，返回的是「关于该页面的分析文章」"
            "→ 不达标。\n"
            "- 用户问「实时/最新数据」，返回的是「历史数据」→ 不达标。\n"
            "- 用户问「列表」，返回的只是「分类概览」→ 不达标。\n"
            "- 用户问「页面 A 的内容」，返回的是「页面 B 的内容」→ 不达标。\n"
            "- 返回内容包含用户要的具体信息 → 达标。\n\n"
            "## 输出（严格 JSON，不要额外解释）\n"
            "{\n"
            '    "sufficient": true 或 false,\n'
            '    "gap": "缺少什么（sufficient=false 时必填）",\n'
            '    "next_hint": "下一步建议（sufficient=false 时必填）"\n'
            "}\n"
        )

        import time as _time
        _t0 = _time.time()

        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    self._llm_provider().generate,
                    [{"role": "user", "content": prompt}],
                ),
                timeout=60.0,
            )
            duration_ms = (_time.time() - _t0) * 1000
            logger.info(
                "[dsh][orchestrator] _assess_result raw response: %r", raw
            )
            dump_llm_call(
                kind="assess_result",
                prompt=prompt,
                response=raw,
                model=getattr(self._llm_provider(), "model", None),
                duration_ms=duration_ms,
                tool_name=tool_name,
                extra={"goal": goal[:200]},
            )
        except asyncio.TimeoutError:
            logger.warning("[dsh][orchestrator] _assess_result timed out (60s)")
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("_assess_result LLM call failed")
            return None

        data = self._parse_json(raw)
        if not isinstance(data, dict):
            return None
        return {
            "sufficient": bool(data.get("sufficient", True)),
            "gap": str(data.get("gap", "")),
            "next_hint": str(data.get("next_hint", "")),
        }

    @staticmethod
    def _get_manager_for_discovery():
        from tools.mcp_discovery import _get_manager
        return _get_manager()

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
        # ── 调试：确认 session_id 是否传进来 ──
        logger.info(
            "[dsh][orchestrator] run() session_id=%r, "
            "resume_state=%r, goal=%r",
            session_id,
            resume_state is not None,
            goal[:100],
        )

        # ── 等 mem0 ready，避免冷启动首轮写入丢失 ──
        try:
            from core.memory.mem0_layer import wait_until_ready
            ready = await wait_until_ready(timeout=30.0)
            logger.info("[dsh][orchestrator] mem0 ready=%r", ready)
        except Exception:
            logger.exception("wait_until_ready 失败，继续执行")

        # 每个任务开始重置"已试过的 MCP 服务器"
        try:
            from tools.mcp_discovery import reset_tried_servers
            reset_tried_servers()
        except Exception:
            logger.exception("failed to reset tried MCP servers")

        agent = self._agent_provider()

        def _refresh_tools() -> Dict[str, Any]:
            """重新从 registry 拉取工具列表。

            MCP discovery 会在任务执行中途动态注册工具，
            因此必须每步刷新，而不是只在开头取一次。
            """
            t = agent.list_all_tools(allowed_danger_levels=allowed_danger_levels)
            if not allow_mcp_discovery:
                t = {
                    name: meta for name, meta in t.items()
                    if name not in self._MCP_DISCOVERY_TOOLS
                }
            return t

        tools = _refresh_tools()
        logger.info(
            "[dsh][orchestrator] allowed_dangers=%r, filtered tools count=%d, "
            "has write_file=%r",
            allowed_danger_levels,
            len(tools),
            "write_file" in tools,
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

        # ── 兼容：validator.check_web_query_completion 需要
        #    state.successful_tool_sequence（list）。
        try:
            if not hasattr(state, "successful_tool_sequence"):
                object.__setattr__(
                    state, "successful_tool_sequence",
                    _SucceededSeqView(state),
                )
        except Exception:
            logger.exception("failed to attach successful_tool_sequence view")
            
        for step_index in range(start_step, self._max_steps + 1):
            # ── 每步刷新工具列表 ──
            # 上一步的 search_and_connect_mcp 可能刚注册了新 MCP 工具，
            # 不刷新会导致 LLM 选了新工具却被判定为"不存在"。
            tools = _refresh_tools()

            # ── L1→L4 自动发现流水线（LLM 不参与） ──
            try:
                from tools.capability.pipeline import maybe_discover
                from tools.mcp_discovery import _get_manager as _get_mcp_manager
                tools = await maybe_discover(
                    goal=goal,
                    tools=tools,
                    manager=_get_mcp_manager(),
                    register_tool=agent.register_dynamic_tool,
                    unregister_tool=agent.unregister_dynamic_tool,
                    allowed_danger_levels=allowed_danger_levels,
                    state=state,
                )
            except Exception:
                logger.exception(
                    "[dsh][orchestrator] capability pipeline failed, "
                    "continue with current tools"
                )

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
            if tool_name in self._MCP_DISCOVERY_TOOLS:
                state.discovery_count += 1

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
            limit_chars = self._HISTORY_LIMITS.get(tool_name, 600)
            state.add_history(
                f"Step {step_index}: {tool_name}({params}) → {status}: {log[:limit_chars]}"
            )

            if result.success:
                state.mark_success(tool_name, params, step_index)

                # ── 结果达标评估 ──
                assessment = await self._assess_result(
                    goal, tool_name, params, result, state
                )
                if assessment and not assessment.get("sufficient", True):
                    gap = assessment.get("gap", "")
                    hint = assessment.get("next_hint", "")
                    # 记录失败的 discovery 关键词，避免下一轮重复
                    if tool_name in self._MCP_DISCOVERY_TOOLS:
                        cap = params.get("capability", "")
                        for kw in cap.split(","):
                            kw = kw.strip().lower()
                            if kw:
                                state.failed_capabilities.add(kw)
                    assess_msg = (
                        f"⚠️ Step {step_index} 结果评估：不达标。\n"
                        f"缺失：{gap}\n"
                        f"建议：{hint}\n"
                        f"**必须继续尝试，不要 done=true。**"
                    )
                    state.add_history(assess_msg)
                    logger.info(
                        "[dsh][orchestrator] result insufficient: %s",
                        assess_msg[:200],
                    )
                    yield {
                        "type": "thinking",
                        "step": step_index,
                        "thought": f"[结果评估] 不达标：{gap}",
                    }

                # ── Mem0 写入：工具成功后提取结构化事实 ──
                try:
                    from core.memory.mem0_layer import (
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