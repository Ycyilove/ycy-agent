"""AgentOrchestrator 主类。

重构后设计：
    - 工具筛选只保留两段：场景过滤 + embedding shortlist
    - 决策权交给 LLM prompt（工具链/前置条件/失败切换都在 prompt 里）
    - 代码只保留安全兜底：环境不可用工具过滤、幂等拦截、完成度校验
"""

from __future__ import annotations

import os
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
from tools.semantics import get_intent_tags

from .prompt import ORCHESTRATOR_PROMPT
from .state import OrchestratorState, params_key
from .validator import requires_tool_call
from .task_classifier import classify_task
from .task_types import (
    TaskType, StopReason, AssessResult, GapKind, can_upgrade,
    TERMINAL_REASONS,
)
from .audit import run_audit, collect_evidence
from core.llm.provider import ContextLengthExceeded


logger = logging.getLogger(__name__)

import os as _os

_READ_TOOL_HINTS = (
    "read", "list", "search", "get_file_info", "stat", "fetch", "crawl",
)


def _is_read_like(name: str) -> bool:
    low = (name or "").lower()
    return any(h in low for h in _READ_TOOL_HINTS)


def _norm_filename(name: str) -> str:
    """归一化文件名：去路径前缀、去首尾空白、统一分隔符。"""
    if not isinstance(name, str):
        return ""
    s = name.replace("\\", "/").strip()
    return _os.path.basename(s)


@dataclass
class OrchestratorStep:
    step_index: int
    thought: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    search_plan: Optional[Dict[str, Any]] = None


def _format_relative_time(iso_ts: Optional[str], now_ts: float) -> str:
    """把 ISO 时间戳转成「刚刚」/「3 分钟前」/「昨天」这类相对时间。"""
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
    """把原始消息列表压成结构化历史块。"""
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

    return "\n".join(lines)


class _SucceededSeqView:
    """把 state.succeeded_tool_names（dict）适配成 list 序列。"""

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

    # 当前运行环境不可用的工具（Windows 裸机缺 WeasyPrint → render_document 不可用）
    _UNAVAILABLE_TOOLS = frozenset({
        "mcp__office-worker__render_document",
        "mcp__office-worker__pdf_ocr",
        "mcp__office-worker__create_book",
    })

    # 静态配置的 MCP server
    _STATIC_MCP_SERVERS = frozenset({"tavily", "fetch"})

    # 单步工具结果进 state.history 的字符上限
    _HISTORY_LIMITS = {
        "search_knowledge": 3000,
        "knowledge_stats": 1500,
        "list_knowledge_sources": 1500,
        "read_text_file": 1500,
        "read_file_range": 1500,
        "read_pdf_text": 2000,
        "read_docx_text": 2000,
        "read_csv": 1200,
        "search_content": 1200,
        "search_files": 1200,
        "fetch_url": 2000,
        "http_get": 2000,
        "write_file": 300,
        "edit_file": 300,
        "create_directory": 200,
        "move_file": 200,
        "delete_file": 200,
        "run_tests": 2000,
    }
    _DEFAULT_HISTORY_LIMIT = 1000

    def __init__(
        self,
        agent_provider: Callable[[], ToolAgent],
        llm_provider: Callable[[], Any],
        max_steps: int = 10,
        shortlister_provider: Optional[Callable[[], Any]] = None,
    ):
        self._agent_provider = agent_provider
        self._llm_provider = llm_provider
        self._max_steps = max_steps
        self._shortlister_provider = shortlister_provider

        try:
            from backend.config import (
                LLM_PROMPT_BUDGET_CHARS,
                LLM_HISTORY_BUDGET_RATIO,
                LLM_HISTORY_ENTRY_CHARS,
                LLM_COMPACT_TRIGGER_RATIO,
                LLM_COMPACT_KEEP_RECENT,
            )
            self._prompt_budget_chars = LLM_PROMPT_BUDGET_CHARS
            self._history_budget_ratio = LLM_HISTORY_BUDGET_RATIO
            self._history_entry_chars = LLM_HISTORY_ENTRY_CHARS
            self._compact_trigger_ratio = LLM_COMPACT_TRIGGER_RATIO
            self._compact_keep_recent = LLM_COMPACT_KEEP_RECENT
        except Exception:
            self._prompt_budget_chars = 24000
            self._history_budget_ratio = 0.4
            self._history_entry_chars = 1000
            self._compact_trigger_ratio = 0.8
            self._compact_keep_recent = 5

    def _format_tool_directory(self, tools: Dict[str, Any]) -> str:
        """只给工具名 + 一句描述 + tags，用于「选工具」阶段。"""
        if not tools:
            return "(无可用工具)"

        _FULL_DESC_TOOLS = frozenset({"search_and_connect_mcp"})

        lines = []
        for name, meta in tools.items():
            desc_raw = (getattr(meta, "description", "") or "").strip()
            if name in _FULL_DESC_TOOLS:
                desc = "\n    ".join(desc_raw.split("\n")[:3]).strip()
            else:
                desc = desc_raw.split("\n")[0].strip()
                if len(desc) > 70:
                    desc = desc[:67] + "..."

            tags = get_intent_tags(name)
            tag_str = ""
            if tags:
                tag_str = f"  [tags: {','.join(str(t) for t in tags[:4])}]"

            marker = "" if name.startswith("mcp__") else "⭐ "

            server_note = ""
            if name.startswith("mcp__"):
                parts = name.split("__", 2)
                if len(parts) == 3:
                    server_note = f"  (MCP server: {parts[1]})"
            lines.append(f"- {marker}`{name}`{server_note}{tag_str}\n    {desc}")

        return "\n".join(lines)

    def _format_single_tool_schema(self, meta: Any) -> str:
        """给单个工具的完整 schema，用于「填参数」阶段。"""
        properties, required_list = self._read_tool_schema(meta)
        required = set(required_list)
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

    @staticmethod
    def _resolve_tool_name(requested: str, available: Dict[str, Any]) -> Optional[str]:
        """把 LLM 幻觉出的简写工具名映射到实际存在的工具。"""
        if not requested:
            return None
        if requested in available:
            return requested

        suffix = f"__{requested}"
        matches = [n for n in available if n.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]

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

    def _parse_json(self, text: str) -> Optional[dict]:
        """从模型输出里提取第一个合法 JSON 对象。"""
        if not text:
            return None
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

        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(clean):
            pos = clean.find("{", idx)
            if pos < 0:
                break
            try:
                obj, end = decoder.raw_decode(clean, pos)
                if isinstance(obj, dict):
                    return obj
                idx = end
            except json.JSONDecodeError:
                idx = pos + 1

        return None

    _SCENE_BLOCKLIST = {
        "code": {"mysql_query", "mysql_execute", "edit_excel", "write_excel",
                 "read_csv", "read_excel", "convert_file_format"},
        "data": {"mysql_query", "mysql_execute", "search_content",
                 "search_files", "edit_file", "write_file"},
        "web": {"mysql_query", "mysql_execute", "edit_excel", "write_excel",
                "read_csv", "read_excel", "edit_file", "write_file",
                "search_content", "search_files"},
    }

    @staticmethod
    def _is_external_fact_query(goal: str) -> bool:
        """结构化判断是否需要外部信息。"""
        if not goal:
            return False
        g = goal.lower()
        _TIME_SENSITIVE = (
            "实时", "最新", "今天", "现在", "当前", "刚刚",
            "news", "latest", "realtime", "today", "now",
        )
        if any(k in g for k in _TIME_SENSITIVE):
            return True
        from tools.orchestration.validator import _looks_like_external_fact_query
        return _looks_like_external_fact_query(goal)

    _SOURCE_BUCKET_RULES = (
        ("official", (
            "gov", "edu", "official", "docs.", "/docs/", "/api/",
            "openai.com", "anthropic.com", "deepseek.com", "github.com",
            "arxiv.org", "nature.com", "science.org",
        )),
        ("news", (
            "news", "reuters", "bloomberg", "techcrunch", "theverge",
            "36kr", "sina", "sohu", "163.com", "bbc", "cnn",
        )),
        ("forum", (
            "reddit", "stackoverflow", "zhihu", "v2ex", "discourse",
            "forum", "bbs", "community", "discuss", "hackernews",
            "news.ycombinator", "stackexchange", "github.com/issues",
        )),
        ("blog", (
            "medium.com", "substack", "blog", "wordpress", "csdn",
            "juejin", "cnblogs", "dev.to", "hashnode",
        )),
    )

    _TIER_LABELS = {
        "official": "官方/一手（权威最高）",
        "news": "新闻/媒体（时效强，权威中）",
        "forum": "论坛/社区（一手经验，权威低但细节多）",
        "blog": "博客/个人（主观，权威低）",
        "unknown": "其他/未分类",
    }

    @classmethod
    def _bucket_source_url(cls, url: str) -> str:
        if not url:
            return "unknown"
        low = url.lower()
        for bucket, keywords in cls._SOURCE_BUCKET_RULES:
            if any(k in low for k in keywords):
                return bucket
        return "unknown"

    @classmethod
    def _extract_web_results(cls, payload: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not isinstance(payload, dict):
            return items

        def _collect(seq):
            if not isinstance(seq, list):
                return
            for r in seq:
                if isinstance(r, dict) and r.get("url"):
                    items.append({
                        "url": r["url"],
                        "title": r.get("title") or "",
                        "content": str(r.get("content") or "")[:800],
                        "score": r.get("score"),
                    })

        _collect(payload.get("results"))
        sc = payload.get("structuredContent")
        if isinstance(sc, dict):
            _collect(sc.get("results"))
        if payload.get("url") and payload.get("content"):
            items.append({
                "url": payload["url"],
                "title": payload.get("title") or "",
                "content": str(payload["content"])[:800],
                "score": None,
            })
        return items

    @classmethod
    def _bucket_web_results(cls, payload: Any) -> Dict[str, List[Dict[str, Any]]]:
        bucketed: Dict[str, List[Dict[str, Any]]] = {}
        for it in cls._extract_web_results(payload):
            b = cls._bucket_source_url(it["url"])
            bucketed.setdefault(b, []).append(it)
        return bucketed

    @classmethod
    def _filter_builtin_by_scene(cls, goal: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        """按 goal 场景过滤内置工具，排除明显无关的（mysql/excel 等）。"""
        if not goal:
            return tools
        g = goal.lower()

        scene = None
        if any(k in g for k in ("修复", "代码", "bug", "单元测试", "pytest",
                                ".py", ".js", ".ts", "重构", "函数")):
            scene = "code"
        elif any(k in g for k in ("excel", "csv", "表格", "数据", "统计")):
            scene = "data"
        elif any(k in g for k in ("网页", "新闻", "最新", "url", "web")):
            scene = "web"
        elif any(k in g for k in ("数据库", "sql", "mysql", "查询表")):
            scene = "db"

        if scene is None:
            return tools

        blocklist = cls._SCENE_BLOCKLIST.get(scene, set())
        filtered = {n: m for n, m in tools.items() if n not in blocklist}

        _removed = set(tools.keys()) - set(filtered.keys())
        logger.info(
            "[dsh][orchestrator][debug] filter_builtin_by_scene: "
            "scene=%s, before=%d, after=%d, removed=%r",
            scene, len(tools), len(filtered), sorted(_removed),
        )

        return filtered or tools

    def _select_shortlist(self, goal: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        """工具筛选只做两段：场景过滤 + embedding shortlist。

        决策权交给 LLM prompt，不再有多层兜底。
        """
        picked: Dict[str, Any] = {}

        # 闲聊：不召回任何工具
        try:
            from tools.semantics import classify_intent as _ci
            _g_intent = _ci(goal or "")
            if _g_intent.primary == "chitchat":
                logger.info(
                    "[dsh][orchestrator] chitchat goal=%r, skip all tools",
                    goal[:60],
                )
                return {}
        except Exception:
            logger.exception(
                "[dsh][orchestrator] chitchat pre-check failed, continue"
            )

        builtin_tools = {
            n: m for n, m in tools.items() if not n.startswith("mcp__")
        }
        mcp_tools = {
            n: m for n, m in tools.items() if n.startswith("mcp__")
        }

        # ── 1. 内置工具：场景过滤 + embedding shortlist ──
        if builtin_tools:
            builtin_tools = self._filter_builtin_by_scene(goal, builtin_tools)
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

        # ── 2. MCP 工具：同样走 embedding shortlist ──
        if mcp_tools and self._shortlister_provider is not None:
            try:
                sl = self._shortlister_provider()
                if sl is not None:
                    sub = sl.shortlist(goal, mcp_tools)
                    if sub:
                        picked.update(sub)
                        logger.info(
                            "[dsh][orchestrator] shortlister kept %d/%d MCP tools",
                            len(sub), len(mcp_tools),
                        )
            except Exception:
                logger.exception(
                    "[dsh][orchestrator] shortlister raised for MCP"
                )

        # ── 3. 兜底：picked 为空时用全量 ──
        if not picked:
            picked = dict(tools)

        # ── 4. discovery 入口必须始终可用 ──
        for name in self._MCP_DISCOVERY_TOOLS:
            if name in tools and name not in picked:
                picked[name] = tools[name]
                logger.info(
                    "[dsh][orchestrator] force-kept discovery entry: %s", name
                )

        return picked

    def _assemble_decide_prompt(
        self,
        shortlist: Dict[str, Any],
        history_text: str,
        goal: str,
        conversation_block: str = "",
    ) -> str:
        """按预算组装 decide prompt。"""
        budget = getattr(self, "_prompt_budget_chars", 24000)

        has_discovery = "search_and_connect_mcp" in shortlist

        skeleton = (
            ORCHESTRATOR_PROMPT
            .replace("{goal}", goal)
            .replace("{max_steps}", str(self._max_steps))
        )

        if not has_discovery:
            skeleton = skeleton.replace(
                "## 关于 search_and_connect_mcp 的返回",
                "## 关于外部搜索（当前无 discovery 工具）\n\n"
                "当前工具目录里**没有** `search_and_connect_mcp`，"
                "**不要选它**。你只能用目录里实际存在的工具。\n\n"
                "（以下原始 discovery 说明已停用）\n"
                "## 关于 search_and_connect_mcp 的返回",
            )

        skeleton_len = len(skeleton) - len("{tool_directory}") - len("{history}")
        remaining = max(2000, budget - skeleton_len)

        tool_dir = self._format_tool_directory(shortlist)
        tool_dir_budget = max(1000, int(remaining * 0.35))
        if len(tool_dir) > tool_dir_budget:
            lines = tool_dir.split("\n")
            keep_lines = max(10, tool_dir_budget // 80)
            tool_dir = "\n".join(lines[:keep_lines]) + "\n...(工具列表已截断)"
        remaining -= len(tool_dir)

        if conversation_block:
            conv_budget = max(500, int(remaining * 0.25))
            if len(conversation_block) > conv_budget:
                conversation_block = (
                    conversation_block[:conv_budget] + "\n...(截断)"
                )
            remaining -= len(conversation_block)

        if not history_text:
            history_text = "(尚无步骤)"
        elif remaining <= 0:
            history_text = (
                "...(早期历史已省略)\n" + history_text[-500:]
            )
        elif len(history_text) > remaining:
            history_text = (
                "...(早期历史已省略)\n" + history_text[-remaining:]
            )

        return conversation_block + (
            skeleton
            .replace("{tool_directory}", tool_dir)
            .replace("{history}", history_text)
        )

    async def _decide(
        self,
        goal: str,
        state: OrchestratorState,
        tools: Dict[str, Any],
        prior_messages: Optional[List[Dict[str, str]]] = None,
        expand_recall: bool = False,
    ) -> Optional[OrchestratorStep]:
        if expand_recall:
            shortlist = dict(tools)
            logger.info(
                "[dsh][orchestrator] _decide expand_recall=True, "
                "using full tools (%d)", len(shortlist),
            )
        else:
            shortlist = self._select_shortlist(goal, tools)

        recent = getattr(state, "recent_discovered_tools", set()) or set()
        if recent:
            added = []
            for name in recent:
                if name in tools and name not in shortlist:
                    shortlist[name] = tools[name]
                    added.append(name)
            if added:
                logger.info(
                    "[dsh][orchestrator] re-added recently-discovered tools: %r",
                    added,
                )

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

        memory_facts_block = ""
        if state is not None and getattr(state, "session_id", None):
            try:
                from core.memory.mem0_layer import recall_recent_facts
                _facts = recall_recent_facts(state.session_id, limit=4)
                if _facts:
                    _lines = ["## 本会话已知事实（长期记忆，供决策参考）"]
                    for f in _facts[:4]:
                        _lines.append(f"- {f[:150]}")
                    _lines.append(
                        "\n**注意**：以上是历史事实，可能已过时；"
                        "如果与当前目标无关，忽略即可。"
                    )
                    memory_facts_block = "\n".join(_lines) + "\n\n"
                    logger.info(
                        "[dsh][orchestrator] injected %d memory facts into decide",
                        len(_facts),
                    )
            except Exception:
                logger.exception("[dsh][orchestrator] recall_recent_facts failed")

        history_text = self._format_history(state.history, goal, state)

        if getattr(state, "last_thinking", None):
            history_text += (
                f"\n\n## 上一步思考摘要\n"
                f"{state.last_thinking[:1000]}\n"
            )

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

        prompt = self._assemble_decide_prompt(
            shortlist, history_text, goal, conversation_block + memory_facts_block
        )

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
                    mode="quick",
                    timeout=110.0,
                ),
                timeout=120.0,
            )
            duration_ms = (_time.time() - _t0) * 1000
            logger.info("[dsh][orchestrator] raw LLM response: %r", response)

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
        except ContextLengthExceeded as exc:
            duration_ms = (_time.time() - _t0) * 1000
            logger.warning(
                "[dsh][orchestrator] _decide context exceeded, "
                "fallback to no-history prompt: %s", exc,
            )
            dump_llm_call(
                kind="decide", prompt=prompt, response=None,
                duration_ms=duration_ms, error=f"context_exceeded: {exc}",
            )
            fallback_prompt = (
                ORCHESTRATOR_PROMPT
                .replace("{tool_directory}", self._format_tool_directory(shortlist))
                .replace("{history}", "(已省略，因上下文超限)")
                .replace("{goal}", goal)
                .replace("{max_steps}", str(self._max_steps))
            )
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._llm_provider().generate,
                        [{"role": "user", "content": fallback_prompt}],
                        mode="quick",
                        timeout=110.0,
                    ),
                    timeout=120.0,
                )
            except Exception:
                logger.exception("_decide fallback also failed")
                return None

            data = self._parse_json(response)
            if data is None:
                return None
            raw_name = data.get("tool_name")
            resolved_name = raw_name
            if raw_name and raw_name not in tools:
                resolved = self._resolve_tool_name(raw_name, tools)
                if resolved:
                    resolved_name = resolved
            return OrchestratorStep(
                step_index=len(state.history) + 1,
                thought=str(data.get("thought", "")),
                tool_name=resolved_name,
                parameters={},
                done=bool(data.get("done", False)),
                search_plan=None,
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

        search_plan = None
        raw_plan = data.get("search_plan")
        if isinstance(raw_plan, dict):
            tiers = raw_plan.get("source_tiers")
            if isinstance(tiers, list):
                tiers = [
                    str(t).strip().lower() for t in tiers
                    if str(t).strip().lower() in
                    ("official", "news", "forum", "blog")
                ]
            else:
                tiers = []
            breadth = str(raw_plan.get("breadth") or "").strip().lower()
            if breadth not in ("narrow", "medium", "wide"):
                breadth = ""
            queries = raw_plan.get("queries")
            if isinstance(queries, list):
                queries = [str(q).strip() for q in queries if str(q).strip()]
            else:
                queries = []
            if tiers or breadth or queries:
                search_plan = {
                    "source_tiers": tiers,
                    "breadth": breadth,
                    "queries": queries,
                }
                logger.info(
                    "[dsh][orchestrator] model-declared search_plan: %r",
                    search_plan,
                )

        return OrchestratorStep(
            step_index=len(state.history) + 1,
            thought=str(data.get("thought", "")),
            tool_name=resolved_name,
            parameters={},
            done=bool(data.get("done", False)),
            search_plan=search_plan,
        )

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

    def _intercept_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        state: OrchestratorState,
        tools: Dict[str, Any],
    ) -> Optional[str]:
        """只做幂等拦截。失败切换交给 prompt 约束。"""
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
            if tool_name == "search_knowledge":
                q = params.get("query", "")
                if q:
                    state.searched_queries.add(q)
            return (
                f"⚠️ 条件幂等工具 {tool_name} 与上一步成功调用完全相同"
                f"（参数相同，中间无其他成功调用），已跳过。使用上一步的结果。\n"
                f"**下一步请换一个未搜过的 query**（已搜过：{sorted(state.searched_queries)}）"
            )

        return None

    @staticmethod
    def _classify_status(log: str, success: bool) -> str:
        if "⛔" in log or "❌ 工具调用被 MCP 服务器拒绝" in log:
            return "MANDATORY_NEXT"
        if "💡 MCP 服务器附带了一个可选建议" in log:
            return "OPTIONAL_HINT"
        return "OK" if success else "FAIL"

    @staticmethod
    def _read_tool_schema(meta: Any) -> tuple[Dict[str, Any], List[str]]:
        from tools.schema_utils import read_tool_schema
        return read_tool_schema(meta)

    @staticmethod
    def _format_discovery_schema_brief(meta: Any) -> str:
        from tools.schema_utils import read_tool_schema
        props, required = read_tool_schema(meta)
        required_set = set(required or [])
        lines = []
        for name, info in props.items():
            if not isinstance(info, dict):
                lines.append(f"- `{name}` (any)")
                continue
            ptype = info.get("type", "any")
            mark = "必填" if name in required_set else "可选"
            lines.append(f"- `{name}` ({ptype}, {mark})")
        return "\n".join(lines) if lines else "(该工具无参数)"

    async def _fill_params(
        self,
        goal: str,
        tool_name: str,
        tools: Dict[str, Any],
        state: Optional[OrchestratorState] = None,
        prior_messages: Optional[List[Dict[str, str]]] = None,
        raw_goal: Optional[str] = None,
        search_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """阶段 2：给单个工具填参数。

        路径参数不交给 LLM 猜——prompt 里给 workspace 真实路径，
        LLM 只需按 schema 填内容类字段。
        """
        meta = tools.get(tool_name)
        if meta is None:
            logger.warning(
                "[dsh][orchestrator] _fill_params: %s not in tools", tool_name
            )
            return {}

        _props, _required = self._read_tool_schema(meta)
        _is_mcp = tool_name.startswith("mcp__")
        _is_discovery = tool_name in self._MCP_DISCOVERY_TOOLS

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

        if (_is_discovery or _is_mcp) and not _props:
            _props = getattr(meta, "parameters", None) or {}
            if not _required and hasattr(meta, "required"):
                _required = list(getattr(meta, "required", []) or [])
            logger.info(
                "[dsh][orchestrator] _fill_params: %s schema empty, "
                "fallback to parameters: %r, required: %r",
                tool_name, list(_props.keys()), _required,
            )

        schema_text = self._format_single_tool_schema(meta)

        if _is_discovery:
            schema_text = self._format_discovery_schema_brief(meta)

        tool_desc = getattr(meta, "description", "") or ""

        history_text = ""
        if state is not None and state.history:
            compact_steps = []
            for h in state.history[-5:]:
                idx = h.find(" → ")
                if idx > 0:
                    compact_steps.append(h[:idx])
                else:
                    compact_steps.append(h[:200])
            history_text = (
                "## 已执行步骤（只列动作，产出见下方文件路径）\n"
                + "\n".join(compact_steps)
                + "\n\n"
            )

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
            path = state.last_read_path or "(未知)"
            if len(content) > 4000:
                total_lines = content.count("\n") + 1
                file_content_block = (
                    f"## 最近读取的文件\n"
                    f"路径：`{path}`（{total_lines} 行，{len(content)} 字符）\n"
                    f"**文件较大，未全部放入 prompt。两种修改方式**：\n"
                    f"1. **按行替换（推荐）**：先用 `read_file_range` 读目标行范围，"
                    f"确认后调用 `edit_file` 传 `start_line`/`end_line`/`new_text`。\n"
                    f"2. **逐字替换**：`old_text` 必须逐字匹配且唯一。\n"
                    f"前 2000 字符预览：\n```\n{content[:2000]}\n```\n\n"
                )
            else:
                file_content_block = (
                    f"## 最近读取的文件内容\n"
                    f"路径：`{path}`\n"
                    f"内容（**`old_text` 必须逐字匹配这里的内容**）：\n"
                    f"```\n{content}\n```\n"
                )

        conversation_block = ""
        if prior_messages and tool_name not in self._MCP_DISCOVERY_TOOLS:
            last_user = None
            for m in reversed(prior_messages):
                if m.get("role") == "user" and (m.get("content") or "").strip():
                    last_user = m
                    break
            if last_user:
                content = (last_user.get("content") or "").strip()
                if len(content) > 400:
                    content = content[:400] + "..."
                conversation_block = (
                    "## 上一轮用户消息（供填参参考）\n"
                    f"- {content}\n\n"
                )

        mem0_block = ""
        if (
            tool_name not in self._MCP_DISCOVERY_TOOLS
            and state is not None
            and getattr(state, "session_id", None)
        ):
            try:
                from core.memory.mem0_layer import recall_file_paths
                known_paths = recall_file_paths(state.session_id, limit=5)
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

        workspace_block = ""
        if state is not None:
            ws_roots = getattr(state, "workspace_roots", None)
            if not ws_roots:
                ws_roots = [os.getcwd()]
            if ws_roots:
                workspace_block = (
                    "## 工作区真实路径（重要）\n"
                    + "\n".join(f"- `{p}`" for p in ws_roots)
                    + "\n\n**文件工具（read_text_file / write_file / edit_file / "
                    "list_files / search_content 等）的 path 参数必须用上面的绝对路径。"
                    "不要用进程 cwd 或相对路径。**\n\n"
                )

        coverage_block = ""
        if tool_name == "search_knowledge" and state is not None:
            target_raw = getattr(state, "_target_files_cache", None) or set()
            target = {_norm_filename(f) for f in target_raw if f}
            covered = {_norm_filename(f) for f in state.covered_documents if f}
            searched = state.searched_queries
            if target:
                missing = target - covered
                coverage_block = (
                    "## 多文档覆盖进度（重要）\n"
                    f"- 目标文档全集（{len(target)} 个）：{sorted(target)}\n"
                    f"- 已覆盖（{len(covered)} 个）：{sorted(covered)}\n"
                    f"- **待覆盖（{len(missing)} 个）**：{sorted(missing)}\n"
                )
                if missing:
                    coverage_block += (
                        f"- **本次请对 `{sorted(missing)[0]}` 调用 search_knowledge**，"
                        f"query 用该文档的主题或文件名\n"
                    )
                else:
                    coverage_block += "- **所有文档已覆盖，可以 done=true**\n"
                coverage_block += "\n"
            if searched:
                coverage_block += (
                    f"## 已搜索过的 query（不要重复）\n"
                    f"{sorted(searched)}\n\n"
                )

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

        search_plan_block = ""
        _is_web_tool = (
            tool_name.startswith("mcp__tavily")
            or tool_name == "fetch_url"
            or tool_name.startswith("mcp__fetch")
        )
        if _is_web_tool and search_plan:
            _tiers = search_plan.get("source_tiers") or []
            _breadth = search_plan.get("breadth") or ""
            _queries = search_plan.get("queries") or []
            _lines = ["## 搜索覆盖计划（你上一步声明的，填参时请遵循）"]
            if _tiers:
                _lines.append(
                    f"- 目标来源类型：{_tiers}（official=官方/一手，"
                    f"news=新闻/媒体，forum=论坛/社区，blog=博客/个人）"
                )
            if _breadth:
                _lines.append(
                    f"- 覆盖深度：{_breadth}"
                    f"（narrow=1~2 次，medium=3~5 次，wide=6+ 次）"
                )
            if _queries:
                _lines.append(f"- 计划查询词：{_queries}")
                _lines.append(
                    "- **如果工具 schema 支持 `queries` 数组，直接填这个数组；"
                    "如果只支持单个 `query`，填第一个查询词**，"
                    "其余由编排层在后续步按计划继续搜。"
                )
            _lines.append(
                "- 如果 schema 有 domain 过滤参数，可把 official 类域名"
                "（如官方文档站）放进 include_domains。"
            )
            search_plan_block = "\n".join(_lines) + "\n\n"

        # ── 剥离 goal 开头的路径前缀，避免 LLM 把 goal 当 file_path ──
        def _strip_goal_path_prefix(text: str) -> str:
            if not text:
                return text
            m = re.match(
                r"^\s*(?:[A-Za-z]:[\\/]|/)[^\s]*?[\\/](?=[\u4e00-\u9fff]|\s|$)",
                text,
            )
            if m:
                stripped = text[m.end():].strip()
                if stripped:
                    return stripped
            return text

        task_text = _strip_goal_path_prefix(raw_goal or goal)
        prompt = (
            f"## 任务\n{task_text}\n\n"
            f"{conversation_block}"
            f"{mem0_block}"
            f"{workspace_block}"
            f"{coverage_block}"
            f"{history_text}"
            f"{file_content_block}"
            f"## 你要调用的工具\n"
            f"`{tool_name}`：{tool_desc}\n\n"
            f"{discovery_block}"
            f"{search_plan_block}"
            f"## 参数表\n{schema_text}\n\n"
            f"## 路径参数规则（强制）\n"
            f"1. **路径类参数（file_path / path / output_path）必须用"
            f"「工作区真实路径」里的绝对路径 + 一个合理文件名**。\n"
            f"2. **绝对不要从「任务」描述里抄路径**——任务描述里的路径"
            f"是用户上下文，不是输出路径。\n"
            f"3. **如果「工作区真实路径」为空，用 `output.<扩展名>` 作为文件名**，"
            f"不要用任务描述里的任何路径。\n"
            f"4. **绝对不要编造不存在的目录**（不要用 `D:/data`、`/tmp` 等占位）。\n\n"
            f"## 输出要求\n"
            f"- 只输出**一个** JSON 对象，代表**本次一次**工具调用。\n"
            f"- **禁止**输出多个 JSON 对象、JSON 数组。\n"
            f"- 不要任何解释，不要包裹在 markdown 代码块里。\n"
            f"- 确保 JSON 合法、字段名与**参数表完全一致**——"
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
                    mode="quick",
                    timeout=110.0,
                ),
                timeout=120.0,
            )
            duration_ms = (_time.time() - _t0) * 1000
            logger.info(
                "[dsh][orchestrator] _fill_params raw response for %s: %r",
                tool_name, raw,
            )

            dump_llm_call(
                kind="fill_params",
                prompt=prompt,
                response=raw,
                model=getattr(self._llm_provider(), "model", None),
                duration_ms=duration_ms,
                tool_name=tool_name,
                extra={"goal": goal[:200]},
            )
        except ContextLengthExceeded as exc:
            duration_ms = (_time.time() - _t0) * 1000
            logger.warning(
                "[dsh][orchestrator] _fill_params context exceeded for %s: %s",
                tool_name, exc,
            )
            dump_llm_call(
                kind="fill_params", prompt=prompt, response=None,
                duration_ms=duration_ms, tool_name=tool_name,
                error=f"context_exceeded: {exc}",
            )
            fallback: Dict[str, Any] = {}
            if _required:
                for k in _required:
                    if k in ("query", "q", "keyword", "search_query"):
                        fallback[k] = goal
            return fallback
        except asyncio.TimeoutError:
            duration_ms = (_time.time() - _t0) * 1000
            logger.warning(
                "[dsh][orchestrator] _fill_params timed out (120s) for %s, "
                "fallback required params with goal",
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
            fallback: Dict[str, Any] = {}
            if _required:
                for k in _required:
                    if k in ("query", "q", "keyword", "search_query"):
                        fallback[k] = goal
            return fallback
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
                tool_name, (raw or "")[:200],
            )
            fallback: Dict[str, Any] = {}
            if _required:
                for k in _required:
                    if k in ("query", "q", "keyword", "search_query"):
                        fallback[k] = goal
            if fallback:
                logger.info(
                    "[dsh][orchestrator] _fill_params: %s parse failed, "
                    "fallback required params %r with goal",
                    tool_name, list(fallback.keys()),
                )
                return fallback
            return {}

        adapted = data

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

        if _is_discovery:
            if not adapted.get("user_query"):
                adapted["user_query"] = goal
                logger.info(
                    "[dsh][orchestrator] _fill_params: %s user_query empty, "
                    "filled with goal", tool_name,
                )
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

        if _required:
            missing = [k for k in _required if not adapted.get(k)]
            if missing:
                for k in missing:
                    if k in ("query", "q", "keyword", "search_query"):
                        adapted[k] = goal
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "required %r empty, filled with goal",
                            tool_name, k,
                        )
                    elif k in ("path", "file_path"):
                        ws = (
                            getattr(state, "workspace_roots", None)
                            if state else None
                        )
                        _base_dir = ws[0] if ws else os.getcwd()
                        if "docx" in tool_name or "word" in tool_name:
                            _ext = ".docx"
                        elif "excel" in tool_name or "xlsx" in tool_name:
                            _ext = ".xlsx"
                        elif "pptx" in tool_name or "ppt" in tool_name:
                            _ext = ".pptx"
                        elif "pdf" in tool_name:
                            _ext = ".pdf"
                        else:
                            _ext = ".txt"
                        adapted[k] = os.path.join(_base_dir, f"output{_ext}")
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "required %r empty, filled with default: %r "
                            "(workspace=%r)",
                            tool_name, k, adapted[k],
                            bool(ws),
                        )
                    elif k == "url":
                        import re as _re
                        m = _re.search(r'https?://[^\s"\']+', goal)
                        if m:
                            adapted[k] = m.group(0)

        for _pk in ("file_path", "path", "output_path"):
            _v = adapted.get(_pk)
            if isinstance(_v, str) and len(_v) > 200:
                logger.warning(
                    "[dsh][orchestrator] _fill_params: %s %s looks like "
                    "goal text (len=%d), dropping",
                    tool_name, _pk, len(_v),
                )
                adapted.pop(_pk, None)

        return adapted

    _VERIFY_TOOL_HINTS = (
        "run_test", "pytest", "test_runner", "jest", "vitest", "mocha",
        "run_python", "run_shell", "run_command", "run_script",
        "bash", "shell", "execute", "exec",
        "lint", "typecheck", "check",
    )

    @classmethod
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

    @classmethod
    def _goal_needs_verification(cls, goal: str) -> bool:
        if not goal:
            return False
        import re as _re
        return bool(_re.search(cls._VERIFY_PATTERN, goal, _re.I))

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

        if SecurityGateway.is_write_operation(tool_name):
            wp = self._extract_path_from_params(params)
            if wp:
                state.written_paths.add(wp)
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
    def _extract_path_from_params(cls, params: Dict[str, Any]) -> Optional[str]:
        if not isinstance(params, dict):
            return None
        for key in cls._OUTPUT_PATH_KEYS:
            v = params.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    async def run(
        self,
        goal: str,
        allowed_danger_levels: Optional[set] = None,
        allowed_tool_names: Optional[set] = None,
        allow_mcp_discovery: bool = True,
        workspace_paths: Optional[List[str]] = None,
        resume_state: Optional[OrchestratorState] = None,
        resume_from_step: int = 1,
        prior_messages: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        task_type: Optional[TaskType] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        logger.info(
            "[dsh][orchestrator] run() session_id=%r, "
            "resume_state=%r, goal=%r",
            session_id,
            resume_state is not None,
            goal[:100],
        )

        try:
            from core.memory.mem0_layer import wait_until_ready
            ready = await wait_until_ready(timeout=30.0)
            logger.info("[dsh][orchestrator] mem0 ready=%r", ready)
        except Exception:
            logger.exception("wait_until_ready 失败，继续执行")

        try:
            from tools.mcp_discovery import reset_tried_servers
            reset_tried_servers()
        except Exception:
            logger.exception("failed to reset tried MCP servers")

        agent = self._agent_provider()

        raw_goal = goal
        try:
            from tools.skills.session_skills import build_skills_prompt_block
            skills_block = build_skills_prompt_block(session_id)
            if skills_block:
                goal = f"{goal}\n\n{skills_block}"
                logger.info(
                    "[dsh][skills] injected skills block (%d chars) for session=%r",
                    len(skills_block), session_id,
                )
        except Exception:
            logger.exception("[dsh][skills] 注入 skills prompt 失败，继续执行")

        def _refresh_tools() -> Dict[str, Any]:
            t = agent.list_all_tools(allowed_danger_levels=allowed_danger_levels)
            if not allow_mcp_discovery:
                t = {
                    name: meta for name, meta in t.items()
                    if name not in self._MCP_DISCOVERY_TOOLS
                }
            t = {
                name: meta for name, meta in t.items()
                if name not in self._UNAVAILABLE_TOOLS
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

        if resume_state is not None:
            state = resume_state
            if state.session_id is None:
                state.session_id = session_id
            if state.task_id is None:
                state.task_id = task_id
            start_step = max(1, resume_from_step)
            logger.info(
                "[dsh][orchestrator] resuming from step %d, history=%d entries, "
                "session_id=%r",
                start_step, len(state.history), state.session_id,
            )
        else:
            state = OrchestratorState(
                goal=goal, session_id=session_id, task_id=task_id,
                task_type=task_type or classify_task(goal),
            )
            start_step = 1

        if workspace_paths:
            state.workspace_roots = list(workspace_paths)
        elif resume_state is not None and resume_state.workspace_roots:
            state.workspace_roots = list(resume_state.workspace_roots)

        try:
            if not hasattr(state, "successful_tool_sequence"):
                object.__setattr__(
                    state, "successful_tool_sequence",
                    _SucceededSeqView(state),
                )
        except Exception:
            logger.exception("failed to attach successful_tool_sequence view")

        for step_index in range(start_step, self._max_steps + 1):
            tools = _refresh_tools()

            if allow_mcp_discovery:
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

            decision = await self._decide(
                goal, state, tools, prior_messages=prior_messages,
            )

            if (
                decision is not None
                and decision.tool_name == "__need_more_tools__"
                and not getattr(state, "_expanded_recall", False)
            ):
                logger.info(
                    "[dsh][orchestrator] LLM requested __need_more_tools__, "
                    "expanding recall"
                )
                state._expanded_recall = True
                state.add_history(
                    f"Step {step_index}: 工具目录不足，扩大召回重试"
                )
                yield {
                    "type": "thinking",
                    "step": step_index,
                    "thought": "[扩大召回] 工具目录不足，重新召回",
                }
                decision = await self._decide(
                    goal, state, tools, prior_messages=prior_messages,
                    expand_recall=True,
                )

            if (
                decision is not None
                and decision.tool_name == "__need_more_tools__"
                and getattr(state, "_expanded_recall", False)
            ):
                logger.warning(
                    "[dsh][orchestrator] __need_more_tools__ again after "
                    "expand, force done"
                )
                state.stop_reason = StopReason.PARTIAL.value
                state.stop_detail = "当前工具集确实无法完成该任务"
                state.add_history(
                    f"Step {step_index}: 工具集不足（已扩大召回仍无解），终止"
                )
                yield {
                    "type": "done",
                    "steps": step_index - 1,
                    "reason": StopReason.PARTIAL.value,
                    "detail": state.stop_detail,
                }
                return

            if decision is None:
                yield {"type": "error", "message": "编排决策失败"}
                return

            if decision.tool_name and not decision.done:
                if decision.search_plan:
                    state.last_search_plan = decision.search_plan
                decision.parameters = await self._fill_params(
                    goal, decision.tool_name, tools, state,
                    prior_messages=prior_messages,
                    raw_goal=raw_goal,
                    search_plan=decision.search_plan,
                )
                logger.info(
                    "[dsh][orchestrator] filled params for %s: %r",
                    decision.tool_name, decision.parameters,
                )
                _meta = tools.get(decision.tool_name)
                _props, _required = (
                    self._read_tool_schema(_meta) if _meta else ({}, [])
                )
                _params = decision.parameters or {}
                _missing = [
                    k for k in _required
                    if k not in _params or _params.get(k) in (None, "")
                ]
                if _missing:
                    state.add_history(
                        f"Step {step_index}: {decision.tool_name} "
                        f"缺少必填参数 {_missing}，已跳过"
                    )
                    logger.warning(
                        "[dsh][orchestrator] %s missing required %r, skip step",
                        decision.tool_name, _missing,
                    )
                    yield {
                        "type": "thinking",
                        "step": step_index,
                        "thought": (
                            f"[参数缺失] {decision.tool_name} 缺 {_missing}，跳过"
                        ),
                    }
                    continue
                if decision.tool_name == "search_knowledge":
                    q = decision.parameters.get("query")
                    if q:
                        state.searched_queries.add(q)

            yield {
                "type": "thinking",
                "step": step_index,
                "thought": decision.thought,
                "log": decision.thought,
            }
            if decision.thought:
                state.last_thinking = str(decision.thought)[:1000]

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
                state.stop_reason = StopReason.DONE.value
                state.stop_detail = ""
                yield {
                    "type": "done",
                    "steps": step_index - 1,
                    "reason": StopReason.DONE.value,
                }
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
                hint = ""
                if tool_name in self._MCP_DISCOVERY_TOOLS:
                    hint = (
                        f"（`{tool_name}` 在当前模式下不可用——"
                        f"ask / 关闭 discovery 时不扩展外部能力。"
                        f"请从目录里选实际存在的工具："
                        f"fetch / http_get / search_knowledge 等）"
                    )
                state.add_history(
                    f"Step {step_index}: 尝试调用无效工具 {tool_name}，已忽略。"
                    f"{hint}可用工具（前 10 个）：{available}..."
                )
                logger.warning(
                    "[dsh][orchestrator] unresolved tool name %r; available=%r; "
                    "is_discovery=%r",
                    tool_name, list(tools.keys()),
                    tool_name in self._MCP_DISCOVERY_TOOLS,
                )
                yield {
                    "type": "thinking",
                    "step": step_index,
                    "thought": (
                        f"[决策错误] 模型选择了不存在的工具 {tool_name}，已忽略。"
                        f"{hint}可用工具：{available}"
                    ),
                }
                continue

            params = decision.parameters or {}

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

            from tools.security.gateway import (
                get_gateway, get_security_context,
            )
            from tools.security.approval import get_approval_store

            sec_ctx = get_security_context()
            sec_meta = tools.get(tool_name)
            sec_decision = (
                get_gateway().check(tool_name, params, sec_meta, sec_ctx)
                if sec_ctx is not None else None
            )

            if sec_decision is not None and not sec_decision.allowed:
                if sec_decision.needs_approval:
                    store = get_approval_store()
                    rec = store.create(
                        task_id=getattr(state, "task_id", "") or "",
                        session_id=getattr(state, "session_id", "") or "",
                        tool_name=tool_name,
                        params=params,
                    )
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
                        "reason": sec_decision.reason,
                        "approval_id": rec.approval_id,
                        "state_snapshot": state.snapshot(),
                        "next_step": step_index + 1,
                    }
                    return
                state.add_history(
                    f"Step {step_index}: 安全网关拒绝 {tool_name} —— {sec_decision.reason}"
                )
                yield {
                    "type": "thinking",
                    "step": step_index,
                    "thought": f"[安全网关] 拒绝 {tool_name}: {sec_decision.reason}",
                }
                continue

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

            def _strip_format_noise(log: str) -> str:
                if not log:
                    return ""
                lines = []
                for line in log.split("\n"):
                    s = line.strip()
                    if s.startswith("✅ 工具 [") or s.startswith("⏱️ 耗时:"):
                        continue
                    if s in ("📦 其他数据:", "📊 结构化数据:", "📄 MCP 返回:",
                            "📊 MCP 数据:", "🔧 已连接 MCP 服务器。"):
                        continue
                    lines.append(line)
                return "\n".join(lines).strip()

            log = agent.format_tool_result(result)
            status = self._classify_status(log, result.success)
            limit_chars = self._HISTORY_LIMITS.get(
                tool_name, self._DEFAULT_HISTORY_LIMIT
            )
            log_for_history = _strip_format_noise(log)
            state.add_history(
                f"Step {step_index}: {tool_name}({params}) → {status}: "
                f"{log_for_history[:limit_chars]}"
            )

            if result.success:
                state.mark_success(tool_name, params, step_index)

                _is_web_tool_now = (
                    tool_name.startswith("mcp__tavily")
                    or tool_name == "fetch_url"
                    or tool_name.startswith("mcp__fetch")
                )
                if _is_web_tool_now:
                    try:
                        _buckets = self._bucket_web_results(result.result)
                        state.web_source_buckets = _buckets
                        _counts = {k: len(v) for k, v in _buckets.items()}
                        logger.info(
                            "[dsh][orchestrator] web source buckets: %r",
                            _counts,
                        )
                        _plan = getattr(state, "last_search_plan", None) or {}
                        _wanted = set(_plan.get("source_tiers") or [])
                        _missing = _wanted - set(_buckets.keys())
                        if _wanted and _missing:
                            state.add_history(
                                f"Step {step_index} 搜索覆盖提示："
                                f"计划覆盖 {sorted(_wanted)}，"
                                f"实际缺少 {sorted(_missing)}"
                                f"（实际分桶：{_counts}）"
                            )
                            logger.info(
                                "[dsh][orchestrator] search plan deviation: "
                                "wanted=%r missing=%r actual=%r",
                                sorted(_wanted), sorted(_missing), _counts,
                            )
                    except Exception:
                        logger.exception("bucket web results failed")

                if tool_name in self._MCP_DISCOVERY_TOOLS:
                    try:
                        payload = result.result
                        new_tools = []
                        if isinstance(payload, dict):
                            tl = payload.get("tools") or []
                            if isinstance(tl, list):
                                new_tools = [t for t in tl if isinstance(t, str)]
                        if new_tools:
                            state.recent_discovered_tools.update(new_tools)
                            if len(state.recent_discovered_tools) > 10:
                                state.recent_discovered_tools = set(
                                    sorted(state.recent_discovered_tools)[-10:]
                                )
                            logger.info(
                                "[dsh][orchestrator] discovery registered %d tools: %r",
                                len(new_tools), new_tools,
                            )
                    except Exception:
                        logger.exception("record discovered tools failed")

                from tools.security.gateway import SecurityGateway
                if SecurityGateway.is_write_operation(tool_name):
                    _wp = self._extract_path_from_params(params)
                    if _wp:
                        state.written_paths.add(_wp)

                    if self._goal_needs_verification(goal):
                        verify = await self._collect_verification(goal, state, tools)
                        if verify is not None:
                            if verify.should_terminate:
                                state.stop_reason = verify.stop_reason.value
                                state.stop_detail = verify.gap
                                state.add_history(
                                    f"Step {step_index} 校验："
                                    f"{verify.stop_reason.value} — {verify.gap}"
                                )
                                yield {
                                    "type": "done",
                                    "steps": step_index,
                                    "reason": verify.stop_reason.value,
                                    "detail": verify.gap,
                                }
                                return
                            if verify.gap:
                                state.add_history(
                                    f"Step {step_index} 验证提示：{verify.gap}\n"
                                    f"建议：{verify.next_hint}"
                                )
                                logger.info(
                                    "[dsh][orchestrator] verification hint: %s",
                                    verify.gap[:120],
                                )
                                state._verification_hint_count += 1
                                if state._verification_hint_count >= 3:
                                    state.stop_reason = StopReason.PARTIAL.value
                                    state.stop_detail = (
                                        "已修改但未能自动验证"
                                        "（连续 3 次验证提示未响应）"
                                    )
                                    yield {
                                        "type": "done",
                                        "steps": step_index,
                                        "reason": StopReason.PARTIAL.value,
                                        "detail": state.stop_detail,
                                    }
                                    return

                if tool_name in ("knowledge_stats", "list_knowledge_sources"):
                    try:
                        payload = result.result
                        if isinstance(payload, dict):
                            files = payload.get("files") or payload.get("sources") or []
                            names = set()
                            for f in files:
                                if isinstance(f, dict) and f.get("filename"):
                                    norm = _norm_filename(f["filename"])
                                    if norm:
                                        names.add(norm)
                            if names:
                                object.__setattr__(state, "_target_files_cache", names)
                                logger.info(
                                    "[dsh][orchestrator] cached target files: %d %r",
                                    len(names), sorted(names),
                                )
                    except Exception:
                        logger.exception("cache target files failed")

                assessment = await self._assess_result(
                    goal, tool_name, params, result, state
                )
                if assessment is not None and not assessment.sufficient:
                    gap = assessment.gap
                    hint = assessment.next_hint
                    if tool_name in self._MCP_DISCOVERY_TOOLS:
                        cap = params.get("capability", "")
                        for kw in cap.split(","):
                            kw = kw.strip().lower()
                            if kw:
                                state.failed_capabilities.add(kw)

                    if assessment.should_terminate:
                        state.stop_reason = assessment.stop_reason.value
                        state.stop_detail = gap
                        state.add_history(
                            f"Step {step_index} 终止："
                            f"{assessment.stop_reason.value} — {gap}"
                        )
                        logger.info(
                            "[dsh][orchestrator] terminate: reason=%s gap=%r",
                            assessment.stop_reason.value, gap[:120],
                        )
                        yield {
                            "type": "done",
                            "steps": step_index,
                            "reason": assessment.stop_reason.value,
                            "detail": gap,
                        }
                        return

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

                try:
                    from core.memory.mem0_layer import remember_tool_success
                    _meta_for_origin = tools.get(tool_name)
                    _origin = getattr(_meta_for_origin, "origin", None) or "builtin"
                    remember_tool_success(
                        session_id=state.session_id or "",
                        tool_name=tool_name,
                        params=params,
                        result=result.result,
                        origin=_origin,
                    )
                except Exception:
                    logger.exception("remember_tool_success 失败")

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
                _fk = (tool_name, params_key(params))
                state._consecutive_failures[_fk] = (
                    state._consecutive_failures.get(_fk, 0) + 1
                )

            yield {
                "type": "tool_result",
                "step": step_index,
                "tool_call": tool_call,
                "result": result,
                "log": log,
            }

        state.stop_reason = StopReason.MAX_STEPS.value
        state.stop_detail = f"已达最大步数 {self._max_steps}"
        yield {
            "type": "max_steps",
            "steps": self._max_steps,
            "reason": StopReason.MAX_STEPS.value,
        }