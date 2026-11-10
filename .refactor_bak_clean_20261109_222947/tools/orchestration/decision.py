"""AgentOrchestrator 的决策 Mixin：工具选择（_decide + _select_shortlist）。

从 core.py 按行号提取（B8-a），方法体零改动。
AgentOrchestrator 改为 `class AgentOrchestrator(DecisionMixin, ...)` 后，
self._select_shortlist / self._decide 自动可用。

依赖的辅助方法（_format_history / _parse_json / _is_external_fact_query /
_read_tool_schema / _filter_builtin_by_scene）仍在 core.py，Mixin 里
self._xxx(...) 调用时 MRO 会找到 AgentOrchestrator 的实现。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from infra.debug_log.llm import dump_llm_call

from .prompt import ORCHESTRATOR_PROMPT
from .state import OrchestratorState, params_key
from .helpers import _build_structured_history
from tools.semantics import get_intent_tags
from .task_types import TaskType
from core.llm.provider import ContextLengthExceeded

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorStep:
    """从 core.py 搬过来（原 L100 附近）。"""
    step_index: int
    thought: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    search_plan: Optional[Dict[str, Any]] = None


class DecisionMixin:
    """工具选择相关方法。self 是 AgentOrchestrator。"""
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
                lines.append(f"    数组元素类型: {itype}")
                iprops = items.get("properties")
                if isinstance(iprops, dict) and iprops:
                    irequired = set(items.get("required") or [])
                    for iname, iinfo in iprops.items():
                        if not isinstance(iinfo, dict):
                            continue
                        itype2 = iinfo.get("type", "any")
                        imark = "必填" if iname in irequired else "可选"
                        idesc = iinfo.get("description", "")
                        sub = f"      - `{iname}` ({itype2}, {imark})"
                        if idesc:
                            sub += f": {idesc}"
                        lines.append(sub)
        return "\n".join(lines)

    @staticmethod
    def _resolve_tool_name(requested: str, available: Dict[str, Any]) -> Optional[str]:
        """把 LLM 幻觉出的简写工具名映射到实际存在的工具（通用，不特化）。"""
        if not requested:
            return None
        if requested in available:
            return requested

        # 1. 后缀匹配（mcp__server__requested）
        suffix = f"__{requested}"
        matches = [n for n in available if n.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]

        # 2. 包含匹配
        contains = [n for n in available if requested in n]
        if len(contains) == 1:
            return contains[0]

        # 3. 操作词 + 格式词同义词映射（通用）
        _OPERATION_SYNONYMS = {
            "write": {"write", "create", "make", "generate", "build", "produce", "render", "save", "export"},
            "read": {"read", "get", "load", "open", "show", "display", "view"},
            "search": {"search", "find", "query", "lookup", "retrieve"},
            "delete": {"delete", "remove", "rm", "drop", "purge"},
            "edit": {"edit", "modify", "update", "patch", "change"},
            "move": {"move", "rename", "relocate"},
            "list": {"list", "ls", "dir", "enumerate", "files"},
            "convert": {"convert", "transform"},
        }
        _FORMAT_SYNONYMS = {
            "pptx": {"ppt", "pptx", "powerpoint", "presentation", "slide", "slides", "deck"},
            "docx": {"doc", "docx", "word", "document", "report"},
            "xlsx": {"xls", "xlsx", "excel", "spreadsheet", "sheet", "table"},
            "pdf": {"pdf"},
            "text": {"txt", "text", "file"},
            "csv": {"csv"},
            "json": {"json"},
            "md": {"md", "markdown"},
        }

        def _split_name(name: str):
            """拆 name 为 (operation_group, format_group) 或 None。"""
            base = name.rsplit("__", 1)[-1]
            parts = base.split("_", 1)
            if len(parts) != 2:
                return None
            prefix, rest = parts
            op_group = None
            for g, members in _OPERATION_SYNONYMS.items():
                if prefix in members:
                    op_group = g
                    break
            fmt_group = None
            for g, members in _FORMAT_SYNONYMS.items():
                if rest in members:
                    fmt_group = g
                    break
            return (op_group, fmt_group)

        req = _split_name(requested)
        if req is not None:
            req_op, req_fmt = req
            for name in available:
                cand = _split_name(name)
                if cand is None:
                    continue
                cand_op, cand_fmt = cand
                if req_op is not None and req_op != cand_op:
                    continue
                if req_fmt is not None and req_fmt != cand_fmt:
                    continue
                if req_op is not None or req_fmt is not None:
                    return name

        # 4. 编辑距离 ≤ 2 兜底（只对短名）
        if len(requested) <= 20:
            def _edit_distance(a: str, b: str) -> int:
                if len(a) < len(b):
                    return _edit_distance(b, a)
                if not b:
                    return len(a)
                prev = list(range(len(b) + 1))
                for i, ca in enumerate(a, 1):
                    cur = [i]
                    for j, cb in enumerate(b, 1):
                        cur.append(min(
                            prev[j] + 1,
                            cur[j - 1] + 1,
                            prev[j - 1] + (ca != cb),
                        ))
                    prev = cur
                return prev[-1]

            for name in available:
                base = name.rsplit("__", 1)[-1]
                if _edit_distance(requested, base) <= 2:
                    return name

        return None
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


        # ── 工具依赖图补全（通用规则）──
        # 如果 shortlist 选中了 key，把它的前置工具也加进来
        for _name in list(picked.keys()):
            _prereqs = self._TOOL_PREREQUISITES.get(_name, frozenset())
            for _p in _prereqs:
                if _p in tools and _p not in picked:
                    picked[_p] = tools[_p]

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
            logger.warning(
                "[dsh][orchestrator] _decide LLM timed out after 120s, "
                "fallback to rule-based"
            )
            dump_llm_call(
                kind="decide",
                prompt=prompt,
                response=None,
                duration_ms=duration_ms,
                error="timeout(120s), fallback to rule-based",
                extra={"tools_in_prompt": list(shortlist.keys())},
            )
            # ── 降级：用规则选工具（不返回 None）──
            return self._fallback_decide(goal, tools, state)
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


    def _fallback_decide(self, goal: str, tools: Dict[str, Any], state: Any = None):
        """LLM 超时降级：规则选工具（不返回 None）。"""
        from tools.semantics import classify_intent

        try:
            intent = classify_intent(goal or "")
        except Exception:
            intent = None

        _FALLBACK_MAP = {
            "generate": ["write_docx", "create_word", "create_pptx", "write_excel"],
            "modify": ["edit_file", "edit_docx", "edit_excel"],
            "read": ["read_text_file", "read_pdf_text", "read_docx_text"],
            "query": ["search_knowledge", "http_get", "fetch_url"],
            "code": ["read_text_file", "search_content"],
        }

        primary = getattr(intent, "primary", "") if intent else ""
        secondary = set(getattr(intent, "secondary", frozenset())) if intent else set()
        # 通用：generate 时按 secondary 选工具（而不是固定顺序）
        if primary == "generate":
            if "presentation" in secondary:
                candidates = ["create_pptx", "office_batch"]
            elif "spreadsheet" in secondary:
                candidates = ["write_excel", "create_excel"]
            elif "document" in secondary:
                candidates = ["write_docx", "create_word"]
            elif "pdf" in secondary:
                candidates = ["convert_to_pdf", "render_document"]
            else:
                candidates = _FALLBACK_MAP.get(primary, [])
        elif primary == "modify":
            if "document" in secondary:
                candidates = ["edit_docx", "edit_file"]
            elif "spreadsheet" in secondary:
                candidates = ["edit_excel", "edit_file"]
            else:
                candidates = _FALLBACK_MAP.get(primary, [])
        elif primary == "read":
            if "pdf" in secondary:
                candidates = ["read_pdf_text", "read_pdf_metadata"]
            elif "document" in secondary:
                candidates = ["read_docx_text", "read_office"]
            elif "spreadsheet" in secondary:
                candidates = ["read_excel", "read_csv"]
            else:
                candidates = _FALLBACK_MAP.get(primary, [])
        else:
            candidates = _FALLBACK_MAP.get(primary, [])
        step_index = len(getattr(state, "history", [])) + 1 if state else 1

        for name in candidates:
            if name in tools:
                return OrchestratorStep(
                    step_index=step_index,
                    thought=f"[降级] LLM 超时，规则选 {name}",
                    tool_name=name,
                    parameters={},
                    done=False,
                )

        for name in tools:
            return OrchestratorStep(
                step_index=step_index,
                thought=f"[降级] LLM 超时，选第一个工具 {name}",
                tool_name=name,
                parameters={},
                done=False,
            )

        return OrchestratorStep(
            step_index=step_index,
            thought="[降级] LLM 超时且无可用工具",
            done=True,
        )
