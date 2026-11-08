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



# ── OrchestratorStep + DecisionMixin 从 decision.py 取（B8-a）──
from .decision import OrchestratorStep, DecisionMixin
from .fill_params import FillParamsMixin
from .assess import AssessMixin
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


class AgentOrchestrator(DecisionMixin, FillParamsMixin, AssessMixin):
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

    @classmethod
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
