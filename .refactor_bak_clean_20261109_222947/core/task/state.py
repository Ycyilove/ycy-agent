"""Orchestrator session state（B11-1 搬到 core/task/）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .task_types import TaskType


def params_key(params: Any) -> str:
    """把参数字典变成稳定的 key。"""
    if not params:
        return "{}"
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(params)


@dataclass
class OrchestratorState:
    """多步编排的会话状态，跨步骤传递。"""
    goal: str = ""
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    history: List[str] = field(default_factory=list)

    successful_calls: Dict[Tuple[str, str], int] = field(default_factory=dict)
    failed_calls: Dict[Tuple[str, str], str] = field(default_factory=dict)
    succeeded_tool_names: Dict[str, int] = field(default_factory=dict)

    last_successful_call_key: Optional[Tuple[str, str]] = None

    total_attempts: int = 0

    # 最近一次成功读取的文件内容，供 _fill_params 使用
    last_read_content: Optional[str] = None
    last_read_path: Optional[str] = None

    # 上一步的 thinking 摘要（短上下文下保留"计划"）
    last_thinking: Optional[str] = None

    # ── 通用证据（供 audit 用，不按 task_type） ──
    read_paths: set = field(default_factory=set)
    written_paths: set = field(default_factory=set)
    verification_evidence: Optional[dict] = None

    # 模型声明的最近一次搜索覆盖计划
    # {"source_tiers": [...], "breadth": "...", "queries": [...]}
    last_search_plan: Optional[dict] = None

    # 最近一次 web 工具返回按来源类型分桶的结果
    # {"official": [...], "news": [...], "forum": [...], ...}
    web_source_buckets: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # 最近 discovery 注册的工具名（供 _decide 补进候选，最多 10 个）
    recent_discovered_tools: set = field(default_factory=set)

    # 连续失败计数（key = (tool_name, params_key) → count）
    # 运行时状态，不进 snapshot
    _consecutive_failures: dict = field(default_factory=dict)

    # 连续收到 verification hint 但未响应的次数
    _verification_hint_count: int = 0

    # 按时间顺序记录成功调用的工具名
    successful_tool_sequence: List[str] = field(default_factory=list)

    failed_capabilities: set = field(default_factory=set)
    discovery_count: int = 0

    covered_documents: set = field(default_factory=set)
    searched_queries: set = field(default_factory=set)
    workspace_roots: Optional[List[str]] = None

    task_type: TaskType = TaskType.INFO_QUERY

    stop_reason: Optional[str] = None
    stop_detail: str = ""

    # 是否已扩展召回（防止 __need_more_tools__ 无限重试）
    _expanded_recall: bool = False
    
    def add_history(self, entry: str) -> None:
        self.history.append(entry)

    def mark_attempt(self) -> None:
        self.total_attempts += 1

    def mark_success(
        self,
        tool_name: str,
        params: Any,
        step_index: int,
    ) -> None:
        key = (tool_name, params_key(params))
        self.successful_calls[key] = step_index
        if tool_name not in self.succeeded_tool_names:
            self.succeeded_tool_names[tool_name] = step_index
        self.last_successful_call_key = key
        self.successful_tool_sequence.append(tool_name)

    def mark_failure(self, tool_name: str, params: Any, error: str) -> None:
        key = (tool_name, params_key(params))
        if key not in self.failed_calls:
            self.failed_calls[key] = (error or "")[:500]

    def has_succeeded(self, tool_name: str, params: Any) -> bool:
        return (tool_name, params_key(params)) in self.successful_calls

    def has_failed(self, tool_name: str, params: Any) -> bool:
        return (tool_name, params_key(params)) in self.failed_calls

    def last_error_for(self, tool_name: str, params: Any) -> Optional[str]:
        return self.failed_calls.get((tool_name, params_key(params)))

    def describe_recent_attempts(self, max_items: int = 5) -> str:
        """给 LLM 展示的"最近已尝试"摘要。

        快照恢复后 successful_calls 的 value 可能是 str（JSON 往返丢失类型），
        排序前统一转 int，转不了的跳过，避免 TypeError 把整个任务打挂。
        """
        items = []

        def _as_int(v: Any) -> Optional[int]:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        # ── 成功调用：按 step 编号排序 ──
        success_pairs = []
        for key, step in self.successful_calls.items():
            step_int = _as_int(step)
            if step_int is None:
                continue
            success_pairs.append((key, step_int))
        success_pairs.sort(key=lambda kv: kv[1])
        for key, step in success_pairs[-max_items:]:
            items.append(f"  ✅ Step {step}: {key[0]}({key[1][:60]})")

        # ── 失败调用：保持原顺序（failed_calls 无 step 编号） ──
        for key, err in list(self.failed_calls.items())[-max_items:]:
            items.append(f"  ❌ {key[0]}({key[1][:60]}) → {err[:80]}")

        return "\n".join(items) if items else "  (无)"

    # ── 快照 / 恢复（用于审批后的 orchestrator 续跑） ──

    def snapshot(self) -> dict:
        return {
            "goal": self.goal,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "history": list(self.history),
            "successful_calls": dict(self.successful_calls),
            "failed_calls": dict(self.failed_calls),
            "succeeded_tool_names": dict(self.succeeded_tool_names),
            "last_successful_call_key": self.last_successful_call_key,
            "total_attempts": self.total_attempts,
            "last_read_content": self.last_read_content,
            "last_read_path": self.last_read_path,
            "last_thinking": self.last_thinking,
            "read_paths": list(self.read_paths),
            "written_paths": list(self.written_paths),
            "verification_evidence": self.verification_evidence,
            "last_search_plan": self.last_search_plan,
            "web_source_buckets": {
                k: list(v) for k, v in self.web_source_buckets.items()
            },
            "recent_discovered_tools": list(self.recent_discovered_tools),
            "successful_tool_sequence": list(self.successful_tool_sequence),
            "covered_documents": list(self.covered_documents),
            "searched_queries": list(self.searched_queries),
            "workspace_roots": list(self.workspace_roots) if self.workspace_roots else None,
            "task_type": self.task_type.value
            if isinstance(self.task_type, TaskType)
            else str(self.task_type),
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "OrchestratorState":
        st = cls(
            goal=snap.get("goal", ""),
            session_id=snap.get("session_id"),
            task_id=snap.get("task_id"),
        )
        st.history = list(snap.get("history", []))
        # ── 重建 successful_calls 的 key ──
        # 正常路径：key 是 tuple(tool_name, params_json)，value 是 int。
        # 但快照经过 JSON 往返后可能出现三种形态：
        #   1. key 是 list（JSON 数组）→ 转 tuple
        #   2. key 是 "(tool_name, params_json)" 字符串 → ast.literal_eval 转 tuple
        #   3. key 是其他 str（无法解析）→ 原样保留，下游会跳过
        def _parse_call_key(k: Any) -> Any:
            if isinstance(k, list) and len(k) == 2:
                return (k[0], k[1])
            if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                try:
                    import ast
                    parsed = ast.literal_eval(k)
                    if isinstance(parsed, tuple) and len(parsed) == 2:
                        return parsed
                except (ValueError, SyntaxError):
                    pass
            return k

        raw_successful = snap.get("successful_calls", {}) or {}
        st.successful_calls = {}
        for k, v in raw_successful.items():
            parsed_key = _parse_call_key(k)
            # value 统一转 int；转不了保留原值（下游 describe_recent_attempts
            # 的 _as_int 会兜底跳过，不会崩）
            try:
                parsed_value: Any = int(v)
            except (TypeError, ValueError):
                parsed_value = v
            st.successful_calls[parsed_key] = parsed_value
        st.failed_calls = dict(snap.get("failed_calls", {}))
        st.succeeded_tool_names = dict(snap.get("succeeded_tool_names", {}))
        st.last_successful_call_key = snap.get("last_successful_call_key")
        st.total_attempts = snap.get("total_attempts", 0)
        st.last_read_content = snap.get("last_read_content")
        st.last_read_path = snap.get("last_read_path")
        st.last_thinking = snap.get("last_thinking")
        st.read_paths = set(snap.get("read_paths", []))
        st.written_paths = set(snap.get("written_paths", []))
        st.verification_evidence = snap.get("verification_evidence")
        st.last_search_plan = snap.get("last_search_plan")
        st.web_source_buckets = {
            k: list(v)
            for k, v in (snap.get("web_source_buckets") or {}).items()
        }
        st.recent_discovered_tools = set(snap.get("recent_discovered_tools", []))
        st.successful_tool_sequence = list(
            snap.get("successful_tool_sequence", [])
        )
        st.covered_documents = set(snap.get("covered_documents", []))
        st.searched_queries = set(snap.get("searched_queries", []))
        st.workspace_roots = snap.get("workspace_roots")
        raw_tt = snap.get("task_type", "info_query")
        try:
            st.task_type = TaskType(raw_tt)
        except ValueError:
            st.task_type = TaskType.INFO_QUERY
        st.stop_reason = snap.get("stop_reason")
        st.stop_detail = snap.get("stop_detail", "")
        return st
