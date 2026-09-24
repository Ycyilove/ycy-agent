"""Orchestrator session state（策略 E）."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    session_id: Optional[str] = None   # ← Mem0 结构化记忆的隔离键
    history: List[str] = field(default_factory=list)

    successful_calls: Dict[Tuple[str, str], int] = field(default_factory=dict)
    failed_calls: Dict[Tuple[str, str], str] = field(default_factory=dict)
    succeeded_tool_names: Dict[str, int] = field(default_factory=dict)

    last_successful_call_key: Optional[Tuple[str, str]] = None

    mandatory_next_tool: Optional[str] = None

    total_attempts: int = 0

    # 最近一次成功读取的文件内容，供 _fill_params 使用
    last_read_content: Optional[str] = None
    last_read_path: Optional[str] = None

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
        """给 LLM 展示的"最近已尝试"摘要。"""
        items = []
        for key, step in sorted(
            self.successful_calls.items(), key=lambda kv: kv[1]
        )[-max_items:]:
            items.append(f"  ✅ Step {step}: {key[0]}({key[1][:60]})")
        for key, err in list(self.failed_calls.items())[-max_items:]:
            items.append(f"  ❌ {key[0]}({key[1][:60]}) → {err[:80]}")
        return "\n".join(items) if items else "  (无)"

    # ── 快照 / 恢复（用于审批后的 orchestrator 续跑） ──

    def snapshot(self) -> dict:
        return {
            "goal": self.goal,
            "session_id": self.session_id,
            "history": list(self.history),
            "successful_calls": dict(self.successful_calls),
            "failed_calls": dict(self.failed_calls),
            "succeeded_tool_names": dict(self.succeeded_tool_names),
            "last_successful_call_key": self.last_successful_call_key,
            "total_attempts": self.total_attempts,
            "last_read_content": self.last_read_content,
            "last_read_path": self.last_read_path,
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "OrchestratorState":
        st = cls(
            goal=snap.get("goal", ""),
            session_id=snap.get("session_id"),
        )
        st.history = list(snap.get("history", []))
        st.successful_calls = dict(snap.get("successful_calls", {}))
        st.failed_calls = dict(snap.get("failed_calls", {}))
        st.succeeded_tool_names = dict(snap.get("succeeded_tool_names", {}))
        st.last_successful_call_key = snap.get("last_successful_call_key")
        st.total_attempts = snap.get("total_attempts", 0)
        st.last_read_content = snap.get("last_read_content")
        st.last_read_path = snap.get("last_read_path")
        return st