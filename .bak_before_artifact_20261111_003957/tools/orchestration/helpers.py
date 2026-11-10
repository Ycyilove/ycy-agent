"""Orchestration 模块级辅助函数（B8 fix）。

从 core.py 提取 + 从 B8-a 丢失恢复：
    - _READ_TOOL_HINTS / _is_read_like
    - _norm_filename
    - _format_relative_time
    - _build_structured_history
    - _SucceededSeqView（class）
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


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
    return os.path.basename(s)


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


__all__ = [
    "_READ_TOOL_HINTS",
    "_is_read_like",
    "_norm_filename",
    "_format_relative_time",
    "_build_structured_history",
    "_SucceededSeqView",
]