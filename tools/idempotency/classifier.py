"""Classify tools by idempotency level.

- "pure"        纯幂等（读/查询类）—— 任意重复无害
- "conditional" 条件幂等（预览/搜索类）—— 相邻相同参数可跳过
- "none"        非幂等（注册/雇佣/支付类）—— 成功后相同参数禁止重复
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


_NON_IDEMPOTENT_TOOL_KEYWORDS = (
    "register", "onboard", "sign_up", "signup", "subscribe",
    "hire", "buy", "fund", "transfer", "pay", "withdraw",
    "deposit", "swap", "mint", "burn", "award", "assign",
    "create", "insert", "update", "delete", "remove", "put",
    "post", "send", "deploy", "publish", "submit",
    "execute", "start", "attach", "open_session", "session_open",
    "invite", "notify",
)

_PURE_IDEMPOTENT_TOOL_KEYWORDS = (
    "get", "list", "read", "calculate", "compute", "count",
    "check", "describe", "info", "status", "view", "show",
)

_CONDITIONAL_IDEMPOTENT_TOOL_KEYWORDS = (
    "preview", "search", "query", "find", "lookup", "look_up",
    "scan", "poll", "watch",
)


def classify_tool_idempotency(tool_name: str, metadata: Any = None) -> str:
    n = (tool_name or "").lower()

    if metadata is not None:
        idem = getattr(metadata, "idempotent", None)
        if idem is False:
            return "none"
        if idem is True:
            if any(kw in n for kw in _NON_IDEMPOTENT_TOOL_KEYWORDS):
                return "none"
            if any(kw in n for kw in _CONDITIONAL_IDEMPOTENT_TOOL_KEYWORDS):
                return "conditional"
            return "pure"

    if any(kw in n for kw in _NON_IDEMPOTENT_TOOL_KEYWORDS):
        return "none"
    if any(kw in n for kw in _CONDITIONAL_IDEMPOTENT_TOOL_KEYWORDS):
        return "conditional"
    if any(kw in n for kw in _PURE_IDEMPOTENT_TOOL_KEYWORDS):
        return "pure"

    return "none"