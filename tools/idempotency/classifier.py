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
    # 新增：写文件相关
    "write", "edit", "modify", "move", "rename",
    "mkdir", "append", "overwrite",
)

_PURE_IDEMPOTENT_TOOL_KEYWORDS = (
    "get", "list", "read", "calculate", "compute", "count",
    "check", "describe", "info", "status", "view", "show",
    # 新增：查询类
    "stats", "statistics", "search", "find", "query", "fetch",
)

_CONDITIONAL_IDEMPOTENT_TOOL_KEYWORDS = (
    "preview", "scan", "poll", "watch",
)


def classify_tool_idempotency(tool_name: str, metadata: Any = None) -> str:
    n = (tool_name or "").lower()

    # ── metadata.idempotent=True → 显式声明为幂等 ──
    if metadata is not None:
        idem = getattr(metadata, "idempotent", None)
        if idem is True:
            # 再按关键词细分 pure / conditional / none
            if any(kw in n for kw in _NON_IDEMPOTENT_TOOL_KEYWORDS):
                return "none"
            if any(kw in n for kw in _CONDITIONAL_IDEMPOTENT_TOOL_KEYWORDS):
                return "conditional"
            return "pure"
        # idem is False 或 None → 走关键词判断（不改）

    # ── 关键词判断（non > conditional > pure > 默认 none） ──
    if any(kw in n for kw in _NON_IDEMPOTENT_TOOL_KEYWORDS):
        return "none"
    if any(kw in n for kw in _CONDITIONAL_IDEMPOTENT_TOOL_KEYWORDS):
        return "conditional"
    if any(kw in n for kw in _PURE_IDEMPOTENT_TOOL_KEYWORDS):
        return "pure"

    return "none"