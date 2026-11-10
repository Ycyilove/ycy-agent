"""SecurityGateway：所有工具执行前的统一安全网关。

替代当前分散在 orchestrator / routes / tool_def 里的 6 处安全检查。
"""

from __future__ import annotations

import os
import re
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SecurityContext:
    allowed_danger_levels: Optional[Set[str]] = None
    allowed_tool_names: Optional[Set[str]] = None
    workspace_roots: Optional[List[str]] = None
    workspace_trusted: bool = False
    session_id: str = ""
    approved_approval_ids: Set[str] = field(default_factory=set)
    trust_levels: Dict[str, int] = field(default_factory=dict)


_security_ctx_var: ContextVar[Optional[SecurityContext]] = ContextVar(
    "security_context", default=None
)


def set_security_context(ctx: Optional[SecurityContext]):
    return _security_ctx_var.set(ctx)


def reset_security_context(token):
    _security_ctx_var.reset(token)


def get_security_context() -> Optional[SecurityContext]:
    return _security_ctx_var.get()


@dataclass
class SecurityDecision:
    allowed: bool
    reason: str = ""
    needs_approval: bool = False


class SecurityGateway:
    """统一安全网关。"""

    _WORKSPACE_KEYS = frozenset({
        "path", "file_path", "directory", "dir", "source", "destination",
        "src", "dst", "target", "target_path", "output_path", "file",
        "old_path", "new_path",
    })
    _URL_KEYS = frozenset({"url", "uri", "href", "endpoint", "target_url"})
    _WRITE_HINTS = (
        "write", "create", "delete", "remove", "update", "modify",
        "move", "rename", "mkdir", "rmdir", "append", "truncate",
        "upload", "edit", "insert", "drop", "exec", "run",
    )

    def __init__(self, ssrf_checker=None):
        if ssrf_checker is None:
            try:
                from tools.tools_def.fetch import _check_url_safety
                ssrf_checker = _check_url_safety
            except Exception:
                logger.exception("[security] 无法加载 SSRF checker")
                ssrf_checker = None
        self._ssrf_check = ssrf_checker

    @classmethod
    def is_write_operation(cls, tool_name: str, meta: Any = None) -> bool:
        if meta is not None:
            dl = getattr(meta, "danger_level", "")
            if dl in ("medium", "high"):
                return True
        lower = (tool_name or "").lower()
        return any(h in lower for h in cls._WRITE_HINTS)

    def check(
        self,
        tool_name: str,
        params: Dict[str, Any],
        meta: Any,
        ctx: Optional[SecurityContext],
    ) -> SecurityDecision:
        if ctx is None:
            return SecurityDecision(True)   # 未设上下文 → 放行（兼容旧路径）

        # 1. 白名单
        if ctx.allowed_tool_names is not None and tool_name not in ctx.allowed_tool_names:
            return SecurityDecision(False, f"工具 {tool_name} 不在允许列表")

        # 2. danger_level
        dl = getattr(meta, "danger_level", "safe") if meta else "safe"
        if ctx.allowed_danger_levels is not None and dl not in ctx.allowed_danger_levels:
            return SecurityDecision(
                False,
                f"工具 danger_level={dl} 不在允许范围 {sorted(ctx.allowed_danger_levels)}",
            )

        # 3. trust_level
        trust = getattr(meta, "trust_level", 0) if meta else 0
        if trust >= 3:
            return SecurityDecision(False, f"工具 {tool_name} 信任级别 L3（拒绝）")
        if trust == 2:
            approval_id = params.get("__approval_id__")
            if not approval_id or approval_id not in ctx.approved_approval_ids:
                return SecurityDecision(
                    False, f"工具 {tool_name} 需审批（L2 动态发现）",
                    needs_approval=True,
                )

        # 4. workspace（所有写操作）
        is_write = self.is_write_operation(tool_name, meta)
        if is_write:
            if not ctx.workspace_roots:
                # workspace 为空 → 无法校验，写操作一律需审批
                approval_id = params.get("__approval_id__")
                if not approval_id or approval_id not in ctx.approved_approval_ids:
                    return SecurityDecision(
                        False,
                        "当前会话未授权任何工作区目录，写操作需审批",
                        needs_approval=True,
                    )
            else:
                for key, val in params.items():
                    if key not in self._WORKSPACE_KEYS or not isinstance(val, str) or not val:
                        continue
                    abs_path = os.path.abspath(val)
                    ok = any(
                        abs_path == os.path.abspath(r)
                        or abs_path.startswith(os.path.abspath(r) + os.sep)
                        for r in ctx.workspace_roots
                    )
                    if not ok:
                        return SecurityDecision(
                            False,
                            f"路径 {val} 不在工作区 {ctx.workspace_roots}",
                            needs_approval=True,
                        )

        # 5. SSRF
        if self._ssrf_check:
            for key, val in params.items():
                if key in self._URL_KEYS and isinstance(val, str):
                    reason = self._ssrf_check(val)
                    if reason:
                        return SecurityDecision(False, f"URL 安全检查失败: {reason}")

        # 6. 工作区未授权时，写操作强制审批
        if is_write and not ctx.workspace_trusted:
            approval_id = params.get("__approval_id__")
            if not approval_id or approval_id not in ctx.approved_approval_ids:
                return SecurityDecision(
                    False, f"写操作 {tool_name} 需审批", needs_approval=True
                )

        return SecurityDecision(True)


_gateway: Optional[SecurityGateway] = None


def get_gateway() -> SecurityGateway:
    global _gateway
    if _gateway is None:
        _gateway = SecurityGateway()
    return _gateway