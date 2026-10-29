"""信任分级。"""

from __future__ import annotations

from typing import Any


class TrustLevel:
    BUILTIN = 0       # 内置工具：完全信任
    STATIC_MCP = 1    # mcp_servers.json 静态配置
    DYNAMIC_MCP = 2   # search_and_connect_mcp 动态发现，需审批
    RUNTIME = 3       # 运行时注册，拒绝


def assign_trust_level(
    tool_name: str,
    origin: str = "",
    manager: Any = None,
) -> int:
    """按工具名 + origin + manager 推断信任级别。"""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        server = parts[1] if len(parts) == 3 else ""
        static = getattr(manager, "_static_configs", set()) if manager else set()
        if server and server in static:
            return TrustLevel.STATIC_MCP
        return TrustLevel.DYNAMIC_MCP
    if origin in ("builtin", "skill"):
        return TrustLevel.BUILTIN
    return TrustLevel.RUNTIME