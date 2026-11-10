"""两条入口的薄壳。委托给 DiscoveryPipeline。"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


async def maybe_discover(
    goal: str,
    tools: Dict[str, Any],
    manager: Any,
    register_tool: Callable,
    unregister_tool: Callable,
    allowed_danger_levels: Optional[set],
    state: Any = None,
) -> Dict[str, Any]:
    from .context import DiscoveryContext
    from .pipeline import make_local_pipeline

    ctx = DiscoveryContext(
        mode="local",
        manager=manager,
        register_tool=register_tool,
        unregister_tool=unregister_tool,
        allowed_danger_levels=allowed_danger_levels,
        user_query=goal or "",
    )
    try:
        return await make_local_pipeline().run(ctx, tools)
    except Exception:
        logger.exception("[maybe_discover] pipeline failed")
        return tools


async def search_and_connect_mcp(
    user_query: str = "",
    capability: str = "",
    limit: int = 100,
) -> Dict[str, Any]:
    from .context import DiscoveryContext
    from .pipeline import make_registry_pipeline

    from tools.mcp_discovery import (
        _get_manager, _register_tool, _unregister_tool,
        _get_tried_servers,
    )

    ctx = DiscoveryContext(
        mode="registry",
        manager=_get_manager(),
        register_tool=_register_tool,
        unregister_tool=_unregister_tool,
        user_query=user_query or "",
        capability=capability or "",
        limit=limit,
        tried=set(_get_tried_servers()),
    )
    await make_registry_pipeline().run(ctx, {})

    if ctx.result:
        return ctx.result

    # 兼容 RankedCandidate + dict
    def _name(c):
        if isinstance(c, dict):
            return c.get("registry_name") or c.get("server_name") or ""
        return getattr(c, "server_name", "") or ""

    return {
        "success": False,
        "error": (
            "\n".join(f"  - {e}" for e in ctx.errors[:5])
            if ctx.errors else "未能找到匹配的 MCP 服务器"
        ),
        "candidates": [_name(c) for c in ctx.candidates[:10] if _name(c)],
    }