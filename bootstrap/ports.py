"""把 tools/rag/mcp_client 的实现注入 core.ports。只被 startup 调用。

设计：
    - 每个实现用薄适配器包一层，避免 core 直接看到具体类
    - 适配器只做方法转发，不加逻辑
    - 注入失败只记日志，不阻断启动（core 侧 get_* 返回 None 时降级）
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from core import ports

logger = logging.getLogger(__name__)


class _RAGAdapter:
    """把 LocalRAGService 适配成 core.ports.RAGPort。"""

    def __init__(self, rag_provider: Callable[[], Any]):
        self._rag_provider = rag_provider

    def resolve_asset_tags(self, text, resources):
        return self._rag_provider().resource_manager.resolve_asset_tags(
            text, resources
        )

    def search(self, query, top_k, **kwargs):
        return self._rag_provider().search(query, top_k, **kwargs)

    def get_stats(self):
        return self._rag_provider().get_stats()


class _ToolAdapter:
    """把 ToolAgent 适配成 core.ports.ToolCatalogPort。"""

    def __init__(self, agent_provider: Callable[[], Any]):
        self._agent_provider = agent_provider

    def list_tools(self, allowed_danger_levels=None):
        return self._agent_provider().list_all_tools(
            allowed_danger_levels=allowed_danger_levels
        )

    def get_metadata(self, name):
        return self._agent_provider().registry.get_metadata(name)

    async def execute(self, tool_call):
        return await self._agent_provider().execute_tool_async(tool_call)

    def format_result(self, result):
        return self._agent_provider().format_tool_result(result)


class _ConfigAdapter:
    """把 backend.config 适配成 core.ports.ConfigPort。

    backend.config 是模块，不是对象，用 __getattr__ 转发。
    """

    def __getattr__(self, name):
        from backend import config as _cfg
        return getattr(_cfg, name)


def wire_ports(
    *,
    memory_layer: Any,
    search_adapter: Any,
    rag_service_provider: Callable[[], Any],
    approval_store: Any,
    tool_agent_provider: Callable[[], Any],
) -> None:
    """注入所有端口。启动时调一次。失败单项不影响其他项。"""
    def _safe(name, fn):
        try:
            fn()
            logger.info("[bootstrap] port wired: %s", name)
        except Exception:
            logger.exception("[bootstrap] port wire failed: %s", name)

    _safe("memory_reader", lambda: ports.set_memory_reader(memory_layer))
    _safe("memory_writer", lambda: ports.set_memory_writer(memory_layer))
    _safe("search_port", lambda: ports.set_search_port(search_adapter))
    _safe(
        "rag_port",
        lambda: ports.set_rag_port(_RAGAdapter(rag_service_provider)),
    )
    _safe("approval_port", lambda: ports.set_approval_port(approval_store))
    _safe(
        "tool_catalog",
        lambda: ports.set_tool_catalog(_ToolAdapter(tool_agent_provider)),
    )
    _safe("config", lambda: ports.set_config(_ConfigAdapter()))