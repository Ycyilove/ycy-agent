"""启动装配。从 backend/app.py 的 app_lifespan 搬过来。

boot_all():
    1. Langfuse
    2. RAG embedding warmup -> mem0
    3. 资源清理
    4. MCP discovery context
    5. MCP 意图索引后台构建
    6. 惰性模块后台预加载
    7. ToolShortlister warmup
    8. wire_ports（B2-b 新增）
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_state: dict = {}


async def boot_all(
    *,
    get_tool_agent: Callable[[], Any],
    get_llm: Callable[[], Any],
    get_local_rag_service: Callable[[], Any],
    get_tool_shortlister: Callable[[], Any],
    mcp_client_manager: Any,
) -> None:
    """按依赖顺序启动。失败不阻断，只记日志。"""
    from infra.observability import init_langfuse
    from core.memory.mem0_layer import init_mem0

    # 1. Langfuse
    try:
        init_langfuse()
    except Exception:
        logger.exception("Langfuse 初始化失败")

    # 2. RAG embedding warmup -> mem0（链式，串行）
    async def _warmup_then_mem0():
        try:
            rag = get_local_rag_service()
            vs = rag.vector_store
            if vs is not None:
                await asyncio.to_thread(
                    lambda: vs._generate_embeddings(["warmup"])
                )
                logger.info("[boot] RAG embedding warmed up")
        except Exception:
            logger.exception("[boot] RAG warmup failed")
        if os.getenv("MEM0_ENABLED", "1") != "0":
            try:
                await asyncio.to_thread(init_mem0)
                logger.info("[boot] mem0 initialized")
            except Exception:
                logger.exception("[boot] mem0 init failed")

    asyncio.create_task(_warmup_then_mem0())

    # 3. 资源清理
    try:
        from core.resource.store import cleanup_expired
        await asyncio.to_thread(cleanup_expired)
    except Exception:
        logger.exception("清理 agent 资源失败")

    # 4. MCP discovery context 注入
    try:
        from tools import mcp_discovery
        agent = get_tool_agent()
        mcp_discovery.set_mcp_discovery_context(
            manager_provider=lambda: mcp_client_manager,
            register_tool=agent.register_dynamic_tool,
            unregister_tool=agent.unregister_dynamic_tool,
            llm_provider=get_llm,
        )
    except Exception:
        logger.exception("MCP discovery context 注入失败")

    # 5. MCP 意图索引后台构建
    async def _build_mcp_index():
        try:
            from tools.mcp_intent_index import load_or_build_index
            await asyncio.to_thread(load_or_build_index, mcp_client_manager)
            logger.info("[boot] MCP semantic index ready")
        except Exception:
            logger.exception("[boot] MCP index build failed")

    asyncio.create_task(_build_mcp_index())

    # 6. 惰性模块后台预加载
    try:
        from infra.lazy_loader import preload_all_in_background
        preload_all_in_background()
        logger.info("[boot] background preload scheduled")
    except Exception:
        logger.exception("[boot] preload schedule failed")

    # 7. ToolShortlister warmup
    async def _warmup_shortlister():
        try:
            sl = get_tool_shortlister()
            if sl is None:
                return
            await asyncio.to_thread(sl.warmup)
            agent = get_tool_agent()
            await asyncio.to_thread(sl._ensure_index, agent.list_all_tools())
            logger.info("[boot] ToolShortlister warmed up")
        except Exception:
            logger.exception("[boot] shortlister warmup failed")

    asyncio.create_task(_warmup_shortlister())

    # 8. wire_ports（B2-b 新增）
    try:
        from bootstrap.ports import wire_ports
        from core.memory import mem0_layer
        from tools.mcp_discovery import make_search_port
        from tools.security.approval import get_approval_store

        wire_ports(
            memory_layer=mem0_layer,
            search_adapter=make_search_port(),
            rag_service_provider=get_local_rag_service,
            approval_store=get_approval_store(),
            tool_agent_provider=get_tool_agent,
        )
        logger.info("[boot] ports wired")
    except Exception:
        logger.exception("[boot] wire_ports failed")

    _state["booted"] = True
    logger.info("[boot] boot_all complete")


async def shutdown_all(*, mcp_worker: Any) -> None:
    """关闭顺序与启动相反。"""
    try:
        if mcp_worker is not None:
            await mcp_worker.stop()
    except Exception:
        logger.exception("关闭 MCP worker 失败")

    try:
        from infra.sandbox import close_backend as close_sandbox_backend
        close_sandbox_backend()
    except Exception:
        logger.exception("关闭 E2B 沙箱失败")