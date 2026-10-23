"""LangChain 后端服务模块 —— 应用装配层。

集成 LLM 调用封装、RAG 知识库、Agent 工具调用、Chains 工作流。
支持 ModelScope API（兼容 OpenAI 格式）和本地模型加载。

架构：
    - 本文件只负责「应用装配」：惰性加载注册、lifespan、中间件、全局单例、路由注册
    - 具体路由实现在 backend/routes/ 下，通过依赖注入接收 provider
    - 服务工厂保留在本文件（涉及全局单例和惰性加载）
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional, Set

from infra.offline import configure_hf_env
configure_hf_env()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    ALLOWED_CORS_ORIGINS,
    DEFAULT_MODEL,
    LOCAL_MODEL_CONFIG,
    MODEL_SOURCE,
    MODESCOPE_API_KEY,
    MODESCOPE_BASE_URL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)
from core.memory.session import get_session_memory
from infra.observability import init_langfuse
from infra.sandbox import close_backend as close_sandbox_backend
from core.memory.mem0_layer import init_mem0
from infra.lazy_loader import (
    register_lazy_module,
    ensure_module_loaded,
    ensure_module_loaded_sync,
    preload_all_in_background,
    get_all_module_status,
)


# ── 注册重模块为惰性加载 ──

register_lazy_module(
    name="anthropic",
    loader=lambda: __import__("anthropic"),
    description="多模态分析所需的 Anthropic 客户端",
)
register_lazy_module(
    name="langchain_openai",
    loader=lambda: __import__("langchain_openai"),
    description="LLM 对话所需的 LangChain OpenAI 封装",
)
register_lazy_module(
    name="langchain_core",
    loader=lambda: __import__("langchain_core"),
    description="LangChain Core（消息、回调等）",
)

# ── 业务模块 import ──

import tools.loader
from tools.tools_def import knowledge as knowledge_tools
from tools.agent import ToolAgent, IntentAnalysis, ToolCallRequest, ToolCallResult

from rag import DocumentParserFactory, ResourceManager, TextChunker, FAISSVectorStore
from mcp_client import MCPClientManager, load_mcp_config
from mcp_client.worker import MCPWorker, MCPManagerProxy
from core.task.service import AgentTaskService

from core.llm.local_model import (
    get_local_model_service,
    download_model_from_modelscope,
    ModelSource,
)
from core.llm.provider import (
    ModelScopeLLM as LLMService,
    MultimodalClient as MultimodalService,
)
from rag import LocalRAGService


# ── 路由工厂 import ──

from backend.routes.agent import create_agent_router
from backend.routes.mcp import create_mcp_router
from backend.routes.tools import create_tools_router
from backend.routes.rag import create_rag_router
from backend.routes.sessions import create_sessions_router
from backend.routes.health import create_health_router
from backend.routes.trash import create_trash_router
from backend.routes.resource import create_resource_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────────────────────────

tool_agent: Optional[ToolAgent] = None
mcp_client_manager: Optional[MCPManagerProxy] = None
mcp_worker: Optional[MCPWorker] = None
agent_task_service: Optional[AgentTaskService] = None
tool_shortlister = None
local_rag_service = None
llm_instance = None
multimodal_client = None
# ── capability index 重建锁 + task（串行化，避免并发写坏 records.json） ──
_capability_index_lock: Optional[asyncio.Lock] = None
_capability_index_task: Optional[asyncio.Task] = None

# ─────────────────────────────────────────────────────────────
# lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    """启动 MCP worker task，所有连接操作在 worker task 里执行。"""
    global mcp_client_manager, mcp_worker

    # ── Langfuse 初始化（未配置时静默 no-op） ──
    try:
        init_langfuse()
    except Exception:
        logger.exception("Langfuse 初始化失败")

    # ── Mem0 初始化（后台异步，不阻塞启动；可用 MEM0_ENABLED=0 关闭） ──
    # 关键：先让 RAG / 短列表的 SentenceTransformer 完成首次加载，
    # mem0 的 embedder 再初始化时命中进程内 HF cache，避免并发联网校验。
    async def _init_mem0_background():
        if os.getenv("MEM0_ENABLED", "1") == "0":
            logger.info("[boot] mem0 disabled by MEM0_ENABLED=0")
            return
        try:
            logger.info("[boot] initializing mem0 in background...")
            await asyncio.to_thread(init_mem0)
            logger.info("[boot] mem0 initialized")
        except Exception:
            logger.exception("Mem0 初始化失败")

    async def _warmup_embeddings_then_mem0():
        # 1) 先同步预热 RAG embedding（触发本地加载，写进程内 cache）
        try:
            rag = get_local_rag_service()
            vec_store = rag.vector_store  # LocalRAGService.__init__ 里已确认此属性名
            if vec_store is not None:
                await asyncio.to_thread(
                    lambda: vec_store._generate_embeddings(["warmup"])
                )
                logger.info("[boot] RAG embedding warmed up")
            else:
                logger.warning("[boot] RAG vector_store is None, skip warmup")
        except Exception:
            logger.exception("[boot] RAG embedding warmup failed (non-fatal)")
        # 2) 再起 mem0 后台线程
        await _init_mem0_background()

    asyncio.create_task(_warmup_embeddings_then_mem0())

    # ── 清理过期的 agent 资源 ──
    try:
        from core.resource.store import cleanup_expired
        await asyncio.to_thread(cleanup_expired)
    except Exception:
        logger.exception("清理 agent 资源失败")

    try:
        configs = load_mcp_config()
    except Exception as error:
        logger.error("MCP 配置加载失败: %s", error)
        configs = []

    # 真 manager（无 async 副作用，构造安全）
    real_manager = MCPClientManager(configs)

    # worker + proxy：所有 async 有状态操作都路由到 worker task
    worker = MCPWorker(real_manager)
    await worker.start()
    proxy = MCPManagerProxy(worker, real_manager)

    # 注入 ToolAgent（拿到的是 proxy）
    agent = get_tool_agent()
    agent.set_mcp_client_manager(proxy)

    # 启动连接（走 proxy → worker）
    if configs:
        # stdio 命令预检：命令不存在直接跳过，不浪费 3 次重试
        from backend.routes.mcp import _resolve_command
        filtered = []
        for c in configs:
            if c.transport == "stdio":
                if _resolve_command(c.command) is None:
                    logger.warning(
                        "[boot] skip MCP %s: command not found (%s)",
                        c.name, c.command,
                    )
                    continue
            filtered.append(c)
        # 用过滤后的 configs 重建 manager（或直接跳过失败项）
        # 简化：只对 filtered 里的 server 调 start
        try:
            await proxy.start(
                agent.register_dynamic_tool,
                agent.unregister_dynamic_tool,
            )
        except Exception:
            logger.exception("MCP 客户端初始化失败")

    mcp_worker = worker
    mcp_client_manager = proxy

    # 注入 discovery context（也用 proxy）
    try:
        from tools import mcp_discovery
        mcp_discovery.set_mcp_discovery_context(
            manager_provider=lambda: mcp_client_manager,
            register_tool=agent.register_dynamic_tool,
            unregister_tool=agent.unregister_dynamic_tool,
            llm_provider=get_llm,
        )
    except Exception:
        logger.exception("MCP discovery context 注入失败")

    # ── 启动后台预加载 ──
    try:
        preload_all_in_background()
        logger.info("[boot] background preload scheduled")
    except Exception:
        logger.exception("[boot] background preload failed to schedule")

    # ── 预热工具筛选器的 embedding 模型 ──
    async def _warmup_shortlister():
        try:
            sl = get_tool_shortlister()
            if sl is None:
                return
            # 在 executor 里跑（模型加载是同步阻塞）
            await asyncio.to_thread(sl.warmup)
            # 顺便索引一次当前工具集
            agent = get_tool_agent()
            tools_map = agent.list_all_tools()
            await asyncio.to_thread(sl._ensure_index, tools_map)
            logger.info("[boot] ToolShortlister warmed up")
        except Exception:
            logger.exception("[boot] ToolShortlister warmup failed")

    asyncio.create_task(_warmup_shortlister())

    mcp_worker = worker
    mcp_client_manager = proxy

    # ── 后台建意图索引（只覆盖已连接 server，不联网） ──
    schedule_capability_index_rebuild(reason="lifespan_start")
    
    try:
        yield
    finally:
        # 取消未完成的索引重建 task，避免退出时挂起
        if _capability_index_task is not None and not _capability_index_task.done():
            try:
                _capability_index_task.cancel()
                await asyncio.wait_for(_capability_index_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                logger.exception("取消 capability index task 失败")

        try:
            if worker is not None:
                await worker.stop()
        except Exception:
            logger.exception("关闭 MCP worker 失败")
        mcp_worker = None
        mcp_client_manager = None

        # ── 关闭 E2B 沙箱 ──
        try:
            close_sandbox_backend()
        except Exception:
            logger.exception("关闭 E2B 沙箱失败")

# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="LLM API 服务", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# 服务工厂
# ─────────────────────────────────────────────────────────────

def get_tool_agent() -> ToolAgent:
    """获取工具 Agent 实例（延迟初始化）。"""
    global tool_agent
    if tool_agent is None:
        tools.loader.load_all_tools()
        tool_agent = ToolAgent()
    return tool_agent


def get_tool_shortlister():
    """获取工具筛选器（延迟初始化）。失败返回 None。"""
    global tool_shortlister
    if tool_shortlister is None:
        try:
            from tools.orchestration.tool_shortlister import ToolShortlister
            tool_shortlister = ToolShortlister(
                top_k=15,
                min_score=0.10,
                min_keep=3,
            )
            logger.info("[boot] ToolShortlister created")
        except Exception:
            logger.exception("[boot] ToolShortlister 创建失败")
            tool_shortlister = None
    return tool_shortlister


def get_local_rag_service():
    """获取本地 RAG 服务实例。"""
    global local_rag_service
    if local_rag_service is None:
        local_rag_service = LocalRAGService()
    return local_rag_service


def get_agent_task_service() -> AgentTaskService:
    """获取工作台任务编排器（不提前加载模型依赖）。"""
    global agent_task_service
    if agent_task_service is None:
        from tools.orchestrator import AgentOrchestrator

        def _orchestrator_factory():
            return AgentOrchestrator(
                agent_provider=get_tool_agent,
                llm_provider=get_llm,
                max_steps=10,
                shortlister_provider=get_tool_shortlister,
            )

        agent_task_service = AgentTaskService(
            agent_provider=get_tool_agent,
            llm_provider=get_llm,
            multimodal_provider=get_multimodal_client,
            session_provider=get_session_memory,
            rag_provider=get_local_rag_service,
            orchestrator_provider=_orchestrator_factory,
        )
    return agent_task_service


def get_llm():
    """延迟获取 LLM 实例。"""
    global llm_instance
    if llm_instance is None:
        llm_instance = LLMService()
    return llm_instance


def get_multimodal_client():
    """延迟获取多模态客户端。"""
    global multimodal_client
    if multimodal_client is None:
        multimodal_client = MultimodalService()
    return multimodal_client

# ─────────────────────────────────────────────────────────────
# capability index 重建
# ─────────────────────────────────────────────────────────────

def _get_capability_index_lock() -> asyncio.Lock:
    global _capability_index_lock
    if _capability_index_lock is None:
        _capability_index_lock = asyncio.Lock()
    return _capability_index_lock


async def rebuild_capability_index(
    reason: str = "manual",
    servers: Optional[Set[str]] = None,
) -> None:
    """重建 MCP 意图索引。

    - servers=None：全量重建（首次启动 / 手动）
    - servers={...}：增量（只处理指定 server）
    """
    if mcp_client_manager is None:
        logger.debug(
            "[boot] rebuild_capability_index skipped (reason=%s): "
            "manager not ready",
            reason,
        )
        return

    lock = _get_capability_index_lock()
    async with lock:
        try:
            from scripts.build_capability_index import (
                build_full, build_incremental,
            )
            if servers:
                await build_incremental(
                    mcp_client_manager, get_llm(), servers,
                )
            else:
                await build_full(mcp_client_manager, get_llm())
            logger.info(
                "[boot] capability index rebuilt (reason=%s, servers=%r)",
                reason, servers,
            )
        except Exception:
            logger.exception(
                "[boot] capability index rebuild failed (reason=%s)", reason,
            )

def schedule_capability_index_rebuild(
    reason: str = "manual",
    servers: Optional[Set[str]] = None,
) -> None:
    """后台调度一次重建。servers=None 全量，否则增量。"""
    global _capability_index_task
    if mcp_client_manager is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug(
            "[boot] schedule_capability_index_rebuild: no running loop "
            "(reason=%s)",
            reason,
        )
        return

    prev_task = _capability_index_task

    async def _run(prev: Optional[asyncio.Task]):
        if prev is not None and not prev.done():
            try:
                await prev
            except Exception:
                pass
        await rebuild_capability_index(reason=reason, servers=servers)

    _capability_index_task = loop.create_task(
        _run(prev_task), name=f"capability-index-rebuild-{reason}",
    )
    
# ─────────────────────────────────────────────────────────────
# knowledge_tools RAG provider
# ─────────────────────────────────────────────────────────────

knowledge_tools.set_rag_provider(get_local_rag_service)


# ─────────────────────────────────────────────────────────────
# 注册路由
# ─────────────────────────────────────────────────────────────

app.include_router(create_health_router(get_all_module_status))

app.include_router(create_trash_router())

app.include_router(create_resource_router())

app.include_router(create_tools_router(get_tool_agent))

app.include_router(
    create_mcp_router(
        mcp_manager_provider=lambda: mcp_client_manager,
        agent_provider=get_tool_agent,
    )
)

app.include_router(
    create_rag_router(
        local_rag_provider=get_local_rag_service,
        llm_provider=get_llm,
    )
)

app.include_router(create_sessions_router(get_session_memory))

app.include_router(create_agent_router(get_agent_task_service, get_session_memory))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "localhost")
    port = int(os.getenv("APP_PORT", "8000"))

    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        reload=False,
        reload_dirs=[
            "backend", "core", "tools", "rag", "infra", "mcp_client", "src",
        ],
        reload_includes=["*.py"],          # 只关心 .py 变更
        reload_excludes=[
            "models/*", "data/*", "logs/*", "vector_store/*",
            "rag_resources/*", "output/*", "*.jsonl", "*.db",
            "*.json", "*.md", "*.txt", "*.pdf", "*.jpg", "*.png",
        ],
    )