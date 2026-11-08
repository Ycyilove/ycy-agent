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
from tools.mcp_intent_index import load_or_build_index

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
from backend.routes.skills import create_skills_router
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

import psutil

def _rss():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

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


# ─────────────────────────────────────────────────────────────
# lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    """启动装配：MCP 建连 + boot_all；关闭：shutdown_all。

    具体启动步骤在 bootstrap/startup.py::boot_all，
    端口注入在 bootstrap/ports.py::wire_ports。
    """
    global mcp_client_manager, mcp_worker

    # ── 1. 建 MCP（必须最早，boot_all 依赖） ──
    try:
        configs = load_mcp_config()
    except Exception:
        logger.exception("MCP 配置加载失败")
        configs = []

    real_manager = MCPClientManager(configs)
    worker = MCPWorker(real_manager)
    await worker.start()
    proxy = MCPManagerProxy(worker, real_manager)

    agent = get_tool_agent()
    agent.set_mcp_client_manager(proxy)

    if configs:
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
        try:
            await proxy.start(
                agent.register_dynamic_tool,
                agent.unregister_dynamic_tool,
            )
        except Exception:
            logger.exception("MCP 客户端初始化失败")

    mcp_worker = worker
    mcp_client_manager = proxy

    # ── 2. 其余全部交给 boot_all ──
    from bootstrap.startup import boot_all, shutdown_all
    await boot_all(
        get_tool_agent=get_tool_agent,
        get_llm=get_llm,
        get_local_rag_service=get_local_rag_service,
        get_tool_shortlister=get_tool_shortlister,
        mcp_client_manager=proxy,
    )

    try:
        yield
    finally:
        await shutdown_all(mcp_worker=worker)
        mcp_worker = None
        mcp_client_manager = None

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
                top_k=8,
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

app.include_router(create_skills_router(get_tool_agent))

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
