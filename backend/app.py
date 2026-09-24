"""LangChain 后端服务模块 —— 应用装配层。

集成 LLM 调用封装、RAG 知识库、Agent 工具调用、Chains 工作流。
支持 ModelScope API（兼容 OpenAI 格式）和本地模型加载。

架构：
    - 本文件只负责「应用装配」：惰性加载注册、lifespan、中间件、全局单例、路由注册
    - 具体路由实现在 backend/routes/ 下，通过依赖注入接收 provider
    - 服务工厂保留在本文件（涉及全局单例和惰性加载）
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    ALLOWED_CORS_ORIGINS,
    DEFAULT_MODEL,
    LOCAL_MODEL_CONFIG,
    MODEL_SOURCE,
    MODESCOPE_API_KEY,
    MODESCOPE_BASE_URL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)
from .services.session_memory import get_session_memory
from .services.observability import init_langfuse
from .services.sandbox_backend import close_backend as close_sandbox_backend
from .services.memory_layer import init_mem0
from .services.lazy_loader import (
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
register_lazy_module(
    name="langchain_text_splitters",
    loader=lambda: __import__("langchain_text_splitters"),
    description="RAG 文本分块",
)
register_lazy_module(
    name="langchain_huggingface",
    loader=lambda: __import__("langchain_huggingface"),
    description="HuggingFace 嵌入模型封装",
)
register_lazy_module(
    name="langchain_community",
    loader=lambda: __import__("langchain_community"),
    description="LangChain 社区组件（legacy RAG）",
)


# ── 业务模块 import ──

import tools.loader
from tools.tools_def import knowledge as knowledge_tools
from tools.agent import ToolAgent, IntentAnalysis, ToolCallRequest, ToolCallResult

from rag import DocumentParserFactory, ResourceManager, TextChunker, FAISSVectorStore
from mcp_client import MCPClientManager, load_mcp_config
from mcp_client.worker import MCPWorker, MCPManagerProxy
from backend import AgentTaskService

from .services.local_model import (
    get_local_model_service,
    download_model_from_modelscope,
    ModelSource,
)
from .services.llm import (
    ModelScopeLLM as LLMService,
    MultimodalClient as MultimodalService,
    RAGService as LegacyRAGService,
)
from .services.local_rag import LocalRAGService as PersistentRAGService


# ── 路由工厂 import ──

from .routes.agent import create_agent_router
from .routes.mcp import create_mcp_router
from .routes.tools import create_tools_router
from .routes.chat import create_chat_router
from .routes.rag import create_rag_router
from .routes.sessions import create_sessions_router
from .routes.models import create_models_router
from .routes.health import create_health_router


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
rag_service = None


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

    # ── Mem0 初始化（未启用时静默 no-op） ──
    try:
        init_mem0()
    except Exception:
        logger.exception("Mem0 初始化失败")

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

    try:
        yield
    finally:
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
        local_rag_service = PersistentRAGService()
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


def get_rag_service():
    """延迟获取 legacy RAG 服务，避免启动时下载模型。"""
    global rag_service
    if rag_service is None:
        rag_service = LegacyRAGService()
    return rag_service


# ─────────────────────────────────────────────────────────────
# 模型切换辅助（models 路由用）
# ─────────────────────────────────────────────────────────────

def _model_info() -> dict:
    return {
        "model_source": MODEL_SOURCE.value,
        "model_name": DEFAULT_MODEL,
        "display_name": (
            DEFAULT_MODEL.split("/")[-1] if "/" in DEFAULT_MODEL else DEFAULT_MODEL
        ),
        "use_local": MODEL_SOURCE == ModelSource.LOCAL,
    }


def _switch_model_source(source: str) -> dict:
    global MODEL_SOURCE
    if source == "online":
        MODEL_SOURCE = ModelSource.ONLINE
        return {"status": "success", "message": "已切换到在线API模式"}
    if source == "local":
        MODEL_SOURCE = ModelSource.LOCAL
        service = get_local_model_service(LOCAL_MODEL_CONFIG)
        if service and service.is_loaded:
            return {
                "status": "success",
                "message": "已切换到本地模型模式，模型加载成功",
            }
        return {
            "status": "warning",
            "message": "已切换到本地模型模式，但模型加载失败，请检查配置",
        }
    raise HTTPException(status_code=400, detail="无效的来源类型，可选: online, local")


# ─────────────────────────────────────────────────────────────
# knowledge_tools RAG provider
# ─────────────────────────────────────────────────────────────

knowledge_tools.set_rag_provider(get_local_rag_service)


# ─────────────────────────────────────────────────────────────
# 注册路由
# ─────────────────────────────────────────────────────────────

app.include_router(create_health_router(get_all_module_status))

app.include_router(create_tools_router(get_tool_agent))

app.include_router(
    create_mcp_router(
        mcp_manager_provider=lambda: mcp_client_manager,
        agent_provider=get_tool_agent,
    )
)

app.include_router(
    create_chat_router(
        llm_provider=get_llm,
        multimodal_provider=get_multimodal_client,
        local_rag_provider=get_local_rag_service,
    )
)

app.include_router(
    create_rag_router(
        legacy_rag_provider=get_rag_service,
        local_rag_provider=get_local_rag_service,
        llm_provider=get_llm,
    )
)

app.include_router(create_sessions_router(get_session_memory))

app.include_router(
    create_models_router(
        model_info_provider=_model_info,
        model_downloader=download_model_from_modelscope,
        model_switcher=_switch_model_source,
    )
)

app.include_router(create_agent_router(get_agent_task_service, get_session_memory))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)