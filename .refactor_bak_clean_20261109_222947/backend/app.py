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

# ── B9-b：7 个 factory 从 backend.deps 取 ──
from .deps import (
    get_tool_agent,
    get_tool_shortlister,
    get_local_rag_service,
    get_agent_task_service,
    get_llm,
    get_multimodal_client,
)
from . import deps
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


# ─────────────────────────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────
# lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    """启动装配：MCP 建连 + boot_all；关闭：shutdown_all。

    具体启动步骤在 bootstrap/startup.py::boot_all，
    端口注入在 bootstrap/ports.py::wire_ports。
    """
    from . import deps

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

    deps.mcp_worker = worker
    deps.mcp_client_manager = proxy

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
        deps.mcp_worker = None
        deps.mcp_client_manager = None

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


    
# ─────────────────────────────────────────────────────────────
# knowledge_tools RAG provider
# ─────────────────────────────────────────────────────────────

knowledge_tools.set_rag_provider(get_local_rag_service)


# ─────────────────────────────────────────────────────────────
# 注册路由
# ─────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════
# 注册路由（B9-c 按 domain 分组）
# ═══════════════════════════════════════════════════════════

# ── Health ──
app.include_router(create_health_router(get_all_module_status))

# ── Sessions（会话 + 回收站）──
app.include_router(create_sessions_router(get_session_memory))
app.include_router(create_trash_router())

# ── Workbench（任务 + 审批 + 资源下载）──
app.include_router(create_agent_router(get_agent_task_service, get_session_memory))
app.include_router(create_resource_router())

# ── Capabilities（工具 + 技能 + MCP）──
app.include_router(create_tools_router(get_tool_agent))
app.include_router(create_skills_router(get_tool_agent))
app.include_router(
    create_mcp_router(
        mcp_manager_provider=lambda: deps.mcp_client_manager,
        agent_provider=get_tool_agent,
    )
)

# ── Knowledge（RAG）──
app.include_router(
    create_rag_router(
        local_rag_provider=get_local_rag_service,
        llm_provider=get_llm,
    )
)


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
