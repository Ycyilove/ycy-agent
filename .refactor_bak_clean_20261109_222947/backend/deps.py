"""全局单例 + 7 个 factory（B9-a）。

从 backend/app.py 提取：
    - _rss()
    - 8 个全局变量（tool_agent / mcp_client_manager / ...）
    - 7 个 factory（get_tool_agent / get_llm / ...）

app.py 保留 FastAPI + CORS + include_router + app_lifespan（B9-b 缩为 1 行调 boot_all）。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import psutil

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
import tools.loader
from tools.agent import ToolAgent
from mcp_client import MCPClientManager
from mcp_client.worker import MCPWorker, MCPManagerProxy
from core.task.service import AgentTaskService
from core.memory.session import get_session_memory
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════
def _rss():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

# ─────────────────────────────────────────────────────────────
# 全局单例

tool_agent: Optional[ToolAgent] = None
mcp_client_manager: Optional[MCPManagerProxy] = None
mcp_worker: Optional[MCPWorker] = None
agent_task_service: Optional[AgentTaskService] = None
tool_shortlister = None
local_rag_service = None
llm_instance = None
multimodal_client = None


# ═══════════════════════════════════════════════════════════
# 7 个 factory
# ═══════════════════════════════════════════════════════════
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
