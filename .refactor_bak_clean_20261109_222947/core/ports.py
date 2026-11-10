"""跨层协议。core 只依赖本文件，不依赖 tools/rag/mcp_client/backend 具体实现。

依赖方向：backend -> core -> ports <- tools/rag/mcp_client/infra
bootstrap 负责把实现注入 core。

设计要点：
    - 每个端口是 Protocol（鸭子类型），core 侧只调方法，不关心实现
    - 注入用模块级全局变量 + set/get 函数，bootstrap 启动时调一次
    - 所有 get_* 返回 Optional，调用方必须处理 None（未注入时降级）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ── 长期记忆 ──

@runtime_checkable
class MemoryReader(Protocol):
    """读长期记忆。core.task.answer.prompt_build 用。"""

    def recall_recent_facts(
        self, session_id: str, limit: int = 8
    ) -> List[str]: ...

    def recall_file_paths(
        self, session_id: str, limit: int = 5
    ) -> List[str]: ...


@runtime_checkable
class MemoryWriter(Protocol):
    """写长期记忆。tools.orchestration / core.task 用。"""

    def remember_user_message(
        self, session_id: str, message: str
    ) -> None: ...

    def remember_tool_success(
        self,
        session_id: str,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        *,
        origin: str,
    ) -> None: ...


# ── 反向兜底搜索 ──

@runtime_checkable
class SearchPort(Protocol):
    """反向兜底搜索。core.task.answer.fallback_search 用。"""

    async def search_web(
        self, query: str, max_results: int = 5
    ) -> Optional[str]:
        """返回已格式化的搜索结果文本；失败返回 None。"""
        ...


# ── RAG ──

@runtime_checkable
class RAGPort(Protocol):
    """知识库。core.task.answer.asset_resolve 用。"""

    def resolve_asset_tags(
        self, text: str, resources: List[Dict[str, Any]]
    ) -> str: ...

    def search(
        self, query: str, top_k: int, **kwargs: Any
    ) -> List[Dict[str, Any]]: ...

    def get_stats(self) -> Dict[str, Any]: ...


# ── 审批 ──

@runtime_checkable
class ApprovalPort(Protocol):
    """审批状态。core.task.approval 用。"""

    def create(
        self,
        task_id: str,
        session_id: str,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Any: ...

    def decide(
        self, approval_id: str, session_id: str, action: str
    ) -> bool: ...

    def verify(
        self, approval_id: str, tool_name: str, params: Dict[str, Any]
    ) -> bool: ...


# ── 工具目录 ──

@runtime_checkable
class ToolCatalogPort(Protocol):
    """工具目录。core.task.service / orchestration 用。"""

    def list_tools(
        self, allowed_danger_levels: Optional[set] = None
    ) -> Dict[str, Any]: ...

    def get_metadata(self, name: str) -> Optional[Any]: ...

    async def execute(self, tool_call: Any) -> Any: ...

    def format_result(self, result: Any) -> str: ...


# ── 配置 ──

@runtime_checkable
class ConfigPort(Protocol):
    """运行时配置。core.llm / core.memory / infra 用。

    backend.config 是具体实现，core 只读这里的属性。
    """

    DEFAULT_MODEL: str
    LOCAL_MODEL_CONFIG: Any
    MODESCOPE_API_KEY: str
    MODESCOPE_BASE_URL: str
    SILICONFLOW_API_KEY: str
    SILICONFLOW_BASE_URL: str
    YUNDU_API_KEY: str
    YUNDU_BASE_URL: str
    MODEL_SOURCE: Any
    RAG_EMBEDDING_MODEL: str
    LANGFUSE_ENABLED: bool
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str
    SANDBOX_BACKEND: str
    E2B_API_KEY: str
    E2B_TEMPLATE: str


# ── 异常 ──

class ContextLengthExceeded(RuntimeError):
    """prompt 超出模型 context 上限。

    由 provider 层抛，上层（_decide / _fill_params）catch 后降级重试。
    重构前在 core.llm.provider，现在移到 ports，core.llm.provider re-export 保持兼容。
    """


# ── 注入 / 获取 ──

_memory_reader: Optional[MemoryReader] = None
_memory_writer: Optional[MemoryWriter] = None
_search_port: Optional[SearchPort] = None
_rag_port: Optional[RAGPort] = None
_approval_port: Optional[ApprovalPort] = None
_tool_catalog: Optional[ToolCatalogPort] = None
_config: Optional[ConfigPort] = None


def set_memory_reader(p: Optional[MemoryReader]) -> None:
    global _memory_reader
    _memory_reader = p


def get_memory_reader() -> Optional[MemoryReader]:
    return _memory_reader


def set_memory_writer(p: Optional[MemoryWriter]) -> None:
    global _memory_writer
    _memory_writer = p


def get_memory_writer() -> Optional[MemoryWriter]:
    return _memory_writer


def set_search_port(p: Optional[SearchPort]) -> None:
    global _search_port
    _search_port = p


def get_search_port() -> Optional[SearchPort]:
    return _search_port


def set_rag_port(p: Optional[RAGPort]) -> None:
    global _rag_port
    _rag_port = p


def get_rag_port() -> Optional[RAGPort]:
    return _rag_port


def set_approval_port(p: Optional[ApprovalPort]) -> None:
    global _approval_port
    _approval_port = p


def get_approval_port() -> Optional[ApprovalPort]:
    return _approval_port


def set_tool_catalog(p: Optional[ToolCatalogPort]) -> None:
    global _tool_catalog
    _tool_catalog = p


def get_tool_catalog() -> Optional[ToolCatalogPort]:
    return _tool_catalog


def set_config(p: Optional[ConfigPort]) -> None:
    global _config
    _config = p


def get_config() -> Optional[ConfigPort]:
    return _config