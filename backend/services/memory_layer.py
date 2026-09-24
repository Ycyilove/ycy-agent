"""Mem0 结构化事实层（方向 B，方案 A）。

设计：
    - 增强而非替换 session_memory：session_memory 存全量消息，
      本模块只存从消息/工具结果里抽出的结构化事实
    - 事实提取用正则规则（不调 LLM），符合"零额外 LLM 成本"约束
    - add(..., infer=False)：跳过 Mem0 的 LLM 提取
    - embedder 用本地 sentence-transformers，向量库用 Qdrant 本地文件
    - 未配置 / init 失败时静默 no-op，不影响主流程

事实类型（先做 3 种）：
    - file_path      任意出现的文件路径
    - file_modified  写类工具成功操作的文件
    - user_intent    预留（本版未启用）

对外接口：
    init_mem0() -> bool
    remember_tool_success(session_id, tool_name, params, result) -> None
    remember_user_message(session_id, message) -> None
    recall_file_paths(session_id, limit=5) -> List[str]
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..config import MEM0_ENABLED, MEM0_API_KEY, MEM0_CONFIG

logger = logging.getLogger(__name__)

_memory: Optional[Any] = None
_enabled: bool = False


# ─────────────────────────────────────────────────────────────
# 事实提取（正则规则，不调 LLM）
# ─────────────────────────────────────────────────────────────

# 路径模式：
#   - Windows: D:\a\b.py  C:/a/b.py
#   - Unix:    /home/user/a.py
#   - 相对:    ./data/a.py   data/broken.py
_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s'\"<>|*?]+"
    r"|\.{1,2}[\\/][^\s'\"<>|*?]+"
    r"|/[^\s'\"<>|*?]+"
    r"|[\w\-]+(?:[\\/][\w\-]+)+)"
    r"\.\w{1,10}",
    re.IGNORECASE,
)

# 写类工具关键词（对齐 orchestrator 的 _WRITE_TOOL_HINTS）
_WRITE_TOOL_HINTS = (
    "write", "edit", "create", "delete", "remove",
    "move", "rename", "mkdir", "append", "insert",
    "update", "patch", "save", "overwrite",
)

_PATH_KEYS = ("file_path", "path", "old_path", "output_path", "directory")


def _is_write_tool(tool_name: str) -> bool:
    lower = (tool_name or "").lower()
    return any(h in lower for h in _WRITE_TOOL_HINTS)


def _extract_paths_from_text(text: str) -> List[str]:
    if not text:
        return []
    found = _PATH_PATTERN.findall(text)
    seen = set()
    result = []
    for p in found:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _extract_paths_from_params(tool_name: str, params: Dict[str, Any]) -> List[str]:
    if not isinstance(params, dict):
        return []
    result = []
    for key in _PATH_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            result.append(value.strip())
    # 写类工具：额外扫描其它字符串参数里的路径
    if _is_write_tool(tool_name):
        for key, value in params.items():
            if key in _PATH_KEYS:
                continue
            if isinstance(value, str):
                result.extend(_extract_paths_from_text(value))
    return result


def _extract_paths_from_result(result: Any) -> List[str]:
    """从工具返回值里抽路径（dict / str）。"""
    if result is None:
        return []
    if isinstance(result, str):
        return _extract_paths_from_text(result)
    if isinstance(result, dict):
        paths: List[str] = []
        for key, value in result.items():
            if not isinstance(value, str):
                continue
            if key.lower() in {"new_path", "path", "file_path", "output_path"}:
                paths.append(value.strip())
            elif _is_write_tool(str(key)) or key.lower() in {"message", "output"}:
                paths.extend(_extract_paths_from_text(value))
        return paths
    return []


def _dedup(paths: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ─────────────────────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────────────────────

def _build_config() -> Dict[str, Any]:
    """构造 Mem0 config。

    方案 A：事实提取用规则（infer=False），不调 LLM。
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    hf_cache = os.path.join(project_root, "models")
    qdrant_path = os.path.join(project_root, "data", "mem0_qdrant")

    return {
        "llm": {
            # infer=False 时不会真正调用，但 Mem0 需要占位配置
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": "sk-placeholder",
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "model_kwargs": {
                    "device": "cpu",
                    "cache_folder": hf_cache,
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "agent_facts",
                "path": qdrant_path,
            },
        },
    }


def init_mem0() -> bool:
    """启动时调用。未启用 / 依赖缺失时静默返回 False。"""
    global _memory, _enabled

    if not MEM0_ENABLED:
        logger.info("[mem0] MEM0_ENABLED=false，结构化记忆关闭")
        _enabled = False
        return False

    try:
        from mem0 import Memory  # noqa: F401
    except ImportError:
        logger.info("[mem0] 未安装 mem0ai，结构化记忆关闭")
        _enabled = False
        return False

    try:
        config = _build_config()
        from mem0 import Memory
        _memory = Memory.from_config(config)
        _enabled = True
        logger.info("[mem0] 已启用 (backend=local, embed=hf, store=qdrant)")
        return True
    except Exception:
        logger.exception("[mem0] 初始化失败，结构化记忆关闭")
        _memory = None
        _enabled = False
        return False


# ─────────────────────────────────────────────────────────────
# 写入
# ─────────────────────────────────────────────────────────────

def _add_fact(session_id: str, fact: str, fact_type: str) -> None:
    if not _enabled or _memory is None or not session_id or not fact:
        return
    try:
        _memory.add(
            [{"role": "user", "content": fact}],
            user_id=session_id,
            infer=False,   # 关键：跳过 LLM 提取
            metadata={"fact_type": fact_type},
        )
    except Exception:
        logger.exception("[mem0] add 失败: type=%s fact=%r", fact_type, fact[:120])


def remember_tool_success(
    session_id: str,
    tool_name: str,
    params: Dict[str, Any],
    result: Any,
) -> None:
    """工具成功后提取事实。

    只在写类工具成功后写入 file_modified；读类工具只在参数里出现
    明确路径时写 file_path。避免每次调用都触发存储。
    """
    if not _enabled or not session_id:
        return

    params_paths = _extract_paths_from_params(tool_name, params)
    result_paths = _extract_paths_from_result(result)

    if _is_write_tool(tool_name):
        # 写类：参数 + 返回值都算 file_modified
        for p in _dedup(params_paths + result_paths):
            _add_fact(session_id, p, "file_modified")
    else:
        # 读类：只记参数里明确给出的路径
        for p in _dedup(params_paths):
            _add_fact(session_id, p, "file_path")


def remember_user_message(session_id: str, message: str) -> None:
    """从用户消息提取路径。"""
    if not _enabled or not session_id or not message:
        return
    for p in _dedup(_extract_paths_from_text(message)):
        _add_fact(session_id, p, "file_path")


# ─────────────────────────────────────────────────────────────
# 检索
# ─────────────────────────────────────────────────────────────

def _iter_results(hits: Any) -> List[str]:
    """兼容 Mem0 search / get_all 的返回格式。"""
    if not hits:
        return []
    if isinstance(hits, dict):
        items = hits.get("results") or hits.get("memories") or []
    elif isinstance(hits, list):
        items = hits
    else:
        return []

    texts: List[str] = []
    for h in items:
        if isinstance(h, dict):
            t = h.get("memory") or h.get("text") or h.get("content")
        elif isinstance(h, str):
            t = h
        else:
            t = None
        if t:
            texts.append(str(t))
    return texts


def recall_file_paths(session_id: str, limit: int = 5) -> List[str]:
    """检索本会话的文件路径。失败返回空列表。

    优先 get_all（拿最近写入的），退到 search。
    """
    if not _enabled or _memory is None or not session_id:
        return []

    texts: List[str] = []
    try:
        if hasattr(_memory, "get_all"):
            hits = _memory.get_all(user_id=session_id, limit=limit * 3)
            texts = _iter_results(hits)
        else:
            hits = _memory.search(
                query="file path", user_id=session_id, limit=limit * 3
            )
            texts = _iter_results(hits)
    except Exception:
        logger.exception("[mem0] recall 失败")
        return []

    out: List[str] = []
    seen = set()
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out