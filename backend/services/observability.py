"""Langfuse 可观测性集成（方向 A）。

设计：
    - 未配置 LANGFUSE_PUBLIC_KEY / SECRET_KEY 时静默降级为 no-op
    - 通过 contextvars 把 per-turn callback handler 传递到
      asyncio.to_thread 里（否则 to_thread 里取不到当前 turn）
    - 每个 turn 一个 handler，同一 turn 内所有 LLM 调用归到同一 trace
    - MultimodalClient 走 Anthropic SDK，本次不接入

对外接口：
    init_langfuse() -> bool
    get_callback_handler() -> Optional[CallbackHandler]
    set_turn_context(session_id, task_id, question) -> contextmanager
    flush_langfuse() -> None
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from ..config import (
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

logger = logging.getLogger(__name__)

# ── 模块级状态 ──
_enabled: bool = False

# 当前 turn 的 callback handler；由 set_turn_context 设置
_handler_var: ContextVar[Optional[Any]] = ContextVar(
    "langfuse_handler", default=None
)


# ── 初始化 ──

def init_langfuse() -> bool:
    """启动时调用。未启用 / 未配置 / 依赖缺失时静默返回 False。"""
    global _enabled

    if not LANGFUSE_ENABLED:
        logger.info("[langfuse] LANGFUSE_ENABLED=false，可观测性关闭")
        _enabled = False
        return False

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.info("[langfuse] 未配置 PUBLIC/SECRET KEY，可观测性关闭")
        _enabled = False
        return False

    try:
        import langfuse  # noqa: F401
    except ImportError:
        logger.warning("[langfuse] 未安装 langfuse 包，可观测性关闭")
        _enabled = False
        return False

    _enabled = True
    logger.info(
        "[langfuse] 已启用 host=%s", LANGFUSE_HOST or "(default)"
    )
    return True


def _make_handler(
    session_id: Optional[str],
    task_id: Optional[str],
    question: Optional[str],
) -> Optional[Any]:
    """兼容 langfuse v2 / v3 的 CallbackHandler 构造。"""
    CallbackHandler = None
    try:
        from langfuse.callback import CallbackHandler  # v2
    except ImportError:
        try:
            from langfuse.langchain import CallbackHandler  # v3
        except ImportError:
            logger.warning("[langfuse] CallbackHandler 不可用")
            return None

    try:
        return CallbackHandler(
            session_id=session_id,
            trace_name=task_id or "agent-task",
            metadata={"question": (question or "")[:500]},
        )
    except TypeError:
        # v3 参数名可能不同，退到最简形态
        try:
            return CallbackHandler()
        except Exception:
            logger.exception("[langfuse] CallbackHandler 创建失败")
            return None
    except Exception:
        logger.exception("[langfuse] CallbackHandler 创建失败")
        return None


# ── 查询 ──

def get_callback_handler() -> Optional[Any]:
    """给 LangChain LLM 调用用。未启用或无 turn context 时返回 None。"""
    if not _enabled:
        return None
    return _handler_var.get()


# ── turn context ──

@contextmanager
def set_turn_context(
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    question: Optional[str] = None,
) -> Iterator[None]:
    """把一个 turn 的所有 LLM 调用绑定到同一个 Langfuse trace。

    未启用时是 no-op；handler 通过 contextvars 传递给 asyncio.to_thread。
    """
    if not _enabled:
        yield
        return

    handler = _make_handler(session_id, task_id, question)
    token = _handler_var.set(handler)
    try:
        yield
    finally:
        _handler_var.reset(token)
        # 每个 turn 结束 flush，避免长连接下 trace 迟迟不上报
        flush_langfuse()


def flush_langfuse() -> None:
    """尽力 flush；失败不影响业务。"""
    if not _enabled:
        return
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        # flush 失败无所谓
        pass