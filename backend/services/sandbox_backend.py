"""E2B 沙箱后端（方向 C）。

设计：
    - 接口与 tools/tools_def/sandbox.py 的 ExecutionResult 对齐
    - run_python_code(code, timeout) 签名保持不变
    - SANDBOX_BACKEND=local 或 E2B 初始化失败时，get_backend() 返回 None
      调用方走本地实现
    - E2B SDK 的 run_code() 是同步阻塞的，本模块内部不做 async 包装，
      由调用方（execute_tool_async）在线程池里跑即可

对外接口：
    get_backend() -> Optional[E2BBackend]
    E2BBackend.execute(code, timeout) -> dict
    close_backend() -> None
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ..config import (
    E2B_API_KEY,
    E2B_TEMPLATE,
    SANDBOX_BACKEND,
)

logger = logging.getLogger(__name__)

_backend: Optional["E2BBackend"] = None
_initialized = False


class E2BBackend:
    """E2B Code Interpreter 沙箱封装。

    每次 execute 复用同一个 Sandbox 实例，减少冷启动。
    Sandbox 在 close_backend() 时统一关闭。
    """

    def __init__(self, api_key: str, template: str = "base"):
        from e2b_code_interpreter import Sandbox
        self._sandbox = Sandbox.create(
            template=template,
            api_key=api_key,
        )
        logger.info(
            "[e2b] sandbox created: template=%s id=%s",
            template,
            getattr(self._sandbox, "sandbox_id", "(unknown)"),
        )

    def execute(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        """执行代码，返回与 ExecutionResult 对齐的 dict。"""
        start = time.time()
        try:
            execution = self._sandbox.run_code(code, timeout=timeout)
        except Exception as exc:
            logger.exception("[e2b] run_code raised")
            return {
                "success": False,
                "output": "",
                "error": f"{type(exc).__name__}: {exc}",
                "execution_time": time.time() - start,
            }

        # stdout / stderr：SDK 返回的是 list[str]，拼成单串
        stdout = "".join(execution.logs.stdout or [])
        stderr = "".join(execution.logs.stderr or [])

        # 错误：execution.error 是对象，取 traceback 或 value
        error_obj = execution.error
        if error_obj is None:
            success = True
            error_msg = None
        else:
            success = False
            # traceback 最可靠，退到 value / name
            error_msg = (
                getattr(error_obj, "traceback", None)
                or getattr(error_obj, "value", None)
                or getattr(error_obj, "name", None)
                or str(error_obj)
            )

        output = stdout
        if stderr:
            output += "\n[stderr]: " + stderr

        return {
            "success": success,
            "output": output.strip() if output.strip() else "代码执行完成（无输出）",
            "error": error_msg,
            "execution_time": time.time() - start,
        }

    def close(self) -> None:
        try:
            self._sandbox.close()
            logger.info("[e2b] sandbox closed")
        except Exception:
            logger.exception("[e2b] sandbox close failed")


def get_backend() -> Optional[E2BBackend]:
    """按配置返回 E2B backend；不可用时返回 None。"""
    global _backend, _initialized

    if _initialized:
        return _backend

    _initialized = True

    if SANDBOX_BACKEND != "e2b":
        logger.info("[e2b] SANDBOX_BACKEND=%r, using local sandbox", SANDBOX_BACKEND)
        _backend = None
        return None

    if not E2B_API_KEY:
        logger.warning("[e2b] E2B_API_KEY 未配置，回退本地沙箱")
        _backend = None
        return None

    try:
        _backend = E2BBackend(api_key=E2B_API_KEY, template=E2B_TEMPLATE)
        logger.info("[e2b] backend ready (template=%s)", E2B_TEMPLATE)
    except Exception:
        logger.exception("[e2b] 初始化失败，回退本地沙箱")
        _backend = None

    return _backend


def close_backend() -> None:
    """lifespan 关闭时调用。"""
    global _backend, _initialized
    if _backend is not None:
        _backend.close()
    _backend = None
    _initialized = False