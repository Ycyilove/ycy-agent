"""Lazy module loader with background preload support.

设计目标：
    - 启动时不阻塞：app_lifespan 只注册预加载任务，不等待
    - 首次使用时确保加载：调用 ensure_loaded() 时若未加载则等待
    - 状态可查询：get_status() 返回每个模块的加载状态
    - 线程池隔离：每个模块加载在独立线程执行，互不阻塞

关键安全约束：
    - `ensure_module_loaded_sync` **禁止在事件循环线程里调用**。
      在事件循环里同步 import 重模块会阻塞事件循环 30+ 秒，
      并可能导致死锁。若检测到事件循环正在运行，直接抛 RuntimeError。
    - 需要同步加载时，请在线程池 worker（例如 asyncio.to_thread 的回调、
      FastAPI 的 sync 路由、独立线程）里调用。

用法：
    from backend.services.lazy_loader import (
        register_lazy_module,
        ensure_module_loaded,
        ensure_module_loaded_sync,
        get_all_module_status,
        preload_all_in_background,
    )

    # 注册一个模块（通常在 app.py 顶层或 lifespan 早期）
    register_lazy_module(
        name="langchain_openai",
        loader=lambda: __import__("langchain_openai"),
        description="LLM 对话所需的 LangChain OpenAI 封装",
    )

    # 在事件循环里（async 函数）
    await ensure_module_loaded("langchain_openai")

    # 在线程池 worker 里（sync 函数）
    ensure_module_loaded_sync("langchain_openai")

    # 后台预加载所有注册的模块（在 lifespan 里启动）
    preload_all_in_background()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class ModuleState:
    """单个模块的加载状态。"""
    name: str
    loader: Callable[[], object]
    description: str = ""
    status: str = "not_loaded"  # not_loaded | loading | loaded | failed
    loaded_at: Optional[float] = None
    load_duration: Optional[float] = None
    error: Optional[str] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _task: Optional[asyncio.Task] = None


# 全局注册表
_modules: Dict[str, ModuleState] = {}


def register_lazy_module(
    name: str,
    loader: Callable[[], object],
    description: str = "",
    replace: bool = False,
) -> ModuleState:
    """注册一个可惰性加载的模块。

    Args:
        name: 模块标识（唯一）
        loader: 同步的可调用对象，执行后模块即可用。
                通常是 `lambda: __import__("module_name")`。
        description: 人类可读的描述，用于状态 API
        replace: 若已存在同名模块，是否覆盖注册

    Returns:
        注册后的 ModuleState
    """
    if name in _modules and not replace:
        logger.debug("[lazy] module already registered: %s", name)
        return _modules[name]

    state = ModuleState(
        name=name,
        loader=loader,
        description=description,
    )
    _modules[name] = state
    logger.info("[lazy] registered module: %s", name)
    return state


async def ensure_module_loaded(name: str, timeout: Optional[float] = None) -> None:
    """确保模块已加载（async 版本）。

    - 已加载 → 立即返回
    - 正在加载 → 等待加载完成
    - 未加载 → 触发加载并等待
    - 加载失败 → 抛出 RuntimeError

    Args:
        name: 模块标识
        timeout: 可选超时（秒）。超时抛 asyncio.TimeoutError。

    Raises:
        KeyError: 模块未注册
        RuntimeError: 模块加载失败
        asyncio.TimeoutError: 加载超时
    """
    if name not in _modules:
        raise KeyError(f"未注册的惰性模块: {name}")

    state = _modules[name]

    # 已加载 → 直接返回
    if state.status == "loaded":
        return

    # 失败过 → 直接抛错，不重试
    if state.status == "failed":
        raise RuntimeError(f"模块 {name} 之前加载失败: {state.error}")

    async def _load():
        # 加锁避免并发重复加载
        async with state._lock:
            if state.status == "loaded":
                return
            if state.status == "failed":
                raise RuntimeError(f"模块 {name} 之前加载失败: {state.error}")

            state.status = "loading"
            start = time.time()
            try:
                # 在独立线程里跑同步 loader，避免阻塞事件循环
                await asyncio.to_thread(state.loader)
                state.status = "loaded"
                state.loaded_at = time.time()
                state.load_duration = time.time() - start
                logger.info(
                    "[lazy] module %s loaded in %.2fs",
                    name, state.load_duration,
                )
            except Exception as error:
                state.status = "failed"
                state.error = f"{type(error).__name__}: {error}"
                logger.exception("[lazy] module %s failed to load", name)
                raise RuntimeError(f"模块 {name} 加载失败: {state.error}") from error

    if timeout is not None:
        await asyncio.wait_for(_load(), timeout=timeout)
    else:
        await _load()


def ensure_module_loaded_sync(name: str, timeout: Optional[float] = None) -> None:
    """同步版本，用于明确不在事件循环里的场景。

    **如果在事件循环里调用，会抛 RuntimeError**——
    这是为了避免"在事件循环线程里同步 import 重模块"导致的
    30 秒阻塞 + 潜在死锁。

    正确的用法：
        - 在线程池 worker 里（例如 FastAPI 的 sync 路由、`asyncio.to_thread` 的回调）
        - 在独立线程里

    错误的用法：
        - 在 `async def` 函数体里直接调用
          （应该用 `await ensure_module_loaded(...)`）

    Raises:
        RuntimeError: 在事件循环里调用时
        KeyError: 模块未注册
        RuntimeError: 模块加载失败
    """
    # 检测是否在事件循环线程里
    try:
        asyncio.get_running_loop()
        in_event_loop = True
    except RuntimeError:
        in_event_loop = False

    if in_event_loop:
        raise RuntimeError(
            f"ensure_module_loaded_sync('{name}') 不能在事件循环线程里调用。"
            f"请改用 'await ensure_module_loaded(\"{name}\")'，"
            f"或用 'await asyncio.to_thread(ensure_module_loaded_sync, \"{name}\")'。"
        )

    # 不在事件循环里 → 安全创建新事件循环执行
    asyncio.run(ensure_module_loaded(name, timeout=timeout))


def preload_all_in_background(
    exclude: Optional[List[str]] = None,
    sequential: bool = False,
) -> None:
    """在后台启动所有已注册模块的预加载。

    Args:
        exclude: 要跳过的模块名列表
        sequential: True = 顺序加载（节省内存峰值，但慢）
                    False = 并发加载（快，但内存峰值高）
    """
    exclude = exclude or []
    names = [n for n in _modules if n not in exclude]
    if not names:
        logger.info("[lazy] no modules to preload")
        return

    logger.info(
        "[lazy] starting background preload for %d modules: %s",
        len(names), names,
    )

    async def _preload_one(name: str):
        try:
            await ensure_module_loaded(name)
        except Exception:
            # 单个失败不影响其他模块
            logger.exception("[lazy] preload failed for %s", name)

    async def _preload_all():
        if sequential:
            for name in names:
                await _preload_one(name)
        else:
            await asyncio.gather(
                *[_preload_one(n) for n in names],
                return_exceptions=True,
            )
        logger.info("[lazy] background preload complete")

    # 用 create_task 让它在事件循环后台跑，不阻塞调用方
    asyncio.create_task(_preload_all())


def get_module_status(name: str) -> Optional[Dict]:
    """查询单个模块的状态。"""
    state = _modules.get(name)
    if state is None:
        return None
    return {
        "name": state.name,
        "description": state.description,
        "status": state.status,
        "loaded_at": state.loaded_at,
        "load_duration": state.load_duration,
        "error": state.error,
    }


def get_all_module_status() -> Dict:
    """查询所有模块的状态。

    Returns:
        {
            "modules": {name: {...}, ...},
            "summary": {
                "total": int,
                "loaded": int,
                "loading": int,
                "failed": int,
                "not_loaded": int,
                "all_ready": bool,
            }
        }
    """
    modules = {name: get_module_status(name) for name in _modules}
    summary = {
        "total": len(modules),
        "loaded": 0,
        "loading": 0,
        "failed": 0,
        "not_loaded": 0,
        "all_ready": False,
    }
    for info in modules.values():
        if info is None:
            continue
        status = info["status"]
        if status in summary:
            summary[status] += 1

    summary["all_ready"] = (
        summary["loaded"] == summary["total"]
        and summary["failed"] == 0
    )

    return {"modules": modules, "summary": summary}