"""MCP 连接生命周期的 worker。

问题：
    mcp SDK 的 stdio/sse 客户端使用 anyio task group。
    它们的 __aenter__ 和 __aexit__ 必须在同一个 asyncio task 里，
    否则抛 "Attempted to exit cancel scope in a different task"。

方案：
    启动一个常驻 task，所有 connect/disconnect/discover/refresh/close/start
    都通过消息队列路由到这个 task 执行。
    连接从建立到关闭的整个生命周期都在同一个 task 里。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPWorker:
    """常驻 task + 消息队列，串行执行 manager 的有状态异步方法。"""

    def __init__(self, manager: Any):
        self.manager = manager
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="mcp-worker")
        await self._ready.wait()

    async def _run(self) -> None:
        self._ready.set()
        while True:
            op, args, kwargs, future = await self._queue.get()
            if op == "__stop__":
                try:
                    await self.manager.close()
                    if not future.done():
                        future.set_result(None)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                return
            try:
                method = getattr(self.manager, op)
                result = await method(*args, **kwargs)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    async def call(self, op: str, *args: Any, **kwargs: Any) -> Any:
        if self._task is None or self._task.done():
            raise RuntimeError("MCP worker 未运行")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((op, args, kwargs, future))
        return await future

    async def stop(self) -> None:
        if self._task is None:
            return
        try:
            await self.call("__stop__")
        except Exception:
            logger.exception("停止 MCP worker 失败")
        try:
            await self._task
        except Exception:
            logger.exception("MCP worker task 退出异常")
        self._task = None


class MCPManagerProxy:
    """MCPClientManager 的代理。

    - connect / discover / disconnect / refresh / close / start 走 worker task
    - 其余同步方法、属性直接转发给真 manager
    - call_tool / read_resource 不走 worker（它们不涉及 task group 进出）
    """

    _WORKER_METHODS = frozenset({
        "connect",
        "discover",
        "disconnect",
        "refresh",
        "close",
        "start",
    })

    def __init__(self, worker: MCPWorker, manager: Any):
        object.__setattr__(self, "_worker", worker)
        object.__setattr__(self, "_manager", manager)

    def __getattr__(self, name: str) -> Any:
        if name in self._WORKER_METHODS:
            async def _call(*args: Any, **kwargs: Any) -> Any:
                return await self._worker.call(name, *args, **kwargs)
            return _call
        return getattr(self._manager, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._manager, name, value)