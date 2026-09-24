"""MCP 管理路由。

manager 是从 app 注入的 MCPManagerProxy：
    - connect / disconnect / discover / refresh 走 worker task，安全
    - 同步方法和属性直接转发给真 manager
"""

import logging
from typing import Any, Callable
from urllib.parse import urlparse

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from mcp_client.config import MCPServerConfig
from mcp_client.dynamic_store import (
    save_server,
    delete_server,
    set_server_enabled,
)

logger = logging.getLogger(__name__)


class AddServerPayload(BaseModel):
    url: str
    name: str | None = None


def _infer_transport(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        if parsed.path.rstrip("/").endswith("/sse"):
            return "sse"
        return "http"
    return "stdio"


def _infer_name(url: str, explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port
    if host:
        name = f"{host}-{port}" if port else host
    else:
        name = f"mcp-{abs(hash(url)) % 100000}"
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
    return safe[:64] or "mcp"


def _unregister_tools(manager: Any, agent: Any, name: str) -> None:
    """反注册某个 server 在 ToolAgent 里注册过的所有工具。"""
    for tool_name in list(manager.server_tool_names.get(name, set())):
        try:
            agent.unregister_dynamic_tool(tool_name)
        except Exception:
            logger.warning("反注册工具失败: %s", tool_name)
    manager.server_tool_names.pop(name, None)


def create_mcp_router(
    mcp_manager_provider: Callable[[], Any],
    agent_provider: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/mcp/status")
    async def mcp_status():
        manager = mcp_manager_provider()
        if manager is None:
            return {"status": "disabled", "servers": []}
        return {"status": "success", "servers": manager.status()}

    @router.post("/api/mcp/servers")
    async def add_mcp_server(payload: AddServerPayload):
        manager = mcp_manager_provider()
        if manager is None:
            raise HTTPException(status_code=503, detail="MCP 管理器未初始化")

        url = (payload.url or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="缺少 url")

        name = _infer_name(url, payload.name)
        if manager.has_server(name):
            raise HTTPException(status_code=409, detail=f"MCP 已存在: {name}")

        transport = _infer_transport(url)
        if transport == "stdio":
            raise HTTPException(
                status_code=400,
                detail="暂不支持从 URL 导入 stdio MCP，请提供 http(s) URL",
            )

        try:
            config = MCPServerConfig.from_mapping(
                name, {"transport": transport, "url": url}
            )
        except Exception as exc:
            logger.exception("构造 MCPServerConfig 失败")
            raise HTTPException(status_code=400, detail=f"配置无效: {exc}") from exc

        manager.register_config(config)
        agent = agent_provider()

        try:
            await manager.connect(name)
            await manager.discover(
                name,
                agent.register_dynamic_tool,
                agent.unregister_dynamic_tool,
            )
        except Exception as exc:
            try:
                await manager.disconnect(name)
            except Exception:
                logger.exception("清理连接失败: %s", name)
            manager.unregister_config(name)
            raise HTTPException(status_code=502, detail=f"连接失败: {exc}") from exc

        try:
            save_server(name, {"transport": transport, "url": url, "enabled": True})
        except Exception:
            logger.exception("持久化 MCP 配置失败: %s", name)

        return {"status": "success", "name": name}

    @router.delete("/api/mcp/servers/{name}")
    async def remove_mcp_server(name: str):
        manager = mcp_manager_provider()
        if manager is None:
            raise HTTPException(status_code=503, detail="MCP 管理器未初始化")
        if not manager.has_server(name):
            raise HTTPException(status_code=404, detail=f"MCP 不存在: {name}")

        agent = agent_provider()

        # 1. 反注册工具（提问不再用到它）
        _unregister_tools(manager, agent, name)

        # 2. 真正断开连接（走 worker task，安全）
        try:
            await manager.disconnect(name)
        except Exception:
            logger.exception("断开 MCP 失败: %s", name)

        # 3. 注销内存配置
        manager.unregister_config(name)

        # 4. 从 mcp_servers.json 删除
        try:
            delete_server(name)
        except Exception:
            logger.exception("删除 MCP 配置失败: %s", name)

        return {"status": "success", "removed": True}

    @router.post("/api/mcp/servers/{name}/toggle")
    async def toggle_mcp_server(name: str, payload: dict = Body(default={})):
        """启用/禁用 MCP。

        禁用：反注册工具 + 断开连接 + 写入 enabled=false。configs 保留。
        启用：连接 + discover + 写入 enabled=true。
        """
        manager = mcp_manager_provider()
        if manager is None:
            raise HTTPException(status_code=503, detail="MCP 管理器未初始化")
        if not manager.has_server(name):
            raise HTTPException(status_code=404, detail=f"MCP 不存在: {name}")

        current = manager.statuses.get(name, {}).get("enabled", True)
        if isinstance(payload, dict) and "enabled" in payload:
            want_enabled = bool(payload["enabled"])
        else:
            want_enabled = not current

        agent = agent_provider()

        if not want_enabled:
            _unregister_tools(manager, agent, name)
            try:
                await manager.disconnect(name)
            except Exception:
                logger.exception("断开 MCP 失败: %s", name)
            manager.statuses[name].update(
                {"status": "disabled", "enabled": False, "tools": 0}
            )
            try:
                set_server_enabled(name, False)
            except Exception:
                logger.exception("写入 enabled=false 失败: %s", name)
            return {"status": "success", "name": name, "enabled": False}

        manager.statuses[name].update({"status": "configured", "enabled": True})
        try:
            await manager.connect(name)
            await manager.discover(
                name,
                agent.register_dynamic_tool,
                agent.unregister_dynamic_tool,
            )
        except Exception as exc:
            try:
                await manager.disconnect(name)
            except Exception:
                pass
            manager.statuses[name].update(
                {"status": "error", "error": str(exc), "enabled": True}
            )
            raise HTTPException(status_code=502, detail=f"连接失败: {exc}") from exc

        try:
            set_server_enabled(name, True)
        except Exception:
            logger.exception("写入 enabled=true 失败: %s", name)
        return {"status": "success", "name": name, "enabled": True}

    @router.post("/api/mcp/reload")
    async def reload_mcp_configs():
        """同步 mcp_servers.json 与内存 manager.configs。

        - 文件新增的：注册 + 连接
        - 文件删除的：反注册工具 + 断开 + 注销
        - 配置变化的：断开 + 重新注册 + 重连
        - 被禁用的（enabled=false）：跳过连接
        """
        manager = mcp_manager_provider()
        if manager is None:
            logger.warning("[dsh][mcp] reload skipped: manager not initialized")
            return {"status": "disabled", "added": [], "removed": [], "changed": []}

        try:
            from mcp_client.config import load_mcp_config
            new_configs = load_mcp_config()
        except Exception as exc:
            logger.exception("读取 MCP 配置文件失败")
            raise HTTPException(status_code=500, detail=f"读取配置失败: {exc}") from exc

        new_map = {c.name: c for c in new_configs}
        old_names = set(manager.configs.keys())
        new_names = set(new_map.keys())

        added = new_names - old_names
        removed = old_names - new_names
        changed = set()
        for name in old_names & new_names:
            old = manager.configs[name]
            new = new_map[name]
            if old.transport != new.transport or old.url != new.url:
                changed.add(name)

        agent = agent_provider()

        # 移除 + 变更：反注册 + 断开 + 注销
        for name in removed | changed:
            _unregister_tools(manager, agent, name)
            try:
                await manager.disconnect(name)
            except Exception:
                logger.exception("断开 MCP 失败: %s", name)
            manager.unregister_config(name)

        # 新增 + 变更：注册 + 尝试连接
        for name in added | changed:
            config = new_map[name]
            manager.register_config(config)
            if not getattr(config, "enabled", True):
                manager.statuses[name].update(
                    {"status": "disabled", "enabled": False}
                )
                continue
            try:
                await manager.connect(name)
                await manager.discover(
                    name,
                    agent.register_dynamic_tool,
                    agent.unregister_dynamic_tool,
                )
            except Exception:
                logger.exception("连接 MCP 失败: %s", name)

        logger.info(
            "[dsh][mcp] reload: added=%r removed=%r changed=%r",
            sorted(added), sorted(removed), sorted(changed),
        )
        return {
            "status": "success",
            "added": sorted(added),
            "removed": sorted(removed),
            "changed": sorted(changed),
        }

    @router.post("/api/mcp/{server_name}/refresh")
    async def refresh_mcp_server(server_name: str):
        manager = mcp_manager_provider()
        if manager is None:
            raise HTTPException(status_code=404, detail="MCP 未配置")
        if not manager.has_server(server_name):
            raise HTTPException(status_code=404, detail=f"MCP 不存在: {server_name}")

        agent = agent_provider()
        try:
            await manager.refresh(
                server_name,
                agent.register_dynamic_tool,
                agent.unregister_dynamic_tool,
            )
            return {"status": "success", "servers": manager.status()}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return router