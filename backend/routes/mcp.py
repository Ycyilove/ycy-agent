"""MCP 管理路由。

manager 是从 app 注入的 MCPManagerProxy：
    - connect / disconnect / discover / refresh 走 worker task，安全
    - 同步方法和属性直接转发给真 manager
"""

import asyncio
import os
import shutil
import subprocess

import logging
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from mcp_client.config import MCPServerConfig
from mcp_client.dynamic_store import (
    save_server,
    delete_server,
    set_server_enabled,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 请求体
# ─────────────────────────────────────────────────────────────

class AddServerPayload(BaseModel):
    """URL 导入（原有接口保留）。"""
    url: str
    name: str | None = None


class ServerConfigBody(BaseModel):
    """通用配置导入。

    支持 stdio / sse / http 三种 transport。
    - stdio: 需要 command，args 可选
    - sse/http: 需要 url
    """
    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env_from: Optional[Dict[str, str]] = None
    header_env: Optional[Dict[str, str]] = None
    connect_timeout: float = 10.0
    call_timeout: float = 30.0
    retries: int = 2
    enabled: bool = True


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────

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


def _config_to_mapping(config: MCPServerConfig) -> dict:
    """把 MCPServerConfig 序列化回 dict，供 save_server 用。"""
    return {
        "transport": config.transport,
        "command": config.command,
        "args": list(config.args),
        "url": config.url,
        "env_from": dict(config.env_from),
        "header_env": dict(config.header_env),
        "connect_timeout": config.connect_timeout,
        "call_timeout": config.call_timeout,
        "retries": config.retries,
        "enabled": config.enabled,
    }


def _unregister_tools(manager: Any, agent: Any, name: str) -> None:
    """反注册某个 server 在 ToolAgent 里注册过的所有工具。"""
    for tool_name in list(manager.server_tool_names.get(name, set())):
        try:
            agent.unregister_dynamic_tool(tool_name)
        except Exception:
            logger.warning("反注册工具失败: %s", tool_name)
    manager.server_tool_names.pop(name, None)

def _resolve_command(command: str) -> Optional[str]:
    """跨平台解析命令路径。

    Windows 上 npx / uvx 等是 .cmd 文件，shutil.which 默认只找 .exe，
    需要显式加上 PATHEXT 里的后缀再试。
    """
    # 先按原样找
    resolved = shutil.which(command)
    if resolved:
        return resolved

    # Windows：按 PATHEXT 逐个后缀试
    if os.name == "nt":
        pathext = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
        for ext in pathext.split(";"):
            ext = ext.strip()
            if not ext:
                continue
            resolved = shutil.which(command + ext)
            if resolved:
                return resolved
    return None


async def _precheck_stdio(command: str, args: list) -> Optional[str]:
    """stdio 命令预检。返回 None 表示通过，否则返回错误原因。"""
    resolved = _resolve_command(command)
    if resolved is None:
        return f"命令不存在: {command}（检查 PATH，或改用完整路径）"

    # 2. 尝试跑一次 --help（不阻塞太久）
    #    有些 MCP server 不认 --help，会在几秒内退出——这不算错误
    try:
        proc = await asyncio.create_subprocess_exec(
            resolved, *args, "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=8.0
            )
        except asyncio.TimeoutError:
            # 超时说明命令在跑（可能在等 stdin），视为通过
            try:
                proc.kill()
            except Exception:
                pass
            return None

        # 退出码 0 或 1 都算"命令能启动"
        # 关键看 stderr 里有没有 "No module named" / "not found" 之类
        err_text = (stderr or b"").decode("utf-8", errors="replace")

        # Python 模块不存在
        if "No module named" in err_text:
            # 提取模块名
            import re
            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", err_text)
            mod = m.group(1) if m else "?"
            return (
                f"Python 模块不存在: {mod}\n"
                f"  提示：pip install {mod.split('.')[0]}"
            )

        # npm 包不存在
        if "404" in err_text and "npm" in err_text.lower():
            return f"npm 包不存在（404）\n  stderr: {err_text[:200]}"

        # 其他明确的错误
        if proc.returncode not in (0, 1) and err_text.strip():
            return f"命令启动失败（exit={proc.returncode}）\n  stderr: {err_text[:300]}"

        return None

    except FileNotFoundError:
        return f"命令不存在: {command}"
    except Exception as exc:
        # 预检本身出错 → 不阻断，让真正的连接去暴露问题
        logger.warning("[dsh][mcp] precheck raised: %s", exc)
        return None

async def _connect_and_discover(
    manager: Any,
    agent: Any,
    name: str,
) -> None:
    """连接并发现工具。失败时清理已注册的配置和连接。"""
    try:
        await manager.connect(name)
        await manager.discover(
            name,
            agent.register_dynamic_tool,
            agent.unregister_dynamic_tool,
        )
    except Exception:
        try:
            await manager.disconnect(name)
        except Exception:
            logger.exception("清理连接失败: %s", name)
        manager.unregister_config(name)
        raise


# ─────────────────────────────────────────────────────────────
# 路由工厂
# ─────────────────────────────────────────────────────────────

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

    # ── URL 导入（原有） ──

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
            await _connect_and_discover(manager, agent, name)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"连接失败: {exc}") from exc

        try:
            save_server(name, _config_to_mapping(config))
        except Exception:
            logger.exception("持久化 MCP 配置失败: %s", name)

        return {"status": "success", "name": name, "saved": True}

    # ── 通用配置导入（新增） ──

    @router.post("/api/mcp/servers/config")
    async def add_mcp_server_config(body: ServerConfigBody):
        """从完整配置添加 MCP server。

        支持三种 transport：
        - stdio: 必须提供 command
        - sse / http: 必须提供 url

        连接失败时保留配置（saved=true），前端可以稍后重试。
        """
        manager = mcp_manager_provider()
        if manager is None:
            raise HTTPException(status_code=503, detail="MCP 管理器未初始化")

        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name 不能为空")

        if manager.has_server(name):
            raise HTTPException(status_code=409, detail=f"MCP 已存在: {name}")

        transport = (body.transport or "stdio").strip().lower()
        if transport not in ("stdio", "sse", "http"):
            raise HTTPException(
                status_code=400, detail=f"不支持的 transport: {transport}"
            )

        if transport == "stdio":
            if not body.command:
                raise HTTPException(status_code=400, detail="stdio 必须提供 command")
            args = body.args or []
            url = None
        else:
            if not body.url:
                raise HTTPException(
                    status_code=400, detail=f"{transport} 必须提供 url"
                )
            args = []
            url = body.url.strip()

        try:
            config = MCPServerConfig.from_mapping(
                name,
                {
                    "transport": transport,
                    "command": body.command,
                    "args": args,
                    "url": url,
                    "env_from": body.env_from or {},
                    "header_env": body.header_env or {},
                    "connect_timeout": body.connect_timeout,
                    "call_timeout": body.call_timeout,
                    "retries": body.retries,
                    "enabled": body.enabled,
                },
            )
        except Exception as exc:
            logger.exception("构造 MCPServerConfig 失败")
            raise HTTPException(
                status_code=400, detail=f"配置校验失败: {exc}"
            ) from exc

        # ── stdio 预检：确认命令能启动 ──
        if transport == "stdio":
            err = await _precheck_stdio(config.command, list(config.args))
            if err:
                raise HTTPException(
                    status_code=400,
                    detail=f"命令预检失败：\n{err}",
                )

        # 先写盘（即使连接失败，配置也保留）
        try:
            save_server(name, _config_to_mapping(config))
        except Exception as exc:
            logger.exception("save_server failed")
            raise HTTPException(
                status_code=500, detail=f"保存配置失败: {exc}"
            ) from exc

        # 注册 + 连接
        manager.register_config(config)
        agent = agent_provider()

        if not config.enabled:
            manager.statuses.setdefault(name, {}).update(
                {"status": "disabled", "enabled": False}
            )
            return {
                "status": "success",
                "name": name,
                "saved": True,
                "connected": False,
                "message": "已保存配置（disabled 状态，未连接）",
            }

        try:
            await _connect_and_discover(manager, agent, name)
        except Exception as exc:
            logger.exception("连接 MCP 失败: %s", name)
            return {
                "status": "partial",
                "name": name,
                "saved": True,
                "connected": False,
                "message": f"已保存配置，但连接失败: {exc}",
            }

        return {
            "status": "success",
            "name": name,
            "saved": True,
            "connected": True,
        }

    # ── 删除 ──

    @router.delete("/api/mcp/servers/{name}")
    async def remove_mcp_server(name: str):
        manager = mcp_manager_provider()
        if manager is None:
            raise HTTPException(status_code=503, detail="MCP 管理器未初始化")
        if not manager.has_server(name):
            raise HTTPException(status_code=404, detail=f"MCP 不存在: {name}")

        agent = agent_provider()
        _unregister_tools(manager, agent, name)

        try:
            await manager.disconnect(name)
        except Exception:
            logger.exception("断开 MCP 失败: %s", name)

        manager.unregister_config(name)

        try:
            delete_server(name)
        except Exception:
            logger.exception("删除 MCP 配置失败: %s", name)

        return {"status": "success", "removed": True}

    # ── 启用/禁用 ──

    @router.post("/api/mcp/servers/{name}/toggle")
    async def toggle_mcp_server(name: str, payload: dict = Body(default={})):
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
            await _connect_and_discover(manager, agent, name)
        except Exception as exc:
            manager.statuses[name].update(
                {"status": "error", "error": str(exc), "enabled": True}
            )
            raise HTTPException(status_code=502, detail=f"连接失败: {exc}") from exc

        try:
            set_server_enabled(name, True)
        except Exception:
            logger.exception("写入 enabled=true 失败: %s", name)
        return {"status": "success", "name": name, "enabled": True}

    # ── 重载 ──

    @router.post("/api/mcp/reload")
    async def reload_mcp_configs():
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

        for name in removed | changed:
            _unregister_tools(manager, agent, name)
            try:
                await manager.disconnect(name)
            except Exception:
                logger.exception("断开 MCP 失败: %s", name)
            manager.unregister_config(name)

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

    # ── 刷新单个 server ──

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