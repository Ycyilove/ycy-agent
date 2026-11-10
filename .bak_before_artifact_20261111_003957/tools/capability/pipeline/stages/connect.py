"""Stage 7：连接候选 + 注册工具。

local 模式：连已配置 server。
registry 模式：先判断候选是否已配置（intent_index 命中），已配置直接 connect，
              未配置走 config_for_entry + connect + 认证预检。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ConnectStage:
    async def run(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.mode == "local":
            return await self._connect_local(ctx, tools)
        return await self._connect_registry(ctx, tools)

    @staticmethod
    def _server_name(c: Any) -> str:
        if isinstance(c, dict):
            return c.get("server_name") or c.get("local_name") or c.get("registry_name") or ""
        return getattr(c, "server_name", "") or ""

    @staticmethod
    def _entry(c: Any) -> Dict[str, Any]:
        if isinstance(c, dict):
            return c
        return getattr(c, "registry_entry", None) or {}

    async def _connect_local(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        new_tools = dict(tools)
        for cand in ctx.candidates[:3]:
            srv = self._server_name(cand)
            if not srv:
                continue
            if not ctx.manager or srv not in ctx.manager.configs:
                continue
            if srv in ctx.manager.connections:
                from tools import get_registry
                reg = get_registry()
                allowed = ctx.allowed_danger_levels or {"safe", "medium", "high"}
                for name, meta in reg.get_all_metadata().items():
                    if name.startswith("mcp__") and meta.danger_level in allowed:
                        new_tools[name] = meta
                continue
            try:
                await ctx.manager.connect(srv)
                await ctx.manager.discover(
                    srv, ctx.register_tool, ctx.unregister_tool,
                )
                ctx.log_stage("connect(local)", f"connected {srv}")
                from tools import get_registry
                reg = get_registry()
                allowed = ctx.allowed_danger_levels or {"safe", "medium", "high"}
                for name, meta in reg.get_all_metadata().items():
                    if name.startswith("mcp__") and meta.danger_level in allowed:
                        new_tools[name] = meta
                return new_tools
            except Exception as error:
                logger.exception("[connect] local connect %s failed", srv)
                ctx.errors.append(f"{srv}: {error}")
        return new_tools

    async def _connect_registry(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.candidates:
            return tools

        for attempt_idx, chosen in enumerate(ctx.candidates[:5]):
            entry = self._entry(chosen)
            local_name = entry.get("local_name") or self._server_name(chosen)
            if not local_name:
                continue

            # ── 情况 A：已配置（intent_index 命中）→ 直接 connect ──
            if ctx.manager.has_server(local_name):
                if local_name not in ctx.manager.connections:
                    try:
                        ctx.log_stage(
                            "connect(registry)",
                            f"already configured, connect: {local_name}",
                        )
                        await ctx.manager.connect(local_name)
                        await ctx.manager.discover(
                            local_name, ctx.register_tool, ctx.unregister_tool,
                        )
                    except Exception as error:
                        logger.exception(
                            "[connect] connect configured %s failed", local_name,
                        )
                        ctx.errors.append(
                            f"{local_name}: {error}"
                        )
                        continue
                ctx.result = self._format_result(entry, local_name, ctx)
                return tools

            # ── 情况 B：未配置 → config_for_entry + connect ──
            try:
                from mcp_client.registry_client import config_for_entry
                config = await config_for_entry(entry)
            except Exception:
                logger.exception(
                    "[connect] config_for_entry failed: %s",
                    entry.get("registry_name"),
                )
                ctx.errors.append(
                    f"{entry.get('registry_name')}: 无法构造配置"
                )
                continue

            if config is None:
                ctx.errors.append(
                    f"{entry.get('registry_name') or entry.get('local_name')}: "
                    f"config 为 None（缺 url/transport）"
                )
                continue

            ctx.manager.register_config(config)
            try:
                ctx.log_stage(
                    "connect(registry)",
                    f"try {attempt_idx + 1}/{min(len(ctx.candidates), 5)}: "
                    f"{entry.get('registry_name') or local_name}",
                )
                await ctx.manager.connect(local_name)
                await ctx.manager.discover(
                    local_name, ctx.register_tool, ctx.unregister_tool,
                )
            except Exception as error:
                err_str = str(error)
                is_auth = any(
                    k in err_str.lower()
                    for k in (
                        "401", "unauthorized", "authentication",
                        "api key", "missing", "forbidden", "403",
                    )
                )
                tag = "[需要认证]" if is_auth else ""
                ctx.errors.append(
                    f"{entry.get('registry_name')}{tag}: {err_str[:150]}"
                )
                ctx.tried.add(entry.get("registry_name", ""))
                try:
                    ctx.manager.unregister_config(local_name)
                except Exception:
                    pass
                continue

            needs_auth = await self._check_auth(config)
            if needs_auth:
                ctx.tried.add(entry.get("registry_name", ""))
                try:
                    ctx.manager.unregister_config(local_name)
                except Exception:
                    pass
                ctx.errors.append(
                    f"{entry.get('registry_name')}[需要认证]"
                )
                continue

            ctx.tried.add(entry.get("registry_name", ""))
            ctx.result = self._format_result(entry, local_name, ctx)
            return tools

        ctx.errors.append(
            f"尝试了 {len(ctx.candidates[:5])} 个候选，全部失败"
        )
        return tools

    async def _check_auth(self, config: Any) -> bool:
        if not config.header_env:
            return False
        import os
        return any(not os.getenv(env) for env in config.header_env.values())

    def _format_result(self, entry: Dict[str, Any], local_name: str, ctx: Any) -> Dict[str, Any]:
        _tools = [
            spec.public_name
            for spec in ctx.manager.tools.values()
            if spec.server_name == local_name
        ]
        return {
            "success": True,
            "server": entry.get("registry_name", local_name),
            "description": entry.get("description", ""),
            "transport": entry.get("transport"),
            "url": entry.get("url"),
            "tools": _tools,
            "source": "configured" if local_name in ctx.manager.configs else "registry",
        }