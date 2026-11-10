"""Stage 5（仅 mode=registry）：Registry 搜索。

原 search_and_connect_mcp 的：
    1. 关键词扩展（_expand_keywords）
    2. 多关键词降级搜索（search_configs_multi）
    3. 排除已 tried 的候选

local 模式跳过此阶段。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RegistryStage:
    async def run(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.mode != "registry":
            return tools
        if ctx.short_circuit or ctx.candidates:
            return tools

        # 1. 关键词
        raw_keywords = self._collect_keywords(ctx)
        if not raw_keywords:
            ctx.log_stage("registry", "no keywords, abort")
            return tools

        from tools.mcp_discovery import _expand_keywords
        keywords = _expand_keywords(raw_keywords)
        ctx.log_stage("registry", f"keywords={keywords!r}")

        # 2. 多关键词搜索
        try:
            from mcp_client.registry_client import search_configs_multi
            candidates, meta = await search_configs_multi(
                keywords, limit=ctx.limit,
            )
        except Exception:
            logger.exception("[registry] search_configs_multi failed")
            ctx.errors.append("Registry 搜索失败")
            return tools

        if not candidates:
            if meta.get("network_error"):
                ctx.errors.append("MCP Registry 网络不可达")
            else:
                ctx.errors.append(
                    f"未找到 {keywords!r} 相关服务器"
                )
            ctx.log_stage("registry", "no candidates")
            return tools

        # 3. 排除已 tried
        fresh = [
            c for c in candidates
            if c["registry_name"] not in ctx.tried
        ]
        if not fresh:
            ctx.errors.append(
                f"关键词 {keywords!r} 找到的 {len(candidates)} 个候选都已尝试过"
            )
            ctx.log_stage("registry", "all candidates tried")
            return tools

        ctx.candidates = fresh
        ctx.log_stage(
            "registry",
            f"{len(fresh)} fresh (excluded {len(candidates) - len(fresh)} tried)",
        )
        return tools

    def _collect_keywords(self, ctx: Any) -> list:
        # 优先 capability，其次从 user_query 自动生成
        cap = (ctx.capability or "").strip()
        raw = [kw.strip() for kw in cap.split(",") if kw.strip()] if cap else []

        if not raw and ctx.user_query:
            try:
                from tools.mcp_discovery import _auto_generate_keywords
                raw = _auto_generate_keywords(ctx.user_query)
            except Exception:
                logger.exception("[registry] auto keyword failed")

        # intent_index 命中但未配置 → server 名作为关键词
        if ctx.candidates:
            extra = [
                c.get("server_name", "") for c in ctx.candidates[:3]
                if c.get("server_name")
            ]
            raw = extra + raw

        return raw