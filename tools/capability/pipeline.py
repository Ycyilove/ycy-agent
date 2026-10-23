"""L1→L2→L4 发现流水线。只处理已配置 server，不联网。

流程：
    1. IntentNormalizer.normalize(goal) → IntentFrame
    2. CapabilityIndex.covers(frame, tools) → 已覆盖则直接返回
    3. CapabilityIndex.search(frame) → 从已配置 server 的索引里检索
    4. TrustGate.evaluate → 准入
    5. 对 manager.configs 里已配置但未连接的 server，connect + discover
    6. 返回合并后的 tools（可能不变）

未配置的新 server 不在这里处理——由 search_and_connect_mcp 兜底。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


async def maybe_discover(
    goal: str,
    tools: Dict[str, Any],
    manager: Any,
    register_tool: Callable,
    unregister_tool: Callable,
    allowed_danger_levels: Optional[set],
    state: Any = None,
) -> Dict[str, Any]:
    # L1 归一化
    from tools.intent_frame.normalizer import IntentNormalizer
    frame = IntentNormalizer().normalize(goal)

    # L2 现有工具是否覆盖
    from tools.capability.index import CapabilityIndex
    idx = CapabilityIndex()
    if idx.covers(frame, tools):
        logger.info(
            "[dsh][pipeline] frame covered: domain=%s action=%s",
            frame.domain, frame.action,
        )
        return tools

    # L2 检索候选（只覆盖已配置 server）
    candidates = idx.search(frame, top_k=10)
    if not candidates:
        logger.info(
            "[dsh][pipeline] no local candidates: domain=%s action=%s "
            "(fallback to search_and_connect_mcp if LLM chooses)",
            frame.domain, frame.action,
        )
        return tools

    # L3 信任准入
    from tools.capability.trust import TrustGate
    gate = TrustGate()
    admitted = []
    for c in candidates:
        # 方案 C：只处理已配置 server
        if c.server_name not in manager.configs:
            logger.info(
                "[dsh][pipeline] skip unconfigured server: %s", c.server_name,
            )
            continue
        decision = gate.evaluate(c, frame)
        if decision.admitted:
            admitted.append(c)
        else:
            logger.info(
                "[dsh][pipeline] rejected %s: %s",
                c.server_name, decision.reason,
            )

    if not admitted:
        return tools

    # L4 连接 + 注册（只连已配置但未连接的）
    new_tools = dict(tools)
    for cand in admitted[:3]:
        srv = cand.server_name
        if srv in manager.connections:
            continue
        try:
            await manager.connect(srv)
            await manager.discover(srv, register_tool, unregister_tool)
            logger.info("[dsh][pipeline] connected configured server: %s", srv)

            from tools import get_registry
            reg = get_registry()
            allowed = allowed_danger_levels or {"safe", "medium", "high"}
            for name, meta in reg.get_all_metadata().items():
                if name.startswith("mcp__") and meta.danger_level in allowed:
                    new_tools[name] = meta
            break
        except Exception:
            logger.exception("[dsh][pipeline] connect %s failed", srv)
            continue

    return new_tools