"""DiscoveryPipeline：7 阶段编排。

用法：
    pipeline = DiscoveryPipeline([
        NormalizeStage(), CoverStage(), LocalSearchStage(),
        RegistryStage(), RankStage(), TrustStage(), ConnectStage(),
    ])
    tools = await pipeline.run(ctx, tools)

行为：
    - 任一阶段返回的 tools 与输入不同 → 下游阶段继续处理新 tools
    - ctx.short_circuit = True → 跳过后续所有阶段
    - ctx.result 有值 → 可直接返回
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DiscoveryPipeline:
    def __init__(self, stages: List[Any]):
        self._stages = list(stages)

    async def run(
        self, ctx: Any, tools: Dict[str, Any],
    ) -> Dict[str, Any]:
        for stage in self._stages:
            name = type(stage).__name__
            if ctx.short_circuit and name != "ConnectStage":
                # 短路：只让 Connect 处理"已有 result 直接返回"
                continue
            try:
                tools = await stage.run(ctx, tools)
            except Exception:
                logger.exception(
                    "[pipeline] stage %s raised", name,
                )
                continue
            logger.debug(
                "[pipeline] stage %s done: tools=%d result=%r short=%r",
                name, len(tools), ctx.result is not None, ctx.short_circuit,
            )
        return tools


# ── 两条入口的默认 pipeline ──

def make_local_pipeline():
    """maybe_discover 用：只处理已配置 server。"""
    from .stages.normalize import NormalizeStage
    from .stages.cover import CoverStage
    from .stages.local_search import LocalSearchStage
    from .stages.trust import TrustStage
    from .stages.connect import ConnectStage

    return DiscoveryPipeline([
        NormalizeStage(),
        CoverStage(),
        LocalSearchStage(),
        TrustStage(),
        ConnectStage(),
    ])


def make_registry_pipeline():
    """search_and_connect_mcp 用：含 Registry + Rank。"""
    from .stages.normalize import NormalizeStage
    from .stages.cover import CoverStage
    from .stages.local_search import LocalSearchStage
    from .stages.registry import RegistryStage
    from .stages.rank import RankStage
    from .stages.trust import TrustStage
    from .stages.connect import ConnectStage

    return DiscoveryPipeline([
        NormalizeStage(),
        CoverStage(),
        LocalSearchStage(),
        RegistryStage(),
        RankStage(),
        TrustStage(),
        ConnectStage(),
    ])