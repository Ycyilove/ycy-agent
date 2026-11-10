"""Stage 6（仅 mode=registry）：启发式 + LLM 排序。

只在 local_search 完全没命中时跑（short_circuit=False 且 candidates 是 registry dict）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RankStage:
    async def run(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.mode != "registry":
            return tools
        if ctx.short_circuit or not ctx.candidates:
            return tools

        # 只处理 dict 候选（来自 RegistryStage）
        from tools.capability.types import RankedCandidate
        if isinstance(ctx.candidates[0], RankedCandidate):
            return tools
        if not isinstance(ctx.candidates[0], dict):
            logger.warning(
                "[rank] 未知候选类型: %s", type(ctx.candidates[0]).__name__,
            )
            return tools
        if "registry_name" not in ctx.candidates[0]:
            logger.warning(
                "[rank] dict 缺 registry_name，跳过排序: %r",
                ctx.candidates[0],
            )
            return tools

        try:
            from tools.mcp_discovery import (
                _heuristic_rank,
                _rank_candidates,
            )
        except Exception:
            logger.exception("[rank] import mcp_discovery failed")
            return tools

        query = ctx.user_query or ctx.capability or ""

        # 1. 启发式预排序
        try:
            ranked = _heuristic_rank(query, ctx.candidates)
        except Exception:
            logger.exception("[rank] heuristic failed")
            ranked = ctx.candidates
        if not ranked:
            ctx.errors.append("候选列表为空")
            return tools

        # 2. LLM 领域判断
        chosen = None
        try:
            chosen = await _rank_candidates(query, ranked)
        except Exception:
            logger.exception("[rank] LLM rank failed, fallback to first")
            chosen = ranked[0] if ranked else None
        if chosen is None:
            ctx.errors.append(
                f"候选 {len(ranked)} 个，但 LLM 判断没有一个领域匹配"
            )
            ctx.log_stage("rank", "LLM rejected all, fallback to first")
            chosen = ranked[0]

        # 3. 转 RankedCandidate
        ordered = [chosen] + [c for c in ranked if c is not chosen]
        converted = []
        for c in ordered[:10]:
            converted.append(self._to_ranked_candidate(c))

        ctx.candidates = converted
        ctx.log_stage(
            "rank",
            f"chosen: {chosen.get('registry_name', '?')} "
            f"(total {len(converted)})",
        )
        return tools

    @staticmethod
    def _to_ranked_candidate(c: Any) -> Any:
        from tools.capability.types import RankedCandidate
        if isinstance(c, RankedCandidate):
            return c
        return RankedCandidate(
            server_name=c.get("local_name") or c.get("registry_name") or "",
            top_tools=[],
            score=0.0,
            intent_features={},
            registry_entry=c,
        )