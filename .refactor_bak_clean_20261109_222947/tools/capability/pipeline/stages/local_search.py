"""Stage 3：从已配置 server 的索引里检索候选。

原 maybe_discover 的 L2 检索 + search_and_connect_mcp 的 capability_tags + intent_index。

两种模式：
    - mode=local：查 `capability_index`（已配置 server 的完整索引）
    - mode=registry：先查 `capability_tags`，再查 `mcp_intent_index`

产出类型：dict 列表（交给 RankStage 或 ConnectStage._connect_local 转换）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LocalSearchStage:
    async def run(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.short_circuit or ctx.frame is None:
            return tools

        if ctx.mode == "local":
            return self._run_local(ctx, tools)
        return self._run_registry(ctx, tools)

    def _run_local(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from tools.capability.index import CapabilityIndex
            idx = CapabilityIndex()
            ctx.candidates = idx.search(ctx.frame, top_k=10)
            ctx.log_stage(
                "local_search(local)",
                f"{len(ctx.candidates)} candidates from capability_index",
            )
        except Exception:
            logger.exception("[local_search] capability_index failed")
        return tools

    def _run_registry(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        # 3a. capability_tags 命中（零 LLM）→ 直接返回结果，短路
        matched = self._match_capability_tags(ctx)
        if matched:
            ctx.result = matched
            ctx.short_circuit = True
            ctx.log_stage("local_search(capability_tags)", "hit")
            return tools

        # 3b. intent_index 语义检索
        try:
            from tools.mcp_intent_index import search_intents
            from tools.intent_frame.normalizer import IntentNormalizer
            from tools.capability.types import RankedCandidate

            _frame = IntentNormalizer().normalize(ctx.user_query or "")
            hits = search_intents(
                ctx.user_query or "", top_k=10, frame=_frame,
            )
            if hits:
                # 转 RankedCandidate，并标记为"本地已配置"（不需 registry）
                converted = [
                    RankedCandidate(
                        server_name=h.get("server_name", ""),
                        top_tools=[
                            t.get("tool_name", "")
                            for t in (h.get("matched_tools") or [])
                        ],
                        score=h.get("top_score", 0.0),
                        intent_features={},
                        registry_entry={"local_name": h.get("server_name", "")},
                    )
                    for h in hits
                ]
                ctx.candidates = converted
                ctx.short_circuit = True  # 不跑 Registry/Rank
                ctx.log_stage(
                    "local_search(intent_index)",
                    f"{len(converted)} server candidates, short-circuit",
                )
                return tools
            ctx.log_stage("local_search(intent_index)", "no hit")
        except Exception:
            logger.exception("[local_search] intent_index failed")
        return tools

    def _match_capability_tags(self, ctx: Any) -> Any:
        """声明式能力标签匹配：intent.tags ∩ spec.capability_tags。"""
        if not ctx.manager or not hasattr(ctx.manager, "tools"):
            return None
        try:
            from tools.semantics import classify_intent
            _intent = classify_intent(ctx.user_query or "")
            _wanted = set(_intent.tags) | set(_intent.secondary)
            if not _wanted:
                return None
            _matched = []
            for _spec in ctx.manager.tools.values():
                _caps = set(getattr(_spec, "capability_tags", ()) or ())
                _overlap = _caps & _wanted
                if _overlap:
                    _matched.append((len(_overlap), _spec))
            if not _matched:
                return None
            _matched.sort(key=lambda x: -x[0])
            _best = _matched[0][1]
            _srv = _best.server_name
            return {
                "success": True,
                "server": _srv,
                "description": (
                    f"(capability_tags match: "
                    f"{sorted(set(getattr(_best, 'capability_tags', ()) or ()) & _wanted)})"
                ),
                "transport": ctx.manager.configs[_srv].transport,
                "url": ctx.manager.configs[_srv].url,
                "tools": [
                    s.public_name for s in ctx.manager.tools.values()
                    if s.server_name == _srv
                ],
                "source": "capability_tags",
            }
        except Exception:
            logger.exception("[local_search] capability_tags match failed")
            return None