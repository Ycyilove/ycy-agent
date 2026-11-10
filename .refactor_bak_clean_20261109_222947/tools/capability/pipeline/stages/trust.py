"""Stage 4：TrustGate 准入。

原 maybe_discover 的 L3：`TrustGate().evaluate(candidate, frame)`。
拒绝的候选写入 ctx.errors，不阻断后续。

兼容两种候选类型：
    - dict（local 模式，来自 capability_index 的 __dict__ 或 search_configs_multi）
    - RankedCandidate（registry 模式，由 RankStage 转换）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TrustStage:
    async def run(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.short_circuit or not ctx.candidates or ctx.frame is None:
            return tools

        try:
            from tools.capability.trust import TrustGate
            from tools.capability.types import RankedCandidate

            gate = TrustGate()
            admitted = []
            for c in ctx.candidates:
                # 统一转成 RankedCandidate
                candidate = self._to_ranked(c, RankedCandidate)

                # 只处理已配置 server（local 模式）
                if ctx.mode == "local":
                    if not ctx.manager or candidate.server_name not in ctx.manager.configs:
                        ctx.log_stage(
                            "trust",
                            f"skip unconfigured: {candidate.server_name}",
                        )
                        continue

                decision = gate.evaluate(candidate, ctx.frame)
                if decision.admitted:
                    admitted.append(c)
                else:
                    ctx.errors.append(
                        f"{candidate.server_name}: {decision.reason}"
                    )
                    ctx.log_stage(
                        "trust",
                        f"rejected {candidate.server_name}: {decision.reason}",
                    )
            ctx.candidates = admitted
            ctx.log_stage("trust", f"{len(admitted)} admitted")
        except Exception:
            logger.exception("[trust] failed")
        return tools

    @staticmethod
    def _to_ranked(c: Any, RankedCandidateCls) -> Any:
        """统一转成 RankedCandidate。"""
        if isinstance(c, RankedCandidateCls):
            return c
        # dict 路径
        return RankedCandidateCls(
            server_name=c.get("local_name") or c.get("registry_name") or c.get("server_name") or "",
            top_tools=c.get("top_tools") or [],
            score=c.get("score", 0.0) or 0.0,
            intent_features=c.get("intent_features") or {},
            registry_entry=c.get("registry_entry") or c,
        )