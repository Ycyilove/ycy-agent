"""Stage 2：现有工具是否覆盖 frame。

原 maybe_discover 的 L2：`CapabilityIndex().covers(frame, tools)`。
命中 → 短路，不再 discovery。

B5 起 `covers()` 用 `DOMAIN_TO_TAGS`（vocab.py），与 frame.domain 对齐。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CoverStage:
    async def run(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.frame is None:
            return tools

        try:
            from tools.capability.index import CapabilityIndex
            idx = CapabilityIndex()
            if idx.covers(ctx.frame, tools):
                ctx.short_circuit = True
                ctx.log_stage(
                    "cover",
                    f"frame covered: domain={ctx.frame.domain} "
                    f"action={ctx.frame.action}",
                )
        except Exception:
            logger.exception("[cover] failed")
        return tools