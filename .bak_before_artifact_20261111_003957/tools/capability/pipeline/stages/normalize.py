"""Stage 1：goal → IntentFrame。

原 maybe_discover 的 L1：`IntentNormalizer().normalize(goal)`。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class NormalizeStage:
    """把用户输入归一化成 IntentFrame。"""

    async def run(self, ctx: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
        from tools.intent_frame.normalizer import IntentNormalizer

        goal = ctx.user_query or ""
        try:
            ctx.frame = IntentNormalizer().normalize(goal)
            ctx.log_stage(
                "normalize",
                f"domain={ctx.frame.domain} action={ctx.frame.action} "
                f"freshness={ctx.frame.freshness}",
            )
        except Exception:
            logger.exception("[normalize] failed")
            ctx.frame = None
        return tools