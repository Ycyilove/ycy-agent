"""TrustGate：纯确定性准入。不调 LLM。"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from .types import RankedCandidate, TrustDecision

logger = logging.getLogger(__name__)


class TrustGate:
    """决定 server 能否自动连接。"""

    def evaluate(
        self, candidate: RankedCandidate, frame,
    ) -> TrustDecision:
        # 1. 领域一致
        cand_domain = (candidate.intent_features or {}).get("domain")
        if cand_domain and cand_domain != frame.domain:
            return TrustDecision(
                server=candidate.server_name,
                admitted=False,
                reason=f"domain mismatch: {cand_domain} vs {frame.domain}",
            )

        # 2. 描述注入检测（离线建索引时标记）
        entry = candidate.registry_entry or {}
        if entry.get("injection_suspected"):
            return TrustDecision(
                server=candidate.server_name,
                admitted=False,
                reason="tool description injection suspected",
            )

        # 3. 凭据预检：需要 header 且环境变量缺失 → 需人工审批
        headers = entry.get("headers") or []
        if headers:
            missing = []
            for h in headers:
                if not isinstance(h, dict):
                    continue
                env_name = h.get("env")
                if env_name and not os.getenv(env_name):
                    missing.append(env_name)
            if missing:
                return TrustDecision(
                    server=candidate.server_name,
                    admitted=False,
                    reason=f"missing credentials: {missing}",
                )

        # 4. 只读校验：若 registry_entry 明确标注了 has_write_tools 且无只读工具
        if entry.get("has_write_tools") and not entry.get("has_read_only"):
            return TrustDecision(
                server=candidate.server_name,
                admitted=False,
                reason="only write tools, needs approval",
            )

        return TrustDecision(
            server=candidate.server_name,
            admitted=True,
            reason="ok",
        )