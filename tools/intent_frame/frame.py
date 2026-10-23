"""IntentFrame：意图归一化后的不可变结构。

L1 之后所有层只读 IntentFrame，看不到原始 prior_messages。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

Domain = Literal["news", "sports", "knowledge", "file", "code", "web", "unknown"]
Action = Literal["search", "read", "write", "compute", "browse"]
Freshness = Literal["realtime", "recent", "any"]


@dataclass(frozen=True)
class IntentFrame:
    raw: str
    domain: Domain = "unknown"
    action: Action = "search"
    entities: List[str] = field(default_factory=list)
    freshness: Freshness = "any"
    confidence: float = 0.0
    source: str = "rule"

    def covers_by(self, other: "IntentFrame") -> bool:
        """当前 frame 是否覆盖 other（用于判断现有工具是否够用）。"""
        if self.domain == "unknown" or other.domain == "unknown":
            return False
        return self.domain == other.domain and self.action == other.action