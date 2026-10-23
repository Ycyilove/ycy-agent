from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RankedCandidate:
    server_name: str
    top_tools: List[str] = field(default_factory=list)
    score: float = 0.0
    intent_features: Dict[str, str] = field(default_factory=dict)
    registry_entry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustDecision:
    server: str
    admitted: bool
    reason: str
    downgraded: List[str] = field(default_factory=list)