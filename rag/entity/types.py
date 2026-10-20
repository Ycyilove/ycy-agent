"""实体 mention 的统一表示。

query 侧和文档侧共用，替代原先 List[str] 的裸字符串。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EntityMention:
    surface: str
    normalized: str
    kind: str          # person / team / award / number / date / org / event / other
    source: str        # pos / regex / dict / field
    start: int
    end: int
    conf: float
    chunk_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "normalized": self.normalized,
            "kind": self.kind,
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "conf": self.conf,
            "chunk_id": self.chunk_id,
        }