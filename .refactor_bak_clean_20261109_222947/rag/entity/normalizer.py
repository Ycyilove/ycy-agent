"""实体归一化：全半角、大小写、连字符、空白。"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_HYPHEN_RE = re.compile(r"[-\u2010-\u2015_]")


def normalize_entity(surface: str) -> str:
    """归一化实体串，用于倒排和别名匹配。

    不做繁简转换（需要额外依赖），但覆盖最常见的等价形式：
        "GZHU-3" / "GZHU3" / "g z h u-3" → "gzhu3"
    """
    if not surface:
        return ""
    text = unicodedata.normalize("NFKC", surface)
    text = _WS_RE.sub("", text)
    text = _HYPHEN_RE.sub("", text)
    return text.lower()