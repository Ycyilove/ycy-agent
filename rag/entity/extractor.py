"""统一实体抽取：正则 + 词典 + POS。

设计：
    - 正则通道覆盖结构化实体（团队编号、奖项、日期、数字）
    - 词典通道覆盖领域词（可注入）
    - POS 通道覆盖中文命名实体（人名/地名/机构）
    - 三通道结果按 (normalized, kind) 去重，保留最长 span
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from .normalizer import normalize_entity
from .types import EntityMention

logger = logging.getLogger(__name__)

# 通用语法模式：跨语言成立，内置
_DATE_RE = re.compile(r"\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# jieba POS → 实体类型。POS 是通用语言信号，不算领域过拟合；
# nz（其他专名）单独标为 other，不强行归入领域类型。
_POS_KIND = {
    "nr": "person", "nrt": "person", "nrfg": "person", "nrf": "person",
    "ns": "org", "nt": "org", "nz": "other",
}


class EntityExtractor:
    def __init__(
        self,
        dictionary: Optional[Dict[str, str]] = None,
        patterns: Optional[Dict[str, str]] = None,
    ):
        """dictionary: {词: kind}；patterns: {kind: 正则字符串}。

        默认只启用通用语法模式（date/number）+ POS 通道。
        领域模式（team/award/...）必须由调用方注入，
        不注入就不识别，避免对单份文档过拟合。
        """
        self._dictionary = dictionary or {}
        self._dict_re = None
        if self._dictionary:
            terms = sorted(self._dictionary.keys(), key=len, reverse=True)
            self._dict_re = re.compile("|".join(re.escape(t) for t in terms))

        self._patterns: Dict[str, List[Pattern]] = {}
        for kind, pattern in (patterns or {}).items():
            try:
                self._patterns[kind] = [re.compile(pattern)]
            except re.error:
                logger.exception("[dsh][entity] 无效 pattern: %s=%r", kind, pattern)

    @classmethod
    def from_schema(cls, schema) -> "EntityExtractor":
        """从 FieldSchema 构造注入领域模式/词典的 extractor。"""
        patterns: Dict[str, str] = {}
        dictionary: Dict[str, str] = {}
        if schema and getattr(schema, "types", None):
            for name, td in schema.types.items():
                if td.value_patterns:
                    patterns[name] = "|".join(td.value_patterns)
                for word in td.dictionary:
                    dictionary[word] = name
        return cls(dictionary=dictionary, patterns=patterns)
        
    def extract(self, text: str, chunk_id: Optional[str] = None) -> List[EntityMention]:
        if not text or not text.strip():
            return []
        mentions: List[EntityMention] = []
        mentions.extend(self._syntax_channel(text, chunk_id))
        mentions.extend(self._custom_channel(text, chunk_id))
        mentions.extend(self._dict_channel(text, chunk_id))
        mentions.extend(self._pos_channel(text, chunk_id))
        return self._dedupe(mentions)

    # ── 通道 ──

    def _syntax_channel(self, text: str, chunk_id: Optional[str]) -> List[EntityMention]:
        """通用语法模式：date / number。不含任何领域类型。"""
        out: List[EntityMention] = []
        for pattern, kind, conf in (
            (_DATE_RE, "date", 0.85),
            (_NUMBER_RE, "number", 0.60),
        ):
            for m in pattern.finditer(text):
                out.append(self._make(m.group(0), kind, "syntax", m.start(), m.end(), conf, chunk_id))
        return out

    def _custom_channel(self, text: str, chunk_id: Optional[str]) -> List[EntityMention]:
        """调用方注入的领域模式。未注入时为空。"""
        out: List[EntityMention] = []
        for kind, patterns in self._patterns.items():
            for pattern in patterns:
                for m in pattern.finditer(text):
                    out.append(self._make(m.group(0), kind, "pattern", m.start(), m.end(), 0.90, chunk_id))
        return out

    def _dict_channel(self, text: str, chunk_id: Optional[str]) -> List[EntityMention]:
        if not self._dict_re:
            return []
        out: List[EntityMention] = []
        for m in self._dict_re.finditer(text):
            kind = self._dictionary.get(m.group(0), "other")
            out.append(self._make(m.group(0), kind, "dict", m.start(), m.end(), 0.95, chunk_id))
        return out

    def _pos_channel(self, text: str, chunk_id: Optional[str]) -> List[EntityMention]:
        try:
            import jieba.posseg as pseg
        except ImportError:
            return []
        out: List[EntityMention] = []
        offset = 0
        for word, flag in pseg.cut(text):
            kind = _POS_KIND.get(flag)
            if kind and len(word) >= 2:
                start = text.find(word, offset)
                if start >= 0:
                    out.append(self._make(word, kind, "pos", start, start + len(word), 0.70, chunk_id))
                    offset = start + len(word)
        return out

    # ── 工具 ──

    @staticmethod
    def _make(surface, kind, source, start, end, conf, chunk_id) -> EntityMention:
        return EntityMention(
            surface=surface,
            normalized=normalize_entity(surface),
            kind=kind,
            source=source,
            start=start,
            end=end,
            conf=conf,
            chunk_id=chunk_id,
        )

    @staticmethod
    def _dedupe(mentions: List[EntityMention]) -> List[EntityMention]:
        """同 (normalized, kind) 保留最长 span、最高置信度。"""
        best: Dict[tuple, EntityMention] = {}
        for m in mentions:
            key = (m.normalized, m.kind)
            cur = best.get(key)
            if cur is None or (m.end - m.start) > (cur.end - cur.start) or m.conf > cur.conf:
                best[key] = m
        return sorted(best.values(), key=lambda m: m.start)