"""字段类型判定：schema 驱动，默认零领域假设。

设计：
    - 内置只认语法类别：number / date / code / free_text
    - 领域类型（person/team/award/...）全部由 FieldSchema 注入，
      不配置就不识别，不猜
    - 判定优先级：列名命中 schema > 值命中 schema 的 pattern/dict > 语法类别 > free_text
    - 不做"2-4 汉字即人名"这类硬编码，人名交给实体抽取器的 POS 通道
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern

from .types import DEFAULT_FIELD_TYPE, SYNTAX_TYPES

logger = logging.getLogger(__name__)

# 通用语法模式：跨语言成立，可内置
_DATE_RE = re.compile(r"^\d{4}[-/年]\d{1,2}([-/月]\d{1,2}日?)?$")
_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")
# code：短标识符，不含语义。只判断"像标识符"，不判断"是团队编号"
_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,31}$")


@dataclass
class FieldTypeDef:
    """一种字段类型的定义，由 schema 提供。"""
    name: str
    header_hints: List[str] = field(default_factory=list)
    value_patterns: List[str] = field(default_factory=list)   # 正则字符串
    dictionary: List[str] = field(default_factory=list)        # 精确匹配词
    is_entity: bool = True       # 是否进实体倒排
    is_anchor: bool = False      # 是否可作为检索锚点


@dataclass
class FieldSchema:
    """字段类型 schema。

    默认只含语法类别，不含任何领域类型。
    领域类型通过 JSON 配置注入。
    """
    types: Dict[str, FieldTypeDef] = field(default_factory=dict)

    # ── 构建 ──

    def __post_init__(self):
        self._compiled: Dict[str, List[Pattern]] = {}
        self._dict_sets: Dict[str, set] = {}
        for name, td in self.types.items():
            self._compiled[name] = [re.compile(p) for p in td.value_patterns]
            self._dict_sets[name] = set(td.dictionary)

    @classmethod
    def empty(cls) -> "FieldSchema":
        return cls(types={})

    @classmethod
    def from_dict(cls, raw: dict) -> "FieldSchema":
        types = {}
        for name, spec in (raw.get("types") or {}).items():
            types[name] = FieldTypeDef(
                name=name,
                header_hints=spec.get("header_hints", []),
                value_patterns=spec.get("value_patterns", []),
                dictionary=spec.get("dictionary", []),
                is_entity=spec.get("is_entity", True),
                is_anchor=spec.get("is_anchor", False),
            )
        return cls(types=types)

    @classmethod
    def load(cls, path: str) -> "FieldSchema":
        p = Path(path)
        if not p.exists():
            logger.info("[dsh][schema] %s 不存在，使用空 schema", path)
            return cls.empty()
        try:
            with open(p, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (OSError, ValueError):
            logger.exception("[dsh][schema] 加载 %s 失败，使用空 schema", path)
            return cls.empty()

    # ── 查询 ──

    def is_entity_type(self, field_type: str) -> bool:
        if field_type in SYNTAX_TYPES and field_type != DEFAULT_FIELD_TYPE:
            return True
        td = self.types.get(field_type)
        return bool(td and td.is_entity)

    def anchor_types(self) -> set:
        return {name for name, td in self.types.items() if td.is_anchor}

    # ── 判定 ──

    def resolve(self, header: Optional[str], value: str) -> tuple:
        """返回 (field_type, type_source)。

        优先级：
            1. 列名命中 schema header_hints → 领域类型
            2. 值命中 schema value_patterns / dictionary → 领域类型
            3. 值命中通用语法模式 → number / date / code
            4. free_text
        """
        h = (header or "").strip().lower()
        if h:
            for name, td in self.types.items():
                if any(hint.lower() in h for hint in td.header_hints):
                    return name, "header"

        v = (value or "").strip()
        if v:
            for name, td in self.types.items():
                if name in self._dict_sets and v in self._dict_sets[name]:
                    return name, "dict"
                for pattern in self._compiled.get(name, []):
                    if pattern.fullmatch(v) or pattern.search(v):
                        return name, "value"

        if _DATE_RE.fullmatch(v):
            return "date", "value"
        if _NUMBER_RE.fullmatch(v):
            return "number", "value"
        if _CODE_RE.fullmatch(v) and not v.isdigit():
            return "code", "value"
        return DEFAULT_FIELD_TYPE, "default"


# 模块级单例，供 service 注入
_default_schema = FieldSchema.empty()


def get_default_schema() -> FieldSchema:
    return _default_schema


def set_default_schema(schema: FieldSchema) -> None:
    global _default_schema
    _default_schema = schema