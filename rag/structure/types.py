"""结构层数据契约：Block / Record / Cell。

不依赖任何文档类型：表格、列表、段落都归一到 Record，
记录是检索的最小单元，字段是关系载体。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class BlockKind(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"


# 通用语法类别：与语言/领域无关，可内置。
# 领域语义类别（person/team/award/...）不在此枚举里，
# 由 FieldSchema 注入，字符串形式存储在 Cell.field_type。
SYNTAX_TYPES = frozenset({"free_text", "number", "date", "code"})
DEFAULT_FIELD_TYPE = "free_text"


@dataclass
class Cell:
    """记录中的一个字段。

    field_type 是字符串：默认是通用语法类别，
    领域类型由 FieldSchema 判定后写入。
    """
    text: str
    col_index: int
    col_name: Optional[str] = None
    field_type: str = DEFAULT_FIELD_TYPE
    type_source: str = "default"   # default / header / value / dict

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "col_index": self.col_index,
            "col_name": self.col_name,
            "field_type": self.field_type,
            "type_source": self.type_source,
        }


@dataclass
class Record:
    """一条可检索记录。表格行 / 列表项 / 段落均可归一为此。"""
    record_id: str
    block_kind: BlockKind
    cells: List[Cell]
    source_filename: str
    header: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    source_chunk_id: Optional[str] = None

    def to_index_text(self) -> str:
        """记录级检索文本：字段名: 字段值 | ..."""
        parts = []
        for cell in self.cells:
            if cell.col_name:
                parts.append(f"{cell.col_name}: {cell.text}")
            else:
                parts.append(cell.text)
        return " | ".join(p for p in parts if p)

    def field_value(self, kind: FieldKind) -> Optional[str]:
        for cell in self.cells:
            if cell.field_kind == kind and cell.text.strip():
                return cell.text.strip()
        return None

    def entity_pairs(self) -> List[tuple]:
        """同记录内的字段值两两配对。

        过滤条件从"非 FREE_TEXT"改为"非 free_text 字符串"，
        领域类型是否算实体由调用方（schema）决定，
        这里只排除明确无信息量的 free_text。
        """
        values = [
            (c.field_type, c.text.strip())
            for c in self.cells
            if c.text.strip() and c.field_type != DEFAULT_FIELD_TYPE
        ]
        return [
            (values[i], values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "block_kind": self.block_kind.value,
            "cells": [c.to_dict() for c in self.cells],
            "source_filename": self.source_filename,
            "header": self.header,
            "page_numbers": self.page_numbers,
            "headings": self.headings,
            "source_chunk_id": self.source_chunk_id,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Record":
        return cls(
            record_id=value["record_id"],
            block_kind=BlockKind(value.get("block_kind", "table")),
            cells=[Cell(**c) for c in value.get("cells", [])],
            source_filename=value.get("source_filename", ""),
            header=value.get("header", []),
            page_numbers=value.get("page_numbers", []),
            headings=value.get("headings", []),
            source_chunk_id=value.get("source_chunk_id"),
        )