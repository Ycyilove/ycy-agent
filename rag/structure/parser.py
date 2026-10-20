"""从 Docling Document 提取 Record。

优先用 Docling 的 tables/lists 结构；取不到时用 chunk 文本的
"连续多行同列数" 启发式兜底。段落不在这里处理，仍走 chunker。
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Iterable, List, Optional

from .field_classifier import FieldSchema, get_default_schema
from .types import BlockKind, Cell, DEFAULT_FIELD_TYPE, Record

logger = logging.getLogger(__name__)

# 兜底启发式：一行被 2+ 个空白/竖线/制表符分隔，视为表格行
_TABLE_ROW_RE = re.compile(r"\s{2,}|\t|\|")


def _table_to_records(
    table,
    source_filename: str,
    headings: List[str],
    page_numbers: List[int],
    schema: FieldSchema,
) -> List[Record]:
    """Docling TableItem → Record 列表。

    依赖 docling_core 的 table.data.grid（row/col 二维结构）：
        data.grid[row][col].text
    若版本字段不同，改这里即可，不影响调用方。
    """
    records: List[Record] = []
    try:
        grid = table.data.grid
    except AttributeError:
        logger.warning("[dsh][struct] table.data.grid 不可用，跳过该表")
        return records

    if not grid:
        return records

    header_row = grid[0]
    header = [getattr(c, "text", "") or "" for c in header_row]

    for row in grid[1:]:
        cells: List[Cell] = []
        for col_idx, cell in enumerate(row):
            text = (getattr(cell, "text", "") or "").strip()
            if not text:
                continue
            col_name = header[col_idx] if col_idx < len(header) else None
            field_type, source = schema.resolve(col_name, text)
            cells.append(
                Cell(
                    text=text,
                    col_index=col_idx,
                    col_name=col_name,
                    field_type=field_type,
                    type_source=source,
                )
            )
        if not cells:
            continue
        records.append(
            Record(
                record_id=uuid.uuid4().hex,
                block_kind=BlockKind.TABLE,
                cells=cells,
                source_filename=source_filename,
                header=header,
                headings=list(headings),
                page_numbers=list(page_numbers),
            )
        )
    return records


def _list_to_records(
    list_item,
    source_filename: str,
    headings: List[str],
    page_numbers: List[int],
    schema: FieldSchema,
) -> List[Record]:
    """Docling ListItem → Record。

    列表项若形如 "key: value"，拆成 key 和 value 两个 Cell；
    否则整项一个 Cell。key 本身是字段名，不参与类型判定，
    统一标为 free_text，避免被误当成实体进倒排。
    """
    records: List[Record] = []
    try:
        items = list_item.items
    except AttributeError:
        return records

    for item in items:
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        cells: List[Cell] = []
        if ":" in text or "：" in text:
            sep = ":" if ":" in text else "："
            key, value = text.split(sep, 1)
            key, value = key.strip(), value.strip()
            if key and value:
                val_type, val_source = schema.resolve(key, value)
                cells = [
                    Cell(
                        text=key,
                        col_index=0,
                        col_name=None,
                        field_type=DEFAULT_FIELD_TYPE,
                        type_source="default",
                    ),
                    Cell(
                        text=value,
                        col_index=1,
                        col_name=key,
                        field_type=val_type,
                        type_source=val_source,
                    ),
                ]
        if not cells:
            field_type, source = schema.resolve(None, text)
            cells = [
                Cell(
                    text=text,
                    col_index=0,
                    col_name=None,
                    field_type=field_type,
                    type_source=source,
                )
            ]
        records.append(
            Record(
                record_id=uuid.uuid4().hex,
                block_kind=BlockKind.LIST,
                cells=cells,
                source_filename=source_filename,
                headings=list(headings),
                page_numbers=list(page_numbers),
            )
        )
    return records


def _fallback_table_from_chunk(
    text: str,
    source_filename: str,
    headings: List[str],
    page_numbers: List[int],
    chunk_id: Optional[str],
    schema: FieldSchema,
) -> List[Record]:
    """Docling 没输出 table 时，从 chunk 文本兜底。

    只处理"整个 chunk 都是同构表格"的情况：连续 2 行以上、
    列数相同、用 2+ 空格/制表符/竖线分隔。
    混合内容的 chunk（表格 + 段落）不会被识别，
    宁可漏掉表格，也不要把叙事文本误切成记录。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return []

    parsed: List[List[str]] = []
    for ln in lines:
        parts = [p.strip() for p in _TABLE_ROW_RE.split(ln) if p.strip()]
        if len(parts) < 2:
            return []
        parsed.append(parts)

    col_count = len(parsed[0])
    if any(len(p) != col_count for p in parsed):
        return []

    header = parsed[0]
    records: List[Record] = []
    for row in parsed[1:]:
        cells: List[Cell] = []
        for col_idx, value in enumerate(row):
            col_name = header[col_idx] if col_idx < len(header) else None
            field_type, source = schema.resolve(col_name, value)
            cells.append(
                Cell(
                    text=value,
                    col_index=col_idx,
                    col_name=col_name,
                    field_type=field_type,
                    type_source=source,
                )
            )
        records.append(
            Record(
                record_id=uuid.uuid4().hex,
                block_kind=BlockKind.TABLE,
                cells=cells,
                source_filename=source_filename,
                header=header,
                headings=list(headings),
                page_numbers=list(page_numbers),
                source_chunk_id=chunk_id,
            )
        )
    return records


def parse_structure(
    doc,
    docling_chunks: Iterable,
    source_filename: str,
    schema: Optional[FieldSchema] = None,
) -> List[Record]:
    """统一入口：Docling Document + chunks → Record 列表。

    段落不在这里产生 Record（仍由 chunker 入库），
    只为表格/列表建 Record，避免重复索引。

    schema 决定字段类型的判定规则；不传时使用空 schema，
    即只识别 number / date / code 这类通用语法类别，
    不猜测 person / team / award 等任何领域类型。
    """
    if schema is None:
        schema = get_default_schema()

    records: List[Record] = []

    for table in getattr(doc, "tables", []) or []:
        prov = getattr(table, "prov", None) or []
        pages = [
            getattr(p, "page_no", None)
            for p in prov
            if getattr(p, "page_no", None)
        ]
        records.extend(
            _table_to_records(table, source_filename, [], pages, schema)
        )

    for lst in getattr(doc, "lists", []) or []:
        prov = getattr(lst, "prov", None) or []
        pages = [
            getattr(p, "page_no", None)
            for p in prov
            if getattr(p, "page_no", None)
        ]
        records.extend(
            _list_to_records(lst, source_filename, [], pages, schema)
        )

    if not records:
        # Docling 没给出结构，用 chunk 文本兜底
        for chunk in docling_chunks:
            headings = list(
                getattr(getattr(chunk, "meta", None), "headings", None) or []
            )
            pages = list(
                getattr(getattr(chunk, "meta", None), "page_numbers", None) or []
            )
            records.extend(
                _fallback_table_from_chunk(
                    getattr(chunk, "text", "") or "",
                    source_filename,
                    headings,
                    pages,
                    None,
                    schema,
                )
            )

    logger.info(
        "[dsh][struct] %s: %d 条记录（table=%d, list=%d）",
        source_filename,
        len(records),
        sum(1 for r in records if r.block_kind == BlockKind.TABLE),
        sum(1 for r in records if r.block_kind == BlockKind.LIST),
    )
    return records