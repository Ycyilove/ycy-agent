#!/usr/bin/env python3
"""测试知识库文档的实体抽取齐全性与正确性。

流程：
    1. 首次解析 PDF，把 Docling 结果（chunks / tables / lists）缓存到根目录
    2. 之后只读缓存，不再重新 OCR
    3. 默认 schema 和配置 schema 各跑一遍 parse_structure + EntityExtractor
    4. 输出全部实体、覆盖率检查、仅靠列名判类型的清单

用法：
    python scripts/test_entity_extraction.py
    python scripts/test_entity_extraction.py --rebuild-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PDF_NAME = "第十届“中国高校计算机大赛-团体程序设计天梯赛”全国总决赛获奖名单.pdf"
CACHE_NAME = "parsed_document_cache.json"
REPORT_NAME = "entity_report.json"

# 测试用 schema：模拟用户为获奖名单注入的配置，不随代码发布
TEST_SCHEMA = {
    "types": {
        "person": {
            "header_hints": ["姓名", "名字", "队员", "选手", "name"],
            "is_entity": True,
            "is_anchor": True,
        },
        "team": {
            "header_hints": ["团队", "队伍", "队名", "team"],
            "value_patterns": [r"[A-Z]{2,}[-\s]?\d{1,4}"],
            "is_entity": True,
            "is_anchor": True,
        },
        "award": {
            "header_hints": ["奖项", "等级", "名次", "奖"],
            "is_entity": True,
            "is_anchor": True,
        },
        "number": {
            "header_hints": ["分数", "得分", "成绩", "积分"],
            "is_entity": True,
        },
        "school": {
            "header_hints": ["学校", "单位", "院校"],
            "is_entity": True,
        },
    }
}

logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


# ── Fake Docling 对象：只保留 parse_structure 用到的属性 ──


class _Prov:
    def __init__(self, page_no):
        self.page_no = page_no


class _FakeCell:
    def __init__(self, text):
        self.text = text


class _FakeGrid:
    def __init__(self, grid):
        self.grid = grid


class _FakeTable:
    def __init__(self, grid, pages):
        self.data = _FakeGrid([[ _FakeCell(t) for t in row] for row in grid])
        self.prov = [_Prov(p) for p in pages]


class _FakeListItem:
    def __init__(self, text):
        self.text = text


class _FakeList:
    def __init__(self, items, pages):
        self.items = [_FakeListItem(t) for t in items]
        self.prov = [_Prov(p) for p in pages]


class _FakeDoc:
    def __init__(self, tables, lists):
        self.tables = tables
        self.lists = lists


class _FakeChunkMeta:
    def __init__(self, headings, pages):
        self.headings = headings
        self.page_numbers = pages


class _FakeChunk:
    def __init__(self, text, headings, pages):
        self.text = text
        self.meta = _FakeChunkMeta(headings, pages)


# ── 缓存构建 / 加载 ──


def build_cache(pdf_path: Path, cache_path: Path) -> dict:
    from rag.docling_parser import DoclingParser

    parser = DoclingParser()
    chunks = parser.parse_to_chunks(str(pdf_path))
    doc = parser.last_document

    cache: dict = {
        "source": pdf_path.name,
        "chunks": [],
        "tables": [],
        "lists": [],
    }

    for c in chunks:
        cache["chunks"].append({
            "text": getattr(c, "text", "") or "",
            "headings": list(getattr(c.meta, "headings", None) or []),
            "page_numbers": list(getattr(c.meta, "page_numbers", None) or []),
        })

    if doc is not None:
        for t in getattr(doc, "tables", []) or []:
            try:
                grid = [
                    [(getattr(cell, "text", "") or "") for cell in row]
                    for row in t.data.grid
                ]
            except AttributeError:
                continue
            pages = [
                getattr(p, "page_no", None)
                for p in (getattr(t, "prov", None) or [])
            ]
            cache["tables"].append({
                "grid": grid,
                "page_numbers": [p for p in pages if p],
            })

        for lst in getattr(doc, "lists", []) or []:
            try:
                items = [(getattr(it, "text", "") or "") for it in lst.items]
            except AttributeError:
                continue
            pages = [
                getattr(p, "page_no", None)
                for p in (getattr(lst, "prov", None) or [])
            ]
            cache["lists"].append({
                "items": items,
                "page_numbers": [p for p in pages if p],
            })

    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache


def load_cache(cache_path: Path) -> dict:
    return json.loads(cache_path.read_text(encoding="utf-8"))


def rebuild_from_cache(cache: dict):
    doc = _FakeDoc(
        tables=[
            _FakeTable(t["grid"], t.get("page_numbers", []))
            for t in cache.get("tables", [])
        ],
        lists=[
            _FakeList(l["items"], l.get("page_numbers", []))
            for l in cache.get("lists", [])
        ],
    )
    chunks = [
        _FakeChunk(c["text"], c.get("headings", []), c.get("page_numbers", []))
        for c in cache.get("chunks", [])
    ]
    return doc, chunks


# ── 抽取 ──


def run_extraction(doc, chunks, filename: str, schema, extractor) -> tuple:
    """返回 (records, mentions, cell_mentions)。

    cell_mentions: (record_id, col_index) -> List[EntityMention]
    """
    from rag.structure.parser import parse_structure

    records = parse_structure(doc, chunks, filename, schema=schema)

    mentions: List = []
    cell_mentions: Dict[tuple, List] = {}

    for rec in records:
        for cell in rec.cells:
            ms = extractor.extract(cell.text, chunk_id=rec.record_id)
            cell_mentions[(rec.record_id, cell.col_index)] = ms
            mentions.extend(ms)

    for ch in chunks:
        mentions.extend(extractor.extract(ch.text))

    return records, mentions, cell_mentions


# ── 报告 ──


def _value_matchable(schema, field_type: str, text: str) -> bool:
    """cell 的值是否能被该类型的 pattern/dict 识别。

    只靠列名判类型、值本身不匹配的 cell 不算 extractor 的漏抽，
    因为 extractor 没有列名上下文。
    """
    td = schema.types.get(field_type)
    if td is None:
        return False
    if text in td.dictionary:
        return True
    for p in td.value_patterns:
        try:
            if re.fullmatch(p, text) or re.search(p, text):
                return True
        except re.error:
            continue
    return False


def build_report(name: str, records, mentions, cell_mentions, schema) -> dict:
    from rag.entity.normalizer import normalize_entity

    declared = {t for t, td in schema.types.items() if td.is_entity}

    by_kind: Dict[str, set] = defaultdict(set)
    for m in mentions:
        by_kind[m.kind].add(m.surface)

    missing: List[dict] = []
    header_only: List[dict] = []

    for rec in records:
        for cell in rec.cells:
            if cell.field_type not in declared:
                continue
            ms = cell_mentions.get((rec.record_id, cell.col_index), [])
            key = normalize_entity(cell.text)
            hit = any(
                m.normalized == key and m.kind == cell.field_type
                for m in ms
            )
            if hit:
                continue
            entry = {
                "record_id": rec.record_id,
                "col_name": cell.col_name,
                "text": cell.text,
                "field_type": cell.field_type,
                "type_source": cell.type_source,
                "mentions": [f"{m.surface}/{m.kind}" for m in ms],
            }
            if _value_matchable(schema, cell.field_type, cell.text):
                missing.append(entry)
            else:
                header_only.append(entry)

    return {
        "name": name,
        "records": len(records),
        "mentions": len(mentions),
        "by_kind": {k: sorted(v) for k, v in by_kind.items()},
        "missing": missing,
        "header_only": header_only,
    }


def print_report(r: dict) -> None:
    print(f"\n--- {r['name']} ---")
    print(f"record 数: {r['records']}")
    print(f"mention 总数: {r['mentions']}")
    for kind in sorted(r["by_kind"]):
        items = r["by_kind"][kind]
        print(f"  {kind}: {len(items)} 个")
        print(f"    示例: {', '.join(items[:10])}")
    print(f"覆盖率 missing（值可匹配但 extractor 没抽到）: {len(r['missing'])}")
    for item in r["missing"][:10]:
        print(
            f"  record={item['record_id'][:8]} "
            f"col={item['col_name']!r} text={item['text']!r} "
            f"type={item['field_type']} mentions={item['mentions']}"
        )
    print(f"仅靠列名判类型（extractor 无列名上下文，不计 missing）: {len(r['header_only'])}")
    for item in r["header_only"][:10]:
        print(
            f"  record={item['record_id'][:8]} "
            f"col={item['col_name']!r} text={item['text']!r} "
            f"type={item['field_type']}"
        )


# ── 主流程 ──


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="强制重新解析 PDF，覆盖缓存",
    )
    args = ap.parse_args()

    pdf_path = _ROOT / PDF_NAME
    cache_path = _ROOT / CACHE_NAME
    report_path = _ROOT / REPORT_NAME

    if not pdf_path.exists():
        print(f"[✗] PDF 不存在: {pdf_path}")
        sys.exit(1)

    if cache_path.exists() and not args.rebuild_cache:
        print(f"[1] 读取缓存: {cache_path.name}")
        cache = load_cache(cache_path)
    else:
        print(f"[1] 解析 PDF（首次，耗时较长）: {pdf_path.name}")
        cache = build_cache(pdf_path, cache_path)
        print(f"    已缓存: {cache_path.name}")

    doc, chunks = rebuild_from_cache(cache)
    print(
        f"[2] 重建: chunks={len(chunks)}, "
        f"tables={len(doc.tables)}, lists={len(doc.lists)}"
    )

    from rag.entity.extractor import EntityExtractor
    from rag.structure.field_classifier import FieldSchema

    # ── 默认 schema ──
    print("\n[3] 默认 schema（无配置）")
    default_schema = FieldSchema.empty()
    default_ext = EntityExtractor()
    records_d, mentions_d, cell_mentions_d = run_extraction(
        doc, chunks, pdf_path.name, default_schema, default_ext,
    )
    report_d = build_report(
        "default", records_d, mentions_d, cell_mentions_d, default_schema,
    )

    # ── 配置 schema ──
    print("[4] 配置 schema（模拟用户注入）")
    cfg_schema = FieldSchema.from_dict(TEST_SCHEMA)
    cfg_ext = EntityExtractor(
        dictionary={
            term: name
            for name, td in cfg_schema.types.items()
            for term in td.dictionary
        },
        patterns={
            name: td.value_patterns[0]
            for name, td in cfg_schema.types.items()
            if td.value_patterns
        },
    )
    records_c, mentions_c, cell_mentions_c = run_extraction(
        doc, chunks, pdf_path.name, cfg_schema, cfg_ext,
    )
    report_c = build_report(
        "configured", records_c, mentions_c, cell_mentions_c, cfg_schema,
    )

    print_report(report_d)
    print_report(report_c)

    report_path.write_text(
        json.dumps(
            {"default": report_d, "configured": report_c},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[✓] 完整实体清单已保存: {report_path.name}")


if __name__ == "__main__":
    main()